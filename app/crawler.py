"""Breadth-first crawl loop from section 5 of the build plan.

Discovery:  the homepage may be a marketing page — a keyword link scanner finds
            the listings/portfolio page even on a different subdomain.
Counting:   properties are counted DETERMINISTICALLY by their unique detail-page
            links (robust across site permutations), and reconciled with any
            explicit total or pagination total the LLM reads off the page.
"""

import asyncio
import logging
import os
from typing import List, Optional, Set

import httpx

from .classifier import classify_page
from .jina_client import (
    count_headings,
    extract_candidate_links,
    extract_detail_keys,
    extract_pagination_links,
    fetch_via_jina,
    normalize_url,
    strip_links_summary,
)
from .schemas import CheckResponse, PageClassification, PageCount

logger = logging.getLogger("property_checker.crawler")

EXACT_BASES = {"stated_total", "counted_items"}


class ClassificationError(Exception):
    """Raised when no page could be classified (e.g. bad OpenAI key/quota)."""


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class ListingPage:
    def __init__(
        self,
        url: str,
        llm_count: Optional[int],
        llm_basis: str,
        detail_keys: Set[str],
        heading_count: int,
    ):
        self.url = url
        self.llm_count = llm_count
        self.llm_basis = llm_basis
        self.detail_keys = detail_keys
        self.heading_count = heading_count


class PageResult:
    def __init__(
        self,
        url: str,
        classification: Optional[PageClassification],
        content: Optional[str],
        fetch_failed: bool,
        classify_failed: bool,
    ):
        self.url = url
        self.classification = classification
        self.content = content
        self.fetch_failed = fetch_failed
        self.classify_failed = classify_failed


async def _process_page(client: httpx.AsyncClient, page_url: str) -> PageResult:
    """Fetch + classify one page, capturing fetch/classify failures separately.

    Classification uses the main content only (not Jina's appended link summary).
    """
    content = await fetch_via_jina(client, page_url)
    if not content:
        return PageResult(page_url, None, None, fetch_failed=True, classify_failed=False)
    try:
        classification = await classify_page(page_url, strip_links_summary(content))
    except Exception as exc:  # API error (auth, rate limit, timeout, etc.)
        logger.warning("Classification failed for %s: %s", page_url, exc)
        return PageResult(page_url, None, content, fetch_failed=False, classify_failed=True)
    return PageResult(page_url, classification, content, fetch_failed=False, classify_failed=False)


async def check_properties(url: str) -> CheckResponse:
    max_pages = _int_env("MAX_PAGES_PER_SITE", 8)
    max_depth = _int_env("MAX_CRAWL_DEPTH", 2)

    start = normalize_url(url)
    visited: set = set()
    queue: List[str] = [start]
    listing_pages: List[ListingPage] = []
    had_fetch_failure = False
    classify_failures = 0
    classify_successes = 0
    depth = 0

    async with httpx.AsyncClient() as client:
        while queue and len(visited) < max_pages and depth <= max_depth:
            batch: List[str] = []
            for page_url in queue:
                page_url = normalize_url(page_url)
                if not page_url or page_url in visited:
                    continue
                visited.add(page_url)
                batch.append(page_url)
                if len(visited) >= max_pages:
                    break

            if not batch:
                break

            results = await asyncio.gather(
                *(_process_page(client, p) for p in batch)
            )

            next_queue: List[str] = []
            for res in results:
                if res.fetch_failed:
                    had_fetch_failure = True
                    continue
                if res.classify_failed:
                    classify_failures += 1
                    continue

                classify_successes += 1
                classification = res.classification
                full = res.content or ""            # includes Jina link summary
                main = strip_links_summary(full)     # page body only

                if classification.type == "individual_listings":
                    listing_pages.append(
                        ListingPage(
                            url=res.url,
                            llm_count=classification.count_if_any,
                            llm_basis=classification.count_basis,
                            # Count cards from the body only, never the link dump.
                            detail_keys=extract_detail_keys(res.url, main, res.url),
                            heading_count=count_headings(main),
                        )
                    )
                    # A listings page may be paginated — follow "next page" links.
                    for link in extract_pagination_links(res.url, full):
                        if link not in visited:
                            next_queue.append(link)

                # Always look for links to (other) listings pages — the entry URL
                # may be a homepage that only shows a few featured properties while
                # the full list lives at /properties. Global detail-key dedup makes
                # exploring extra pages safe (no double counting). Discovery uses
                # the FULL content so nav links in Jina's summary are included.
                for link in extract_candidate_links(res.url, full):
                    if link not in visited:
                        next_queue.append(link)

                if classification.type == "categories":
                    for link in classification.child_links:
                        link = normalize_url(link)
                        if link and link not in visited:
                            next_queue.append(link)

            queue = next_queue
            depth += 1

    # No page classified at all but classifications were attempted → hard error
    # (almost always a bad/missing OpenAI key or quota). Surface it.
    if classify_successes == 0 and classify_failures > 0:
        raise ClassificationError(
            "Every page classification failed — check OPENAI_API_KEY and quota."
        )

    return _build_response(listing_pages, visited, had_fetch_failure)


def _build_response(
    listing_pages: List[ListingPage],
    visited: set,
    had_fetch_failure: bool,
) -> CheckResponse:
    pages_checked = sorted(visited)

    if not listing_pages:
        return CheckResponse(
            has_10_plus_properties=False,
            property_count=None,
            count_type="estimated",
            source="no property count signal found",
            pages_checked=pages_checked,
            breakdown=[],
        )

    # Dedupe listing pages by URL.
    seen_urls: set = set()
    unique: List[ListingPage] = []
    for p in listing_pages:
        if p.url in seen_urls:
            continue
        seen_urls.add(p.url)
        unique.append(p)

    # Deterministic signal: unique property-detail links across ALL listing
    # pages (global dedup, so paginated/overlapping pages don't double count).
    global_keys: Set[str] = set()
    for p in unique:
        global_keys |= p.detail_keys
    det_total = len(global_keys)

    # LLM signals, taken as the max across pages (a stated/paginated total
    # usually describes the whole collection, not just one page).
    stated_totals = [
        p.llm_count for p in unique
        if p.llm_basis == "stated_total" and p.llm_count is not None
    ]
    pagination_totals = [
        p.llm_count for p in unique
        if p.llm_basis == "pagination" and p.llm_count is not None
    ]
    llm_counted_sum = sum(
        p.llm_count for p in unique
        if p.llm_basis in ("counted_items", "unknown") and p.llm_count is not None
    )
    heading_total = sum(p.heading_count for p in unique)

    max_stated = max(stated_totals) if stated_totals else None
    max_pagination = max(pagination_totals) if pagination_totals else None

    # ── Reconcile into one total + basis ────────────────────────────────────
    # The deterministic detail-key union is primary: across categories it sums
    # distinct properties, across pagination it dedupes overlap. LLM stated /
    # pagination totals refine it (a site may advertise more than we scraped).
    if det_total > 0:
        total, basis = det_total, "counted_items"
        if max_pagination is not None and max_pagination > det_total:
            total, basis = max_pagination, "pagination"
        # A single listings page that explicitly claims MORE than we could find
        # links for is lazy-loading the rest — trust the stated total.
        if len(unique) == 1 and max_stated is not None and max_stated > total:
            total, basis = max_stated, "stated_total"
    elif stated_totals:
        # No parseable detail links. Sum per-category stated totals; a single
        # page's stated total stands alone.
        total = sum(stated_totals) if len(stated_totals) > 1 else stated_totals[0]
        basis = "stated_total"
    elif max_pagination is not None:
        total, basis = max_pagination, "pagination"
    elif llm_counted_sum > 0:
        total, basis = llm_counted_sum, "counted_items"
    elif heading_total > 0:
        total, basis = heading_total, "counted_items"
    else:
        total, basis = 0, "unknown"

    if total <= 0:
        return CheckResponse(
            has_10_plus_properties=False,
            property_count=None,
            count_type="estimated",
            source="no property count signal found",
            pages_checked=pages_checked,
            breakdown=[_page_count(p) for p in unique],
        )

    # Exact when the number is stated, or a deterministic card count captured
    # every page (not undercut by a larger pagination total) and nothing failed.
    exact = (
        basis == "stated_total"
        or (
            basis == "counted_items"
            and det_total > 0
            and (max_pagination is None or max_pagination <= det_total)
        )
    )
    count_type = "confirmed" if (exact and not had_fetch_failure) else "estimated"

    if len(unique) == 1:
        source = f"count from {unique[0].url} (basis: {basis})"
    else:
        source = f"aggregated across {len(unique)} listing pages (basis: {basis})"

    return CheckResponse(
        has_10_plus_properties=total >= 10,
        property_count=total,
        count_type=count_type,
        source=source,
        pages_checked=pages_checked,
        breakdown=[_page_count(p) for p in unique],
    )


def _page_count(p: ListingPage) -> PageCount:
    """Per-page breakdown row: prefer the deterministic key count for display."""
    det = len(p.detail_keys)
    if det > 0:
        return PageCount(url=p.url, count=det, count_basis="counted_items")
    return PageCount(url=p.url, count=p.llm_count, count_basis=p.llm_basis)
