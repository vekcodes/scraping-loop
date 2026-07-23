"""Breadth-first crawl loop from section 5 of the build plan.

Fetches the homepage, classifies pages, follows both classifier-detected
category links AND keyword-scanned candidate links (so it can find the
listings page even when the homepage itself is a marketing page), then
tallies individual-listing counts. Fetches run concurrently per level.
"""

import asyncio
import logging
import os
from typing import List, Optional, Tuple

import httpx

from .classifier import classify_page
from .jina_client import (
    extract_candidate_links,
    fetch_via_jina,
    normalize_url,
)
from .schemas import CheckResponse, PageClassification, PageCount

logger = logging.getLogger("property_checker.crawler")


class ClassificationError(Exception):
    """Raised when no page could be classified (e.g. bad OpenAI key/quota)."""


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class ListingPage:
    def __init__(self, url: str, count: Optional[int], count_basis: str = "unknown"):
        self.url = url
        self.count = count
        self.count_basis = count_basis


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
    """Fetch + classify one page, capturing fetch/classify failures separately."""
    content = await fetch_via_jina(client, page_url)
    if not content:
        return PageResult(page_url, None, None, fetch_failed=True, classify_failed=False)
    try:
        classification = await classify_page(page_url, content)
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

                if classification.type == "individual_listings":
                    listing_pages.append(
                        ListingPage(
                            res.url,
                            classification.count_if_any,
                            classification.count_basis,
                        )
                    )
                    # Don't wander off a listings page — avoids double counting.
                    continue

                # For homepage (irrelevant) and category pages alike, discover
                # where the listings actually live via the keyword link scanner.
                candidates = extract_candidate_links(res.url, res.content or "")
                for link in candidates:
                    if link not in visited:
                        next_queue.append(link)

                if classification.type == "categories":
                    for link in classification.child_links:
                        link = normalize_url(link)
                        if link and link not in visited:
                            next_queue.append(link)

            queue = next_queue
            depth += 1

    # If nothing classified at all AND we had classify failures, that's a hard
    # error (almost always a bad/again-missing OpenAI key or quota) — surface it
    # rather than pretending "no properties found."
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

    # Dedupe listing pages by URL (keep the first count seen per URL).
    seen_urls: set = set()
    unique_listings: List[ListingPage] = []
    for p in listing_pages:
        if p.url in seen_urls:
            continue
        seen_urls.add(p.url)
        unique_listings.append(p)

    breakdown = [
        PageCount(url=p.url, count=p.count, count_basis=p.count_basis)
        for p in unique_listings
    ]

    counted = [p for p in unique_listings if p.count is not None]
    total = sum(p.count for p in counted)

    all_counted = len(counted) == len(unique_listings)
    # "confirmed" only when every listing page produced a count, no fetch
    # failed, and no count was a soft estimate (pagination math / unknown).
    exact_bases = {"stated_total", "counted_items"}
    all_exact = all(p.count_basis in exact_bases for p in counted)
    count_type = (
        "confirmed"
        if (all_counted and all_exact and not had_fetch_failure)
        else "estimated"
    )

    if len(unique_listings) == 1:
        source = f"count from {unique_listings[0].url} ({unique_listings[0].count_basis})"
    else:
        source = f"summed across {len(unique_listings)} listing pages"

    return CheckResponse(
        has_10_plus_properties=total >= 10,
        property_count=total if total > 0 else None,
        count_type=count_type,
        source=source,
        pages_checked=pages_checked,
        breakdown=breakdown,
    )
