"""Breadth-first crawl loop from section 5 of the build plan.

Fetches the homepage, classifies pages, drills into category pages, and
tallies individual-listing counts — all with concurrent fetches per level.
"""

import asyncio
import os
from typing import List, Optional
from urllib.parse import urljoin

import httpx

from .classifier import classify_page
from .jina_client import fetch_via_jina, normalize_url
from .schemas import CheckResponse, PageClassification


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class ListingPage:
    def __init__(self, url: str, count: Optional[int]):
        self.url = url
        self.count = count


async def _process_page(
    client: httpx.AsyncClient, page_url: str
) -> tuple[str, Optional[PageClassification]]:
    """Fetch + classify a single page. Returns (url, classification|None)."""
    content = await fetch_via_jina(client, page_url)
    if not content:
        return page_url, None
    classification = await classify_page(page_url, content)
    return page_url, classification


async def check_properties(url: str) -> CheckResponse:
    max_pages = _int_env("MAX_PAGES_PER_SITE", 8)
    max_depth = _int_env("MAX_CRAWL_DEPTH", 2)

    start = normalize_url(url)
    visited: set[str] = set()
    queue: List[str] = [start]
    listing_pages: List[ListingPage] = []
    had_fetch_failure = False
    depth = 0

    async with httpx.AsyncClient() as client:
        while queue and len(visited) < max_pages and depth <= max_depth:
            # Take only as many as we have room for this level.
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
            for page_url, classification in results:
                if classification is None:
                    had_fetch_failure = True
                    continue
                if classification.type == "individual_listings":
                    listing_pages.append(
                        ListingPage(page_url, classification.count_if_any)
                    )
                elif classification.type == "categories":
                    for link in classification.child_links:
                        absolute = normalize_url(urljoin(page_url, link))
                        if absolute and absolute not in visited:
                            next_queue.append(absolute)
                # "irrelevant" → drop

            queue = next_queue
            depth += 1

    return _build_response(listing_pages, visited, had_fetch_failure)


def _build_response(
    listing_pages: List[ListingPage],
    visited: set[str],
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
        )

    counted = [p for p in listing_pages if p.count is not None]
    total = sum(p.count for p in counted)

    all_counted = len(counted) == len(listing_pages)
    count_type = "confirmed" if (all_counted and not had_fetch_failure) else "estimated"

    listing_urls = [p.url for p in listing_pages]
    if len(listing_urls) == 1:
        source = f"count from {listing_urls[0]}"
    else:
        source = f"summed across {len(listing_urls)} listing pages"

    return CheckResponse(
        has_10_plus_properties=total >= 10,
        property_count=total if total > 0 else None,
        count_type=count_type,
        source=source,
        pages_checked=pages_checked,
    )
