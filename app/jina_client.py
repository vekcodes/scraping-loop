"""Wraps r.jina.ai fetching with error handling.

r.jina.ai renders JS-heavy pages on Jina's own infrastructure and returns
clean markdown, so this service only ever makes plain HTTP calls.
"""

import os
import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx

JINA_BASE = "https://r.jina.ai/"


def _jina_timeout() -> float:
    try:
        return float(os.getenv("JINA_TIMEOUT_SECONDS", "15"))
    except ValueError:
        return 15.0


def _headers() -> dict:
    headers = {"Accept": "text/plain"}
    api_key = os.getenv("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def fetch_via_jina(client: httpx.AsyncClient, target_url: str) -> Optional[str]:
    """Fetch a page as markdown via r.jina.ai.

    Returns the markdown string, or None on any failure (timeout, 4xx/5xx,
    network error) so the caller can skip the page rather than crash.
    """
    if not target_url:
        return None

    jina_url = JINA_BASE + target_url
    try:
        resp = await client.get(
            jina_url,
            headers=_headers(),
            timeout=_jina_timeout(),
            follow_redirects=True,
        )
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException):
        return None

    content = resp.text.strip()
    return content or None


def normalize_url(url: str) -> str:
    """Add a scheme if missing and strip trailing whitespace/fragments."""
    url = url.strip()
    if not url:
        return url
    if not urlparse(url).scheme:
        url = "https://" + url
    # Drop fragments — they don't change the fetched page.
    return url.split("#", 1)[0]


def registered_domain(host: str) -> str:
    """Return the last two labels of a host (e.g. 'revavista.com').

    Good enough to treat 'rentals.revavista.com' and 'www.revavista.com' as the
    same site while rejecting truly external domains.
    """
    host = (host or "").lower().split(":")[0]
    parts = [p for p in host.split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def same_site(url_a: str, url_b: str) -> bool:
    return registered_domain(urlparse(url_a).netloc) == registered_domain(
        urlparse(url_b).netloc
    )


# Keywords that flag a link as a likely listings/portfolio page. Matched against
# both the link's URL and its anchor text.
CANDIDATE_KEYWORDS = (
    "propert",      # property, properties
    "listing",
    "rental",
    "portfolio",
    "our-homes",
    "our homes",
    "homes",
    "vacation-rental",
    "accommodation",
    "villa",
    "cabin",
    "condo",
    "where-we-manage",
    "where we manage",
    "search",
    "book",
    "stays",
)

# Markdown link: [anchor text](url)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")


def extract_candidate_links(base_url: str, content: str) -> List[str]:
    """Scan a page's markdown for links that likely lead to property listings.

    Returns absolute, same-site, deduped URLs whose URL or anchor text matches
    a candidate keyword. This is the 'link scanner' from the build plan: it lets
    the crawler find the listings page even when the current page (e.g. a
    marketing homepage) is itself classified 'irrelevant'.
    """
    if not content:
        return []

    seen: set = set()
    out: List[str] = []
    for anchor, raw_link in _MD_LINK_RE.findall(content):
        link = normalize_url(urljoin(base_url, raw_link))
        if not link or link in seen:
            continue
        haystack = (raw_link + " " + anchor).lower()
        if not any(kw in haystack for kw in CANDIDATE_KEYWORDS):
            continue
        if not same_site(base_url, link):
            continue
        # Skip obvious asset/image links.
        if re.search(r"\.(png|jpe?g|svg|gif|webp|pdf|css|js)(\?|$)", link, re.I):
            continue
        seen.add(link)
        out.append(link)
    return out
