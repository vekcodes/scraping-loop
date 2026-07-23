"""Wraps r.jina.ai fetching with error handling.

r.jina.ai renders JS-heavy pages on Jina's own infrastructure and returns
clean markdown, so this service only ever makes plain HTTP calls.
"""

import os
import re
from typing import List, Optional, Set
from urllib.parse import parse_qsl, urljoin, urlparse

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

# Every URL inside a markdown link target: `](url)`. Unlike _MD_LINK_RE, this
# also catches the OUTER href of a nested image link `[![alt](img)](href)`,
# where the anchor-capturing regex would only see the inner image URL.
_ANY_LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")


def _all_link_urls(content: str) -> List[str]:
    return _ANY_LINK_RE.findall(content or "")


def extract_candidate_links(base_url: str, content: str) -> List[str]:
    """Scan a page's markdown for links that likely lead to property listings.

    Returns absolute, same-site, deduped URLs whose URL or anchor text matches
    a candidate keyword. This is the 'link scanner' from the build plan: it lets
    the crawler find the listings page even when the current page (e.g. a
    marketing homepage) is itself classified 'irrelevant'.
    """
    if not content:
        return []

    # Also index anchor text by URL so keyword matches can use link text too.
    anchor_by_url = {}
    for anchor, raw in _MD_LINK_RE.findall(content):
        anchor_by_url.setdefault(raw, anchor)

    seen: set = set()
    out: List[str] = []
    for raw_link in _all_link_urls(content):
        link = normalize_url(urljoin(base_url, raw_link))
        if not link or link in seen:
            continue
        haystack = (raw_link + " " + anchor_by_url.get(raw_link, "")).lower()
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


# ── Deterministic property-card counting ─────────────────────────────────────
# The LLM is unreliable at tallying cards (it double-counts grid+list renders or
# miscounts image links). Instead we deterministically count each property's
# unique detail-page link, which is stable across the common site permutations:
#   - path slugs:        /stowaway, /white-house-cottage
#   - query-id links:    /details.aspx?PropertyID=391025
#   - nested detail:     /rentals/beach-villa
# Each unique property → one "detail key"; the count is the number of keys.

# First path segment that marks a listings container (its children are details).
_PROPERTY_PARENTS: Set[str] = {
    "property", "properties", "rental", "rentals", "home", "homes", "listing",
    "listings", "accommodation", "accommodations", "villa", "villas", "let",
    "lets", "holiday-lets", "stay", "stays", "apartment", "apartments", "house",
    "houses", "unit", "units", "cottage", "cottages", "cabin", "cabins",
}

# Single-segment slugs that are navigation/utility pages, never a property.
_NAV_SLUGS: Set[str] = {
    "", "about", "about-us", "contact", "contact-us", "blog", "news", "home",
    "index", "search", "book", "booking", "bookings", "faq", "faqs", "content",
    "reviews", "review", "gallery", "location", "locations", "areas", "area",
    "cart", "login", "account", "register", "terms", "privacy", "cookies",
    "owners", "owner", "list-your-property", "things-to-do", "offers",
    "gift-vouchers", "gift-voucher", "team", "careers", "press", "sitemap",
    "enquire", "enquiry", "services", "why-us", "how-it-works",
}

# Query-param names that carry a specific item id (details.aspx?PropertyID=...).
_ID_PARAM_RE = re.compile(
    r"(?i)(^id$|.*id$|property|unit|listing|ref|home|cottage|villa|accom|prop)"
)
# Resource handlers / assets that are never property pages.
_ASSET_EXT_RE = re.compile(
    r"\.(png|jpe?g|svg|gif|webp|css|js|pdf|ico|mp4|woff2?|axd|ashx|asmx|json|xml)$",
    re.I,
)
# Page extensions stripped before the nav-slug check (search.aspx -> search).
_PAGE_EXT_RE = re.compile(r"\.(aspx?|html?|php|jsp|cfm)$", re.I)


def _detail_key(link: str, base_url: str, listing_url: str) -> Optional[str]:
    """Return a stable key identifying a property detail page, or None."""
    pu = urlparse(link)
    if registered_domain(pu.netloc) != registered_domain(urlparse(base_url).netloc):
        return None
    # Never count the listing page itself.
    if link.split("#")[0].rstrip("/") == listing_url.split("#")[0].rstrip("/"):
        return None

    segs = [s for s in pu.path.split("/") if s]
    if segs and _ASSET_EXT_RE.search(segs[-1]):
        return None

    # 1) Query-id detail links (e.g. /details.aspx?PropertyID=391025)
    id_params = [
        f"{k.lower()}={v}"
        for k, v in parse_qsl(pu.query)
        if _ID_PARAM_RE.match(k) and v
    ]
    if id_params:
        return pu.path.lower() + "?" + "&".join(sorted(id_params))

    # 2) Single-segment slug (e.g. /stowaway)
    if len(segs) == 1:
        slug = _PAGE_EXT_RE.sub("", segs[0].lower())
        if slug in _NAV_SLUGS or slug in _PROPERTY_PARENTS:
            return None
        return "/" + slug

    # 3) Nested detail under a listings container (e.g. /rentals/beach-villa)
    if len(segs) >= 2 and segs[0].lower() in _PROPERTY_PARENTS:
        last = segs[-1].lower()
        if last in _NAV_SLUGS or last in _PROPERTY_PARENTS:
            return None
        return "/" + "/".join(s.lower() for s in segs)

    return None


def extract_detail_keys(base_url: str, content: str, listing_url: str) -> Set[str]:
    """Deterministically extract unique property-detail keys from a page.

    len(result) is the number of distinct properties linked on the page. Safe to
    union across multiple pages/categories — duplicate properties collapse.
    """
    keys: Set[str] = set()
    if not content:
        return keys
    for raw_link in _all_link_urls(content):
        link = normalize_url(urljoin(base_url, raw_link))
        if not link:
            continue
        key = _detail_key(link, base_url, listing_url)
        if key:
            keys.add(key)
    return keys


_HEADING_RE = re.compile(r"^#{2,4}\s+(.+?)\s*$", re.M)


def count_headings(content: str) -> int:
    """Count distinct card-style headings — a secondary cross-check signal."""
    if not content:
        return 0
    return len({h.strip() for h in _HEADING_RE.findall(content)})


# Pagination links so paginated listings can be fully crawled and summed.
_PAGINATION_RE = re.compile(r"(?i)([?&](page|p|pg|pagenum|offset|start)=\d+|/page/\d+)")


def extract_pagination_links(base_url: str, content: str) -> List[str]:
    """Return same-site 'next page' style links found on a listings page."""
    if not content:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for raw_link in _all_link_urls(content):
        link = normalize_url(urljoin(base_url, raw_link))
        if not link or link in seen:
            continue
        if not same_site(base_url, link):
            continue
        if _PAGINATION_RE.search(link):
            seen.add(link)
            out.append(link)
    return out
