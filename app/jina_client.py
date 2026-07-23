"""Wraps r.jina.ai fetching with error handling.

r.jina.ai renders JS-heavy pages on Jina's own infrastructure and returns
clean markdown, so this service only ever makes plain HTTP calls.
"""

import asyncio
import os
import re
from typing import List, Optional, Set
from urllib.parse import parse_qsl, urljoin, urlparse

import httpx

JINA_BASE = "https://r.jina.ai/"

# Retry rate-limited / transient fetches. Unauthenticated r.jina.ai allows only
# ~20 req/min, so a batch run hits 429s; back off and retry rather than dropping
# the page (a dropped fetch silently undercounts).
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0


def _jina_timeout() -> float:
    try:
        return float(os.getenv("JINA_TIMEOUT_SECONDS", "15"))
    except ValueError:
        return 15.0


def _headers() -> dict:
    headers = {
        "Accept": "text/plain",
        # Append a "Links/Buttons" summary of every anchor on the page. Nav menus
        # (often JS-rendered) are otherwise dropped from the markdown, so a
        # homepage may not expose its "/properties" link without this.
        "X-With-Links-Summary": "true",
    }
    api_key = os.getenv("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# Markers that begin Jina's appended link/image summary section.
_SUMMARY_MARKERS = ("Links/Buttons", "\n## Links", "\nLinks:", "\nImages:")


def strip_links_summary(content: str) -> str:
    """Return only the page's main content, dropping Jina's link/image summary.

    Counting property cards must use the main content — the appended link
    summary lists nav/utility pages that would otherwise inflate the count.
    """
    if not content:
        return content
    idxs = [i for i in (content.find(m) for m in _SUMMARY_MARKERS) if i > 0]
    return content[: min(idxs)] if idxs else content


async def fetch_via_jina(client: httpx.AsyncClient, target_url: str) -> Optional[str]:
    """Fetch a page as markdown via r.jina.ai.

    Returns the markdown string, or None on any failure (timeout, 4xx/5xx,
    network error) so the caller can skip the page rather than crash.
    """
    if not target_url:
        return None

    jina_url = JINA_BASE + target_url
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.get(
                jina_url,
                headers=_headers(),
                timeout=_jina_timeout(),
                follow_redirects=True,
            )
            resp.raise_for_status()
            content = resp.text.strip()
            return content or None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # Retry only on rate-limit / transient server errors.
            if status in (429, 502, 503, 504) and attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            return None
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            return None
    return None


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


# Property inventory often lives on a dedicated booking subdomain that is a
# DIFFERENT registered domain (e.g. site "heartwoodfh.com" links its listings on
# "book.heartwoodfurnishedhomes.com"). Follow these cross-domain, since they host
# the same company's properties.
_BOOKING_HOST_RE = re.compile(
    r"(?i)^(book|rentals?|reserve|reservations?|booking|portal|stay|guest|search|"
    r"availability|vacation|properties|listings?)\."
)


def is_booking_host(url: str) -> bool:
    return bool(_BOOKING_HOST_RE.match(urlparse(url).netloc))


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
        # Same site, or a booking portal on another domain (follow those too).
        if not same_site(base_url, link) and not is_booking_host(link):
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


# ── Stated-total extraction ──────────────────────────────────────────────────
# Big/JS sites rarely ship every property card in the HTML, but they usually
# PRINT the total ("442 properties", "Showing 1–20 of 442", "84 results").
# That printed number is the most reliable count when the full card list isn't
# scrapable, so we extract it deterministically.

# Listing nouns a real total is attached to. Kept plural/collection-oriented.
_TOTAL_NOUN = (
    r"(?:propert(?:y|ies)|vacation\s+rentals?|holiday\s+(?:homes?|lets?|rentals?)"
    r"|rental\s+homes?|rentals?|listings?|results?|villas?|cabins?|condos?"
    r"|accommodations?|homes?|properties\s+found)"
)

# Only "result-set total" phrasings are trusted — the number is bound inside a
# "… of N …" / "N … found" construction that genuinely denotes how many items a
# listing holds. The looser "N properties" phrasing is deliberately NOT used: on
# directories/aggregators it matches per-region/marketing counts ("293 rentals",
# "3,886 rentals with Private Pool") and produces wild overcounts.
_STATED_PATTERNS = [
    # "Showing 1–20 of 442", "1-12 of 84 results"
    re.compile(r"(?i)\b\d[\d,]*\s*[-–—]\s*\d[\d,]*\s+of\s+([\d,]{2,})"),
    # "of 442 properties", "of 84 results"
    re.compile(r"(?i)\bof\s+([\d,]{2,})\s+" + _TOTAL_NOUN),
    # "442 properties found", "84 results found"
    re.compile(r"(?i)\b([\d,]{2,})\s+(?:propert(?:y|ies)|results?|homes?|rentals?)\s+found"),
]

_MIN_STATED = 10
_MAX_STATED = 200000


def extract_stated_total(content: str) -> Optional[int]:
    """Return the largest trusted result-set total ("… of 442 …"), or None.

    Only result-set phrasings count, so per-card values ("3 bedrooms"), prices
    ("$450/night"), and directory/marketing counts ("293 rentals") are ignored.
    """
    if not content:
        return None
    best: Optional[int] = None
    for pat in _STATED_PATTERNS:
        for m in pat.finditer(content):
            try:
                n = int(m.group(1).replace(",", ""))
            except (ValueError, IndexError):
                continue
            if _MIN_STATED <= n <= _MAX_STATED and (best is None or n > best):
                best = n
    return best


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
