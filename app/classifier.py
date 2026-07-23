"""Builds the classification prompt, calls the LLM, parses the JSON.

The model is configurable (CLASSIFIER_MODEL) so you can upgrade to a stronger
model for hard-to-count sites without touching code. The prompt is hardened to
return an EXACT total: it prefers a stated total or pagination total over
counting tiles, and de-duplicates grid/list views of the same property.
"""

import json
import logging
import os
from typing import Optional

from openai import AsyncOpenAI

from .schemas import PageClassification

logger = logging.getLogger("property_checker.classifier")

# Default model. Override with CLASSIFIER_MODEL (e.g. "gpt-4o", "gpt-4.1") when
# a site needs stronger reasoning to count exactly.
DEFAULT_MODEL = "gpt-4o-mini"


def _model() -> str:
    return os.getenv("CLASSIFIER_MODEL", DEFAULT_MODEL)


def _max_content_chars() -> int:
    """How much page markdown to send.

    Kept large by default so long listing pages aren't truncated (truncation
    silently undercounts). Lower it via MAX_CONTENT_CHARS only to cut cost.
    """
    try:
        return int(os.getenv("MAX_CONTENT_CHARS", "48000"))
    except (TypeError, ValueError):
        return 48000


CLASSIFICATION_PROMPT = """You are analyzing one page from a property management company's website.
Your job is to determine, as EXACTLY as possible, how many individual \
properties this company manages/lists.

Page URL: {page_url}

Page content:
{content}

Step 1 — classify this page as exactly one of:
- "individual_listings": the page shows actual individual properties/units \
(each with its own name, photo, price, or detail-page link)
- "categories": the page shows groupings/collections/property TYPES \
(e.g. "Beachfront", "Kissimmee", "Luxury Villas") that must be clicked into \
to reach actual properties — not specific properties themselves
- "irrelevant": nothing to do with property listings or categories \
(About Us, Contact, Blog, Attractions, etc.)

Step 2 — if "individual_listings", determine count_if_any using this PRIORITY \
order and report which basis you used in count_basis:
  1. "stated_total": the page explicitly states the total \
("12 properties", "84 homes", "Showing 1–12 of 12"). Use that number.
  2. "pagination": the page shows a paginated total \
("Showing 12 of 84", "Page 1 of 7"). Use the FULL total (e.g. 84), NOT the \
number visible on this one page. If only "Page 1 of 7" is shown and each \
page holds the same visible count V, estimate total = V × 7.
  3. "counted_items": no stated total — count the DISTINCT properties visible. \
IMPORTANT: many sites render each property twice (a grid card AND a list row, \
or a thumbnail AND a title link). Count each unique property ONCE — dedupe by \
property name or by its detail-page URL/ID. Do not count navigation, \
attractions, or blog links as properties.
  Use count_basis "unknown" only if you truly cannot tell.

Step 3 — if "categories", list every link (absolute URLs) that leads to a \
category/collection or to a listings page (not the current page itself).

Output strict JSON only:
{{
  "type": "individual_listings" | "categories" | "irrelevant",
  "count_if_any": <integer or null>,
  "count_basis": "stated_total" | "pagination" | "counted_items" | "unknown",
  "child_links": [<url strings>] or []
}}"""


_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


async def classify_page(page_url: str, content: str) -> PageClassification:
    """Classify one page's markdown.

    Raises on API errors (so a bad key/quota surfaces); only a malformed model
    response falls back to 'irrelevant'.
    """
    prompt = CLASSIFICATION_PROMPT.format(
        page_url=page_url,
        content=content[: _max_content_chars()],
    )

    resp = await _get_client().chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        return PageClassification(
            type=data.get("type", "irrelevant"),
            count_if_any=data.get("count_if_any"),
            count_basis=data.get("count_basis") or "unknown",
            child_links=data.get("child_links") or [],
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Unparseable classification for %s: %r", page_url, raw[:200])
        return PageClassification(
            type="irrelevant", count_if_any=None, count_basis="unknown", child_links=[]
        )
