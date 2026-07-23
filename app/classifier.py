"""Builds the classification prompt, calls gpt-4o-mini, parses the JSON."""

import json
import os
from typing import Optional

from openai import AsyncOpenAI

from .schemas import PageClassification

# Cap how much markdown we send to keep token cost and latency bounded.
MAX_CONTENT_CHARS = 16000

CLASSIFICATION_PROMPT = """You are analyzing one page from a property management company's website.

Page URL: {page_url}

Page content:
{content}

Classify this page as exactly one of:
- "individual_listings": the page shows actual individual properties/units \
available for booking or management (each with its own name, photo, or \
detail page)
- "categories": the page shows groupings, collections, or property TYPES \
(e.g. "Beachfront", "Downtown", "Luxury Villas") that must be clicked into \
to see actual properties — not a specific property itself
- "irrelevant": the page has nothing to do with a property listing or \
category (e.g. About Us, Contact, Blog)

If "individual_listings": count the properties shown on this page. Look for \
an explicit count ("47 properties"), pagination text ("Page 1 of 6", \
"Showing 12 of 84"), or count the visible listing items directly.

If "categories": list every link on this page (absolute URLs) that leads to \
a category/collection (not the current page itself).

Output strict JSON only:
{{
  "type": "individual_listings" | "categories" | "irrelevant",
  "count_if_any": <integer or null>,
  "child_links": [<url strings>] or []
}}"""


_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


async def classify_page(page_url: str, content: str) -> PageClassification:
    """Classify one page's markdown. Returns 'irrelevant' on any failure."""
    prompt = CLASSIFICATION_PROMPT.format(
        page_url=page_url,
        content=content[:MAX_CONTENT_CHARS],
    )

    try:
        resp = await _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        return PageClassification(
            type=data.get("type", "irrelevant"),
            count_if_any=data.get("count_if_any"),
            child_links=data.get("child_links") or [],
        )
    except Exception:
        # Any failure (API error, bad JSON, validation) → treat as irrelevant
        # so one bad page never fails the whole crawl.
        return PageClassification(type="irrelevant", count_if_any=None, child_links=[])
