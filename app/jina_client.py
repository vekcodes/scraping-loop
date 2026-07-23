"""Wraps r.jina.ai fetching with error handling.

r.jina.ai renders JS-heavy pages on Jina's own infrastructure and returns
clean markdown, so this service only ever makes plain HTTP calls.
"""

import os
from typing import Optional
from urllib.parse import urlparse

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
