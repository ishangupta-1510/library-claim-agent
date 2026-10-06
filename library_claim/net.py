"""Outbound HTTP for the post-sweep stages: retry transient failures, never crash the packet.

One slow price search or catalog lookup must not lose a sweep. Each call is
retried with backoff on timeouts, connection errors, 429 and 5xx; if it still
fails, the caller gets None and records that the lookup failed, so the line
goes to the review queue instead of the packet being lost.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger("library_claim.net")

RETRIES = 3
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


async def get(client: httpx.AsyncClient, url: str, *, params: dict | None = None, timeout: float = 30.0,
              headers: dict | None = None) -> httpx.Response | None:
    """GET with retries. Returns the response (any status), or None if the call kept failing."""
    for attempt in range(RETRIES + 1):
        try:
            response = await client.get(url, params=params, timeout=timeout, headers=headers)
            if response.status_code not in TRANSIENT_STATUS or attempt == RETRIES:
                return response
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == RETRIES:
                logger.warning("GET %s failed after %d attempts: %s", url, attempt + 1, type(exc).__name__)
                return None
        await asyncio.sleep(min(8.0, 1.5 * 2**attempt))
    return None
