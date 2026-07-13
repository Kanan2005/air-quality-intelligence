"""
Shared outbound HTTP helper for all external API calls.

On some Windows setups, asyncio's default ProactorEventLoop causes
httpx.AsyncClient to hang on the SSL read for certain hosts -- the TCP
connection succeeds, but the response is never received until the
timeout fires, even though the exact same request via curl or a plain
synchronous client completes in ~1-3s.

To sidestep this everywhere at once, every outbound call in this app
should go through get() / post() below: they use the synchronous
httpx.Client (which doesn't go through the async socket path at all)
and run it inside a worker thread via asyncio.to_thread, so the FastAPI
event loop is never blocked while still avoiding the hang.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx


def _sync_request(
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
) -> httpx.Response:
    with httpx.Client(timeout=timeout) as client:
        resp = client.request(method, url, params=params, data=data, headers=headers)
        resp.raise_for_status()
        return resp


async def get(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
) -> httpx.Response:
    """Async-safe GET. Returns the full httpx.Response (call .json() or .text)."""
    return await asyncio.to_thread(_sync_request, "GET", url, params=params, headers=headers, timeout=timeout)


async def post(
    url: str,
    *,
    data: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
) -> httpx.Response:
    """Async-safe POST. Returns the full httpx.Response (call .json() or .text)."""
    return await asyncio.to_thread(_sync_request, "POST", url, data=data, headers=headers, timeout=timeout)


async def get_json(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
) -> Any:
    resp = await get(url, params=params, headers=headers, timeout=timeout)
    return resp.json()