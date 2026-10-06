"""Record and replay outbound HTTP, so the mock flow runs the real lookup code with no network.

A recording wraps the real transport and keeps each response keyed by the
request's method, URL and sorted query; credentials are dropped from the key
and never stored. Replay answers from that store; a request that was not
recorded gets a 404, which the lookup code already treats as "nothing found".
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import httpx

SECRET_PARAMS = {"key", "api_key", "apikey", "token", "access_token"}


def request_key(request: httpx.Request) -> str:
    params = sorted((k, v) for k, v in request.url.params.multi_items() if k.lower() not in SECRET_PARAMS)
    base = str(request.url.copy_with(query=None))
    return f"{request.method} {base}" + (f"?{urlencode(params)}" if params else "")


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport | None = None):
        self.inner = inner or httpx.AsyncHTTPTransport()
        self.store: dict[str, dict] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        body = await response.aread()
        self.store[request_key(request)] = {
            "status": response.status_code,
            "content_type": response.headers.get("content-type", "application/json"),
            "body": body.decode("utf-8", errors="replace"),
        }
        # The body is already decoded: drop the encoding headers so it is not decompressed a second time.
        headers = {k: v for k, v in response.headers.items() if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")}
        return httpx.Response(response.status_code, headers=headers, content=body, request=request)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.store, sort_keys=True), encoding="utf-8")


class ReplayTransport(httpx.AsyncBaseTransport):
    def __init__(self, path: Path):
        self.store: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.misses: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        entry = self.store.get(request_key(request))
        if entry is None:
            self.misses.append(request_key(request))
            return httpx.Response(404, json={"error": "not recorded"}, request=request)
        return httpx.Response(entry["status"], headers={"content-type": entry["content_type"]},
                              content=entry["body"].encode("utf-8"), request=request)
