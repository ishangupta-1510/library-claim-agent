"""Start the app:  python -m library_claim  [--port 8000]

Serves on this computer only (127.0.0.1): the camera works there without
HTTPS, and nobody else on the network can spend the API keys. For a phone,
put an HTTPS proxy in front (see README: `tailscale serve`).
"""

from __future__ import annotations

import argparse

import uvicorn

from . import mock
from .config import settings
from .server import DEMO


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m library_claim")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg = settings()
    url = f"http://localhost:{args.port}"
    print(f"\n  Library claim agent: {url}\n")
    print(f"  Mock flow (no APIs)  {'ready' if mock.available(DEMO) else 'unavailable: dev_data/synthetic is missing'}")
    live = "ready" if cfg.google_api_key else "needs GOOGLE_API_KEY in .env"
    print(f"  Demo sweep / live    {live}")
    print(f"  Prices (live sweeps) {'ready' if cfg.serpapi_key else 'needs SERPAPI_KEY in .env (lines stay unpriced)'}")
    print(f"  Size marker          {cfg.marker_size_cm} cm (MARKER_SIZE_CM)  ·  marker page: {url}/marker\n")
    uvicorn.run("library_claim.server:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
