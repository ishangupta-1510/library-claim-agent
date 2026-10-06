"""Record what the offline mock flow replays, from one real run over the demo footage.

    python -m scripts.record_mock_fixtures sweeps/<a replay of dev_data/synthetic/frames>

Writes dev_data/synthetic/recorded/:
  vision/<stage>-<frame>.json   the run's vision answers (from its raw/ folder)
  http.json                     catalog and FX responses, recorded live now (credentials never stored)
  prices/                       the SerpAPI searches the packet used, copied from .cache/serpapi

Vision is not called again (the answers come from the run), and price
searches come from the local cache only, so recording spends no quota.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from library_claim.cassette import RecordingTransport
from library_claim.config import settings
from library_claim.stages.vision import RecordedVision
from library_claim.sweep import OfflineLookups, SweepSession

FRAMES = Path("dev_data/synthetic/frames")
OUT = Path("dev_data/synthetic/recorded")
PRICE_CACHE = Path(".cache/serpapi")


class RecordingLookups(OfflineLookups):
    """Live catalog/FX calls, recorded; prices from the local search cache."""

    def __init__(self, recorder: RecordingTransport):
        super().__init__(http_recording=OUT / "http.json", price_cache=PRICE_CACHE)
        self.recorder = recorder

    def transport(self):
        return self.recorder


# Only what parse_google_shopping / parse_ebay read; full SerpAPI pages made the recording 18 MB.
SHOPPING_FIELDS = ("title", "extracted_price", "second_hand_condition", "source", "product_link", "link")
EBAY_FIELDS = ("title", "price", "link", "condition")


def _keep_used_fields(response: dict) -> dict:
    out = {}
    if "shopping_results" in response:
        out["shopping_results"] = [{k: r[k] for k in SHOPPING_FIELDS if k in r} for r in response["shopping_results"]]
    if "organic_results" in response:
        out["organic_results"] = [{k: r[k] for k in EBAY_FIELDS if k in r} for r in response["organic_results"]]
    return out


async def record(run: Path) -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "vision").mkdir(parents=True)
    for path in sorted((run / "raw").glob("vision_*.json")):
        shutil.copy(path, OUT / "vision" / path.name)

    async def emit(event: dict) -> None:
        pass

    recorder = RecordingTransport()
    cfg = replace(settings(), marker_size_cm=10.0, sweeps_dir=Path("sweeps"))
    sweep = SweepSession(cfg, RecordedVision(OUT / "vision"), emit, device="fixture recording",
                         offline=RecordingLookups(recorder))
    sweep.country = "IN"
    for path in sorted(FRAMES.glob("*.jpg")):
        await sweep.add_frame(path.read_bytes())
    packet = await sweep.finish()

    recorder.save(OUT / "http.json")
    (OUT / "prices").mkdir()
    for path in set(sweep.prices.cache_used):
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["response"] = _keep_used_fields(entry["response"])
        (OUT / "prices" / path.name).write_text(json.dumps(entry), encoding="utf-8")
    print(f"{packet.totals.book_count} books, {packet.totals.books_identified} identified; "
          f"{len(recorder.store)} HTTP responses, {len(set(sweep.prices.cache_used))} price searches -> {OUT}")


if __name__ == "__main__":
    asyncio.run(record(Path(sys.argv[1])))
