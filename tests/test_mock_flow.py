"""The offline mock flow reproduces the recorded run with no network at all."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from library_claim.config import settings
from library_claim.stages.vision import RecordedVision
from library_claim.sweep import OfflineLookups, SweepSession
from scripts.evaluate import evaluate

DEMO = Path("dev_data/synthetic")
RECORDED = DEMO / "recorded"


@pytest.fixture
def no_network(monkeypatch):
    async def refuse(self, request):
        raise AssertionError(f"network call in offline mode: {request.url}")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)


async def test_mock_flow_builds_the_recorded_packet_offline(tmp_path, no_network):
    async def emit(event):
        pass

    cfg = replace(settings(), sweeps_dir=tmp_path, marker_size_cm=10.0, google_api_key="", serpapi_key="")
    sweep = SweepSession(cfg, RecordedVision(RECORDED / "vision"), emit, device="test",
                         offline=OfflineLookups(RECORDED / "http.json", RECORDED / "prices"))
    sweep.country = "IN"
    for path in sorted((DEMO / "frames").glob("*.jpg")):
        await sweep.add_frame(path.read_bytes())
    packet = await sweep.finish()

    results = evaluate(packet.model_dump(), json.loads((DEMO / "ground_truth.json").read_text(encoding="utf-8")))
    assert results["book_count"]["pass"] and results["titles"]["pass"]
    assert packet.totals.currency == "INR" and packet.totals.books_replacement_cost > 0
    priced = [b for b in packet.books if b.replacement_cost.amount]
    assert priced and all(b.replacement_cost.url for b in priced)  # every price still carries its source
