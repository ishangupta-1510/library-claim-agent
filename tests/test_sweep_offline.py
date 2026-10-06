"""The whole sweep pipeline offline: synthetic frames, a fake vision model, no network.

This proves the stages connect: frames are saved and scored, planes register,
vision runs only on new coverage, duplicates merge, books get dimensions from
the marker, and the packet + report come out with code-computed totals.
"""

import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from library_claim.config import settings
from library_claim.sweep import SweepSession
from tests.test_tracking import MARKER_CM, _pan, _shelf_face


class FakeVision:
    """Returns one readable 'book' in the middle of every rectified/raw image it is shown."""

    def __init__(self):
        self.calls = []

    async def generate_json(self, image_jpeg, prompt, schema):
        self.calls.append(prompt[:20])
        if "books" in schema["properties"]:
            payload = {"books": [{"box_2d": [100, 480, 900, 520], "orientation": "upright", "title": "", "author": "",
                                  "publisher": "", "all_text": "", "legible": False, "age_cues": []}]}
        else:
            payload = {"items": [{"box_2d": [0, 0, 500, 500], "category": "lighting", "description": "floor lamp",
                                  "material": "metal", "brand_model": "", "is_artwork": False}]}
        return payload, {"input_tokens": 1000, "output_tokens": 100, "latency_s": 0.01}


@pytest.fixture
def cfg(tmp_path):
    return replace(settings(), sweeps_dir=tmp_path, marker_size_cm=MARKER_CM, serpapi_key="", google_api_key="")


async def test_sweep_produces_traceable_packet(cfg, monkeypatch):
    # No network in tests: identification gets no candidates, pricing finds nothing.
    async def no_candidates(*a, **k):
        return []
    monkeypatch.setattr("library_claim.stages.identify.fetch_candidates", no_candidates)

    events = []

    async def emit(event):
        events.append(event)

    vision = FakeVision()
    sweep = SweepSession(cfg, vision, emit, device="test")
    frames, _ = _pan(_shelf_face(), n=6)
    feedback = []
    for _, image in frames:
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        feedback.append(await sweep.add_frame(jpeg.tobytes()))
    blurred = cv2.GaussianBlur(frames[0][1], (41, 41), 15)
    ok, jpeg = cv2.imencode(".jpg", blurred)
    feedback.append(await sweep.add_frame(jpeg.tobytes()))

    assert feedback[0]["marker"] and feedback[0]["metric"]
    assert all(f["plane"] == "P1" for f in feedback[:6])  # chained onto one unit
    assert "blur" in feedback[-1]["problems"]

    for corner in ([0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]):
        sweep.add_ar_point("floor_corner", corner)
    sweep.add_ar_point("ceiling", [1, 2.7, 1])

    packet = await sweep.finish()
    out = sweep.dir

    assert len(vision.calls) < 12  # only frames with new coverage went to the model
    assert packet.totals.book_count >= 1
    first = packet.books[0]
    assert first.status == "unidentified" and first.title == ""
    assert first.spine_height_cm and 50 < first.spine_height_cm < 90  # fake box spans most of the shelf face
    assert (out / first.frame_ref).exists()
    assert packet.room.floor_area_m2 == pytest.approx(12.0) and packet.room.wall_area_m2 == pytest.approx(14 * 2.7)
    assert packet.items and packet.items[0].replacement_cost.low is None  # no price source, so no price
    assert packet.totals.books_replacement_cost == 0
    assert packet.stages["time_to_packet_s"] >= 0 and "vision_spines" in packet.stages["latency_s"]

    saved = json.loads((out / "claim_packet.json").read_text())
    assert saved["totals"]["book_count"] == packet.totals.book_count
    assert (out / "report.html").read_text(encoding="utf-8").count("<tr>") > 5
    assert any(e["type"] == "inventory" for e in events) and events[-1]["type"] == "packet"
    # Stage errors are caught per frame in production; in tests they must not happen at all.
    assert not [e for e in events if e["type"] == "stage_error"]
