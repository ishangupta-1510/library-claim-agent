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

    async def generate_json(self, image_jpeg, prompt, schema, call_id=""):
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

    saved = json.loads((out / "claim_packet.json").read_text(encoding="utf-8"))
    assert saved["totals"]["book_count"] == packet.totals.book_count
    assert (out / "report.html").read_text(encoding="utf-8").count("<tr>") > 5
    assert any(e["type"] == "inventory" for e in events) and events[-1]["type"] == "packet"
    # Stage errors are caught per frame in production; in tests they must not happen at all.
    assert not [e for e in events if e["type"] == "stage_error"]


class QuotaAfterOne(FakeVision):
    """Answers once, then reports the daily quota used up, as the free tier does."""

    async def generate_json(self, image_jpeg, prompt, schema, call_id=""):
        if self.calls:
            from library_claim.stages.vision import VisionQuotaExhausted

            self.calls.append("refused")
            raise VisionQuotaExhausted("gemini: daily request quota used up")
        return await super().generate_json(image_jpeg, prompt, schema, call_id)


async def test_daily_vision_quota_stops_calls_and_the_packet_still_builds(cfg, monkeypatch):
    async def no_candidates(*a, **k):
        return []
    monkeypatch.setattr("library_claim.stages.identify.fetch_candidates", no_candidates)

    async def emit(event):
        pass

    vision = QuotaAfterOne()
    sweep = SweepSession(cfg, vision, emit, device="test")
    for _, image in _pan(_shelf_face(), n=6)[0]:
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        await sweep.add_frame(jpeg.tobytes())
    packet = await sweep.finish()

    assert vision.calls.count("refused") == 1  # no call after the quota ran out
    assert packet.totals.book_count >= 1  # what was read before still counts
    assert any("quota" in e for e in packet.stages["service_events"])
    assert packet.stages["vision_jobs_not_run"] >= 1


class SlowVision(FakeVision):
    """Holds each answer until released, so the unit can change while a call is in flight."""

    def __init__(self):
        super().__init__()
        import asyncio
        self.release = asyncio.Event()

    async def generate_json(self, image_jpeg, prompt, schema, call_id=""):
        await self.release.wait()
        return await super().generate_json(image_jpeg, prompt, schema, call_id)


async def test_an_answer_arriving_after_its_unit_was_merged_lands_on_the_merged_unit(cfg, monkeypatch):
    import asyncio

    async def no_candidates(*a, **k):
        return []
    monkeypatch.setattr("library_claim.stages.identify.fetch_candidates", no_candidates)

    async def emit(event):
        pass

    vision = SlowVision()
    sweep = SweepSession(cfg, vision, emit, device="test")
    frames, _ = _pan(_shelf_face(), n=3)
    for _, image in frames:
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        await sweep.add_frame(jpeg.tobytes())
    await asyncio.sleep(0.05)  # calls are now in flight, holding the plane they were queued with
    # Merge the unit into a fresh one, as loop closure does, while the answers are pending.
    old = sweep.planes[0]
    target = sweep._new_plane(metric=old.metric)
    shift = np.array([[1, 0, 100.0], [0, 1, 0], [0, 0, 1]])
    sweep._absorb(into=target, fragment=old, into_from_fragment=shift)
    vision.release.set()
    packet = await sweep.finish()

    assert {b.shelf.split(" · ")[0] for b in packet.books} == {"Unit A"}  # one unit, no orphans
    assert all(b.plane_id == target.id for b in sweep.inventory.books)
    assert target.id != old.id


async def test_appraisal_threshold_is_converted_to_the_claim_currency(cfg):
    from library_claim.stages.pricing import FxRate

    class Rates:
        async def fx(self, base, quote):
            return FxRate(1 / 96.3, "2026-10-05") if (base, quote) == ("INR", "USD") else None

    async def emit(event):
        pass

    sweep = SweepSession(replace(cfg, appraisal_threshold=10000, appraisal_threshold_currency="INR"), None, emit)
    assert await sweep._appraisal_threshold(Rates(), "INR") == 10000
    assert await sweep._appraisal_threshold(Rates(), "USD") == pytest.approx(103.84, abs=0.01)  # not $10,000


async def test_malformed_room_points_are_refused(cfg):
    async def emit(event):
        pass

    sweep = SweepSession(cfg, None, emit)
    assert sweep.add_ar_point("floor_corner", ["a", 0, 0]) == {"applied": False}
    assert sweep.add_ar_point("floor_corner", [float("nan"), 0, 0]) == {"applied": False}
    assert sweep.add_ar_point("floor_corner", [1, 0, 2])["applied"]


def test_an_item_box_on_the_marker_is_the_marker_not_an_item():
    from library_claim.sweep import _covers

    marker = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], float)
    assert _covers((95, 95, 205, 210), marker)  # "marker tag sticker artwork"
    assert not _covers((0, 0, 1000, 700), marker)  # the bookcase the marker stands on
    assert not _covers((400, 100, 500, 200), marker)  # something beside it
