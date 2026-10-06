"""One sweep, end to end: frames in during the walk, claim packet out after it.

During the sweep (per keyframe, fast):
    save frame -> quality -> marker -> register to a shelving-unit plane
    -> if the frame adds new shelf area: queue vision (spines + items)
    -> merged inventory and capture feedback go out as events

After the sweep (background agents, concurrent):
    identification -> pricing (book + items) -> room geometry -> packet -> report

Every stage records its latency and model/API usage, so cost per sweep and
latency per stage come from measurements.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import httpx
import numpy as np
from rapidfuzz import fuzz

from .config import Settings, locale_for
from .report import write_report
from .schemas import Book, ClaimPacket, Dimensions, Item, Price, PriceRange, Room, Sweep
from .stages import identify as identify_stage
from .stages.inventory import Inventory, InventoryBook, Sighting
from .stages.item_pricing import price_item
from .stages.pricing import PriceClient, price_book
from .stages.quality import FrameQuality, QualityMeter
from .stages.room import measure_room, to_ft2
from .stages.scale import PlaneScale, detect_markers, plane_scale, rectify, to_plane_cm
from .stages.tracking import Features, features, link_features
from .stages.vision import ITEM_PROMPT, ITEM_SCHEMA, SPINE_PROMPT, SPINE_SCHEMA, VisionModel, parse_items, parse_spines
from .totals import finalize

Emit = Callable[[dict], Awaitable[None]]

NEW_COVERAGE_FOR_VISION = 0.5  # each spine still lands in 2-3 vision frames at ~70% frame overlap
ITEM_SCAN_EVERY_S = 6.0
COVERAGE_CELL = 2.0  # plane units per coverage cell (cm on metric planes)
RECTIFIED_PX_PER_CM = 12.0  # fallback only; normally the frame's native resolution is kept
MIN_RECTIFIED_PX_PER_CM = 8.0
MAX_RECTIFIED_PX_PER_CM = 40.0
RECTIFIED_MAX_PX = 2400
MIN_MARKER_PX = 40  # smaller markers give unreliable corners
LINK_CANDIDATES = 6  # recent frames of the current unit tried when chaining a new frame
STRONG_LINK = 150  # inliers at which the first candidate is accepted without trying others
MIN_KEYPOINTS = 60  # fewer features than this: the frame cannot be placed reliably
CLOSURE_WINDOW = 12  # frames during which a new fragment keeps trying to rejoin an older plane
CLOSURE_CANDIDATES = 8  # keyframes per older plane tried for a rejoin
CLOSURE_MIN_INLIERS = 40  # stricter than chaining: a wrong merge is worse than a missed one

# Published per-token prices (USD per 1M tokens) used for the cost report; update with your model's rates.
MODEL_PRICES_USD = {"input": 0.30, "output": 2.50}
SERPAPI_USD_PER_SEARCH = 0.015  # paid-plan rate; the free plan costs nothing but caps searches


def _map_box(box: tuple[float, float, float, float], homography: np.ndarray) -> tuple[float, float, float, float]:
    """Map an upright box through a homography; returns the upright bounds of the mapped quad."""
    x0, y0, x1, y1 = box
    corners = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], np.float64)
    mapped = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
    (mx0, my0), (mx1, my1) = mapped.min(axis=0), mapped.max(axis=0)
    return float(mx0), float(my0), float(mx1), float(my1)


def _native_px_per_cm(plane_from_frame: np.ndarray, shape) -> float:
    """The frame's own resolution on the shelf plane, measured at its centre.

    Rectifying at a fixed 12 px/cm halved the resolution of close-up frames
    (about 23 px/cm), shrinking spine text before the model read it. Keeping
    the native resolution preserves every legible letter.
    """
    h, w = shape[:2]
    centre = np.array([[[w / 2, h / 2]], [[w / 2 + 1, h / 2]], [[w / 2, h / 2 + 1]]], np.float64)
    mapped = cv2.perspectiveTransform(centre, plane_from_frame).reshape(3, 2)
    cm_per_px = (np.linalg.norm(mapped[1] - mapped[0]) + np.linalg.norm(mapped[2] - mapped[0])) / 2
    return float(np.clip(1 / cm_per_px, MIN_RECTIFIED_PX_PER_CM, MAX_RECTIFIED_PX_PER_CM)) if cm_per_px > 0 else RECTIFIED_PX_PER_CM


EDGE_MARGIN = 0.015  # share of the frame size treated as "touching the edge"


def _cut_off(box_px, view, shape) -> bool:
    """Whether a detected spine box touches the original frame's border.

    On a rectified view, the box is mapped back into the original frame first,
    since the rectified canvas is larger than the area the camera saw.
    """
    h, w = shape[:2]
    x0, y0, x1, y1 = box_px
    corners = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], np.float64)
    if view is not None:
        corners = cv2.perspectiveTransform(corners, np.linalg.inv(view.image_to_rectified))
    pts = corners.reshape(4, 2)
    mx, my = EDGE_MARGIN * w, EDGE_MARGIN * h
    return bool((pts[:, 0] < mx).any() or (pts[:, 0] > w - mx).any() or (pts[:, 1] < my).any() or (pts[:, 1] > h - my).any())


def _spread(items: list, n: int) -> list:
    """Up to n items evenly spread over the list (always including the last)."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[min(len(items) - 1, int(i * step))] for i in range(n - 1)] + [items[-1]]


@dataclass
class Plane:
    """One shelving unit's coordinate system."""

    id: str
    metric: bool
    # frame id -> homography from frame pixels to plane units (cm if metric, else first-frame pixels)
    from_frame: dict[str, np.ndarray] = field(default_factory=dict)
    covered: set[tuple[int, int]] = field(default_factory=set)
    # Features of registered frames, kept to rejoin fragments of this unit later (loop closure).
    keyframes: list[tuple[str, Features]] = field(default_factory=list)
    created_at_frame: int = 0


@dataclass
class StageClock:
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict[str, dict] = field(default_factory=dict)

    def add(self, stage: str, seconds: float) -> None:
        self.timings[stage] = round(self.timings.get(stage, 0.0) + seconds, 2)

    def count(self, stage: str, **amounts: float) -> None:
        bucket = self.usage.setdefault(stage, {})
        for key, value in amounts.items():
            bucket[key] = bucket.get(key, 0) + value


class SweepSession:
    def __init__(self, settings: Settings, vision: VisionModel | None, emit: Emit, device: str = ""):
        self.settings = settings
        self.vision = vision
        self.emit = emit
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.dir = settings.sweeps_dir / self.id
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "raw").mkdir(exist_ok=True)
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.device = device
        self.country = settings.country
        self.inventory = Inventory()
        self.quality = QualityMeter()
        self.planes: list[Plane] = []
        self.current: Plane | None = None
        self.last_frame: tuple[str, np.ndarray] | None = None
        self.recent_frames: list[tuple[str, Features]] = []  # registered frames of the current unit, with features
        self.frame_log: list[dict] = []
        self.item_sightings: list[dict] = []
        self.last_item_scan = -ITEM_SCAN_EVERY_S
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker = asyncio.create_task(self._vision_worker())
        self.clock = StageClock()
        self.ar_points: list[dict] = []
        self.notes: list[dict] = []  # policyholder statements, with the book/item they apply to
        self.excluded_shelves: set[str] = set()
        self.art_answers: dict[int, bool] = {}  # item sighting index -> is print
        self.ended_at: float | None = None
        self.merges: list[dict] = []  # loop closures, for the stage report

    # ---------------- during the sweep ----------------

    def elapsed(self) -> float:
        return round(time.monotonic() - self.started, 2)

    async def add_frame(self, jpeg: bytes) -> dict:
        """Handle one keyframe; returns the capture feedback the agent can speak.

        The OpenCV work (decode, quality, marker, feature matching) runs in a
        worker thread so the event loop keeps relaying live audio smoothly.
        Frames are processed one at a time, so the thread never races itself.
        """
        t0 = time.monotonic()
        feedback, jobs = await asyncio.to_thread(self._process_frame, jpeg)
        for job in jobs:
            await self.queue.put(job)
        self.clock.add("capture_frame", time.monotonic() - t0)
        return feedback

    def _process_frame(self, jpeg: bytes) -> tuple[dict, list[tuple]]:
        frame_id = f"f{len(self.frame_log):04d}"
        (self.dir / "frames" / f"{frame_id}.jpg").write_bytes(jpeg)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        record = {"frame_id": frame_id, "t": self.elapsed(), "path": f"frames/{frame_id}.jpg"}
        if image is None:
            self.frame_log.append({**record, "error": "unreadable image"})
            return {"frame_id": frame_id, "problems": ["unreadable image"]}, []

        markers = detect_markers(image)
        quality = self.quality.assess(image, [m.corners for m in markers])
        record["quality"] = quality.to_dict()
        feedback = {"frame_id": frame_id, "problems": quality.problems(), "plane": None, "marker": False, "metric": False}
        jobs: list[tuple] = []
        plane = None
        if quality.usable and quality.featureless:
            record["skipped"] = "featureless (wall or floor)"
        elif quality.usable:
            plane, _, marker = self._register(frame_id, image, markers)
            if plane is None:
                record["skipped"] = "too little texture to place on a shelving unit"
        if plane is not None:
            record.update(plane=plane.id, metric=plane.metric, marker=marker)
            feedback.update(plane=plane.id, marker=marker, metric=plane.metric)
            gain = self._coverage_gain(plane, frame_id, image.shape)
            record["new_coverage"] = round(gain, 2)
            if gain >= NEW_COVERAGE_FOR_VISION:
                jobs.append(("spines", frame_id, image, plane, quality))
            if self.elapsed() - self.last_item_scan >= ITEM_SCAN_EVERY_S:
                self.last_item_scan = self.elapsed()
                jobs.append(("items", frame_id, image, plane, quality))
            self.last_frame = (frame_id, image)
        self.frame_log.append(record)
        return feedback, jobs

    def _new_plane(self, metric: bool) -> Plane:
        plane = Plane(id=f"P{len(self.planes) + 1}", metric=metric, created_at_frame=len(self.frame_log))
        self.planes.append(plane)
        return plane

    def _register(self, frame_id: str, image: np.ndarray, markers) -> tuple[Plane | None, PlaneScale | None, bool]:
        """Place the frame on its shelving unit's plane: by marker, by chaining, or as a new unit."""
        scales = [plane_scale(m, self.settings.marker_size_cm) for m in markers]
        scales = [s for s in scales if s.marker_px >= MIN_MARKER_PX]
        scale = max(scales, key=lambda s: s.marker_px, default=None)
        feats = features(image)
        if scale is None and len(feats.keypoints) < MIN_KEYPOINTS:
            return None, None, False  # bare shelf back or a smear: cannot be placed reliably
        chained = self._chain(feats)

        if scale is not None:
            if self.current is not None and chained is not None:
                if not self.current.metric:
                    self._make_metric(self.current, frame_id, chained, scale.homography)
                plane = self.current
            else:
                plane = self.current = self._new_plane(metric=True)
            plane.from_frame[frame_id] = scale.homography
            self._remember(plane, frame_id, feats)
            return self._close_loop(plane, frame_id, feats), scale, True
        if chained is not None:
            self.current.from_frame[frame_id] = chained
            self._remember(self.current, frame_id, feats)
            return self._close_loop(self.current, frame_id, feats), None, False
        # No marker and no link: a new fragment (a new unit, or a gap in the current one).
        plane = self.current = self._new_plane(metric=False)
        plane.from_frame[frame_id] = np.eye(3)
        self._remember(plane, frame_id, feats)
        return self._close_loop(plane, frame_id, feats), None, False

    def _remember(self, plane: Plane, frame_id: str, feats: Features) -> None:
        if self.recent_frames and self.recent_frames[-1][0] not in plane.from_frame:
            self.recent_frames = []  # a new unit started
        self.recent_frames = (self.recent_frames + [(frame_id, feats)])[-LINK_CANDIDATES:]
        plane.keyframes.append((frame_id, feats.compact()))

    def _close_loop(self, plane: Plane, frame_id: str, feats: Features) -> Plane:
        """Rejoin a young fragment to an older plane it overlaps (loop closure).

        A gap in the sweep (blur, a fast move, a bare stretch of shelf) starts a
        new fragment with no scale. When a later frame of that fragment overlaps
        any frame of an older plane, the fragment is transformed into that plane:
        it gains the older plane's scale, and books seen on both sides merge.
        Only young or unscaled fragments are checked, which keeps the cost small.
        """
        age = len(self.frame_log) - plane.created_at_frame
        # Only unscaled fragments gain from rejoining, and only while young (bounds the cost).
        if plane.metric or age > CLOSURE_WINDOW:
            return plane
        for other in reversed(self.planes):
            if other is plane:
                continue
            for other_id, other_feats in _spread(other.keyframes, CLOSURE_CANDIDATES):
                lk = link_features(other_feats, feats)
                if not lk.ok or lk.inliers < CLOSURE_MIN_INLIERS:
                    continue
                # other plane <- other frame <- this frame <- this plane
                other_from_this = other.from_frame[other_id] @ lk.homography @ np.linalg.inv(plane.from_frame[frame_id])
                self._absorb(into=other, fragment=plane, into_from_fragment=other_from_this)
                return other
        return plane

    def _absorb(self, into: Plane, fragment: Plane, into_from_fragment: np.ndarray) -> None:
        for fid, homography in fragment.from_frame.items():
            into.from_frame[fid] = into_from_fragment @ homography
        for book in self.inventory.books:
            if book.plane_id == fragment.id:
                book.plane_id = into.id
                for s in book.sightings:
                    s.box_plane = _map_box(s.box_plane, into_from_fragment)
                    s.metric = into.metric
        into.keyframes.extend(fragment.keyframes)
        into.covered.clear()  # coverage cells were in the fragment's units; recomputed as frames arrive
        self.planes.remove(fragment)
        self.recent_frames = [(fid, f) for fid, f in self.recent_frames if fid in into.from_frame]
        if self.current is fragment:
            self.current = into
        merged = self.inventory.merge_overlaps(into.id)
        self.merges.append({"into": into.id, "fragment": fragment.id, "frames": len(fragment.from_frame), "books_merged": merged})

    def _chain(self, feats: Features):
        """Homography onto the current unit's plane via the best-overlapping recent frame.

        Linking only to the previous frame breaks at every row change: the last
        frame of one row barely overlaps the first of the next, but the frame
        above it overlaps a lot. So try the recent frames and keep the link with
        the most RANSAC inliers.
        """
        if self.current is None:
            return None
        best, best_inliers = None, 0
        for prev_id, prev_feats in reversed(self.recent_frames):
            if prev_id not in self.current.from_frame:
                continue
            lk = link_features(prev_feats, feats)
            if lk.ok and lk.inliers > best_inliers:
                best, best_inliers = self.current.from_frame[prev_id] @ lk.homography, lk.inliers
                if best_inliers >= STRONG_LINK:
                    break  # good enough; skip the remaining candidates
        return best

    def _make_metric(self, plane: Plane, frame_id: str, plane_from_frame: np.ndarray, cm_from_frame: np.ndarray) -> None:
        """A marker appeared on a unit tracked in pixels: convert the whole unit to centimetres."""
        cm_from_plane = cm_from_frame @ np.linalg.inv(plane_from_frame)
        for fid, homography in plane.from_frame.items():
            plane.from_frame[fid] = cm_from_plane @ homography
        for book in self.inventory.books:
            if book.plane_id != plane.id:
                continue
            for s in book.sightings:
                s.box_plane = _map_box(s.box_plane, cm_from_plane)
                s.metric = True
        plane.covered.clear()
        plane.metric = True

    def _footprint(self, plane: Plane, frame_id: str, shape) -> np.ndarray:
        h, w = shape[:2]
        corners = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], np.float64)
        return cv2.perspectiveTransform(corners, plane.from_frame[frame_id]).reshape(4, 2)

    def _coverage_gain(self, plane: Plane, frame_id: str, shape) -> float:
        """Share of this frame's footprint on the plane not yet sent to vision."""
        quad = self._footprint(plane, frame_id, shape)
        cell = COVERAGE_CELL if plane.metric else 20.0
        quad = np.clip(quad, -5000, 5000)
        (x0, y0), (x1, y1) = quad.min(axis=0), quad.max(axis=0)
        xs = np.arange(np.floor(x0 / cell), np.ceil(x1 / cell))
        ys = np.arange(np.floor(y0 / cell), np.ceil(y1 / cell))
        if len(xs) * len(ys) > 400_000:
            return 1.0
        contour = quad.astype(np.float32).reshape(-1, 1, 2)
        cells = {(int(x), int(y)) for x in xs for y in ys if cv2.pointPolygonTest(contour, ((x + 0.5) * cell, (y + 0.5) * cell), False) >= 0}
        if not cells:
            return 0.0
        gain = len(cells - plane.covered) / len(cells)
        if gain >= NEW_COVERAGE_FOR_VISION:
            plane.covered |= cells
        return gain

    async def _vision_worker(self) -> None:
        while True:
            kind, frame_id, image, plane, quality = await self.queue.get()
            try:
                if self.vision is not None:
                    if kind == "spines":
                        await self._detect_spines(frame_id, image, plane, quality)
                    else:
                        await self._detect_items(frame_id, image, plane)
            except Exception as exc:  # one bad frame must not stop the sweep
                await self.emit({"type": "stage_error", "stage": kind, "frame_id": frame_id, "error": type(exc).__name__})
            finally:
                self.queue.task_done()

    async def _call_vision(self, stage: str, jpeg: bytes, prompt: str, schema: dict, frame_id: str) -> dict:
        t0 = time.monotonic()
        payload, info = await self.vision.generate_json(jpeg, prompt, schema)
        self.clock.add(stage, time.monotonic() - t0)
        self.clock.count(stage, calls=1, input_tokens=info.get("input_tokens", 0), output_tokens=info.get("output_tokens", 0))
        (self.dir / "raw" / f"{stage}-{frame_id}.json").write_text(json.dumps({"usage": info, "response": payload}, indent=1))
        return payload

    async def _detect_spines(self, frame_id: str, image: np.ndarray, plane: Plane, quality: FrameQuality) -> None:
        homography = plane.from_frame[frame_id]
        if plane.metric:
            scale = PlaneScale(homography, -1, self.settings.marker_size_cm, 1.0, 0.0)
            view = rectify(image, scale, px_per_cm=_native_px_per_cm(homography, image.shape))
            # An oblique frame covers a lot of plane; keep the image the model sees to a sane size
            # by lowering the resolution (measurements stay exact: they divide by px_per_cm).
            longest = max(view.image.shape[:2])
            if longest > RECTIFIED_MAX_PX:
                view = rectify(image, scale, px_per_cm=RECTIFIED_PX_PER_CM * RECTIFIED_MAX_PX / longest)
            sent = view.image
            ok, jpeg = cv2.imencode(".jpg", sent, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (self.dir / "frames" / f"{frame_id}-rectified.jpg").write_bytes(jpeg.tobytes())
        else:
            view, sent = None, image
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        payload = await self._call_vision("vision_spines", jpeg.tobytes(), SPINE_PROMPT, SPINE_SCHEMA, frame_id)
        detections = parse_spines(payload, sent.shape[1], sent.shape[0])

        new_books = 0
        for det in detections:
            if det.legible and _cut_off(det.box_px, view, image.shape):
                # Part of this spine is outside the frame, so its text is partial
                # ("MELUHA" for "The Immortals of Meluha"). Count it; read it elsewhere.
                det.title = det.author = ""
                det.legible = False
            if view is not None:
                x0, y0, x1, y1 = det.box_px
                ox, oy = view.origin_cm
                box = (ox + x0 / view.px_per_cm, oy + y0 / view.px_per_cm, ox + x1 / view.px_per_cm, oy + y1 / view.px_per_cm)
            else:
                pts = to_plane_cm(PlaneScale(homography, -1, 1, 1, 0), np.array([[det.box_px[0], det.box_px[1]], [det.box_px[2], det.box_px[3]]]))
                box = (float(pts[0, 0]), float(pts[0, 1]), float(pts[1, 0]), float(pts[1, 1]))
            sighting = Sighting(frame_id, box, plane.metric, det.orientation, det.title, det.author, det.publisher,
                                det.all_text, det.legible, det.age_cues, 1 - quality.blur_effect)
            _, is_new = self.inventory.add(plane.id, sighting)
            new_books += is_new
        self.inventory.assign_shelves()
        legible = sum(d.legible for d in detections)
        await self.emit({
            "type": "inventory", "frame_id": frame_id, "plane": plane.id, "detected": len(detections),
            "legible": legible, "new_books": new_books, "book_count": len(self.inventory.books),
            "books": self.live_books(),
        })

    async def _detect_items(self, frame_id: str, image: np.ndarray, plane: Plane | None) -> None:
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        payload = await self._call_vision("vision_items", jpeg.tobytes(), ITEM_PROMPT, ITEM_SCHEMA, frame_id)
        added = []
        for det in parse_items(payload, image.shape[1], image.shape[0]):
            dims = None
            if plane is not None and plane.metric and frame_id in plane.from_frame:
                pts = to_plane_cm(PlaneScale(plane.from_frame[frame_id], -1, 1, 1, 0), np.array([[det.box_px[0], det.box_px[1]], [det.box_px[2], det.box_px[3]]]))
                w, h = abs(pts[1, 0] - pts[0, 0]), abs(pts[1, 1] - pts[0, 1])
                if 1 < w < 600 and 1 < h < 400:
                    dims = (round(float(w)), round(float(h)))
            entry = {"frame_id": frame_id, "plane": plane.id if plane else None, "category": det.category,
                     "description": det.description, "material": det.material, "brand_model": det.brand_model,
                     "is_artwork": det.is_artwork, "dims_cm": dims}
            if not self._duplicate_item(entry):
                self.item_sightings.append(entry)
                added.append(entry)
        if added:
            await self.emit({"type": "items", "frame_id": frame_id, "added": added, "item_count": len(self.item_sightings),
                             "needs_question": [i for i in added if i["is_artwork"]]})

    def _duplicate_item(self, entry: dict) -> bool:
        """Same category and near-identical description = the same object seen again."""
        return any(
            other["category"] == entry["category"]
            and fuzz.token_set_ratio(other["description"], entry["description"]) >= 85
            for other in self.item_sightings
        )

    def live_books(self) -> list[dict]:
        out = []
        for i, book in enumerate(self.inventory.books):
            best = book.best
            height, thickness = book.dimensions_cm()
            out.append({"id": f"B{i + 1:03d}", "shelf": book.shelf, "position": book.position, "title": best.title,
                        "author": best.author, "legible": best.legible, "height_cm": height, "thickness_cm": thickness,
                        "frame_ref": book.frame_ref(), "excluded": book.shelf in self.excluded_shelves})
        return out

    # ---------------- voice corrections ----------------

    def book_in_view(self) -> InventoryBook | None:
        """The book closest to the centre of the most recent frame."""
        if self.last_frame is None or self.current is None:
            return None
        fid, image = self.last_frame
        if fid not in self.current.from_frame:
            return None
        h, w = image.shape[:2]
        centre = to_plane_cm(PlaneScale(self.current.from_frame[fid], -1, 1, 1, 0), np.array([[w / 2, h / 2]]))[0]
        books = [b for b in self.inventory.books if b.plane_id == self.current.id]
        if not books:
            return None
        return min(books, key=lambda b: np.hypot((b.box[0] + b.box[2]) / 2 - centre[0], (b.box[1] + b.box[3]) / 2 - centre[1]))

    def note_book_in_view(self, statement: str) -> dict:
        book = self.book_in_view()
        if book is None:
            return {"applied": False, "reason": "no book on screen yet"}
        self.notes.append({"target": id(book), "statement": statement, "t": self.elapsed()})
        index = self.inventory.books.index(book)
        return {"applied": True, "book_id": f"B{index + 1:03d}", "title_read": book.best.title or "(unreadable spine)", "shelf": book.shelf}

    def exclude_shelf_in_view(self, reason: str) -> dict:
        book = self.book_in_view()
        if book is None or not book.shelf:
            return {"applied": False, "reason": "no shelf on screen yet"}
        self.excluded_shelves.add(book.shelf)
        self.notes.append({"target": book.shelf, "statement": reason, "t": self.elapsed()})
        return {"applied": True, "shelf": book.shelf}

    def answer_art_question(self, is_print: bool) -> dict:
        pending = [i for i, item in enumerate(self.item_sightings) if item["is_artwork"] and i not in self.art_answers]
        if not pending:
            return {"applied": False, "reason": "no artwork waiting for an answer"}
        self.art_answers[pending[-1]] = is_print
        return {"applied": True, "item": self.item_sightings[pending[-1]]["description"], "is_print": is_print}

    def add_ar_point(self, kind: str, position: list[float]) -> dict:
        if kind not in ("floor_corner", "ceiling") or len(position) != 3:
            return {"applied": False}
        self.ar_points.append({"kind": kind, "position": [float(v) for v in position], "t": self.elapsed(),
                               "frame_ref": self.last_frame[0] if self.last_frame else ""})
        return {"applied": True, "floor_corners": sum(p["kind"] == "floor_corner" for p in self.ar_points)}

    # ---------------- after the sweep ----------------

    async def finish(self) -> ClaimPacket:
        self.ended_at = time.monotonic()
        duration = round(self.ended_at - self.started, 1)
        await self.emit({"type": "phase", "phase": "processing"})
        t0 = time.monotonic()
        await self.queue.join()
        self.worker.cancel()
        self.clock.add("vision_backlog_after_sweep", time.monotonic() - t0)
        self.inventory.assign_shelves()

        locale = locale_for(self.country)
        fallback = locale_for(self.settings.compare_country)
        async with httpx.AsyncClient(timeout=30) as http:
            prices = PriceClient(http, self.settings.serpapi_key, max_live_searches=self.settings.price_search_budget)
            books = await self._identify_and_price(http, prices, locale, fallback)
            items = await self._price_items(prices, locale)
            comparison = await self._locale_comparison(books, prices, fallback)
            self.clock.count("pricing_searches", searches=prices.live_searches, cached=len(prices.raw) - prices.live_searches)
            (self.dir / "raw" / "price_searches.json").write_text(json.dumps(prices.raw, indent=1))

        t0 = time.monotonic()
        room = self._room()
        self.clock.add("room", time.monotonic() - t0)

        packet = ClaimPacket(
            sweep=Sweep(id=self.id, captured_at=self.started_at, device=self.device, duration_s=duration,
                        country=locale.country, currency=locale.currency),
            room=room, books=books, items=items, locale_comparison=comparison,
        )
        finalize(packet)
        packet.stages = self._stage_report(time.monotonic() - self.ended_at)
        (self.dir / "frames.json").write_text(json.dumps(self.frame_log, indent=1))
        (self.dir / "claim_packet.json").write_text(packet.model_dump_json(indent=2))
        write_report(packet, self.dir)
        await self.emit({"type": "packet", "sweep_id": self.id, "totals": packet.totals.model_dump(),
                         "review_count": len(packet.review_queue)})
        return packet

    def _notes_for(self, book: InventoryBook) -> list[str]:
        return [n["statement"] for n in self.notes if n["target"] == id(book)]

    async def _identify_and_price(self, http, prices, locale, fallback) -> list[Book]:
        t_id = time.monotonic()
        sem = asyncio.Semaphore(4)
        cache: dict[tuple[str, str, str], identify_stage.Identification] = {}

        async def ident(book: InventoryBook):
            best = book.best
            key = (best.title.lower(), best.author.lower(), best.publisher.lower())
            if key not in cache:
                async with sem:
                    cache[key] = await identify_stage.identify(
                        identify_stage.SpineReading(best.title, best.author, best.publisher, best.all_text),
                        http, self.settings.google_api_key)
            return cache[key]

        idents = await asyncio.gather(*(ident(b) for b in self.inventory.books))
        self.clock.add("identification", time.monotonic() - t_id)

        t_price = time.monotonic()

        async def price(book: InventoryBook, ident):
            if ident.status != "identified" or book.shelf in self.excluded_shelves:
                return None
            async with sem:
                return await price_book(
                    title=ident.title, author=ident.author, isbn=ident.isbn, edition_year=ident.year if ident.isbn else "",
                    notes=self._notes_for(book), spine_text=book.best.all_text, visual_flags=book.best.age_cues,
                    locale=locale, fallback=fallback, threshold=self.settings.appraisal_threshold, client=prices)

        priced = await asyncio.gather(*(price(b, i) for b, i in zip(self.inventory.books, idents)))
        self.clock.add("pricing_books", time.monotonic() - t_price)

        out = []
        for index, (book, ident, bp) in enumerate(zip(self.inventory.books, idents, priced), start=1):
            height, thickness = book.dimensions_cm()
            best = book.best
            notes = self._notes_for(book) + ident.reasons
            if book.readings_disagree():
                notes.append("different titles read from different frames of this spine")
            status = ident.status
            if bp is not None and bp.appraisal.needed:
                status = "needs_appraisal"
            if book.shelf in self.excluded_shelves:
                notes.append("policyholder excluded this shelf (not theirs)")
            record = Book(
                id=f"B{index:03d}", shelf=book.shelf, position=book.position, frame_ref=f"frames/{book.frame_ref()}.jpg",
                status=status, orientation=best.orientation, spine_text=best.all_text,
                title=ident.title if ident.status == "identified" else "",
                author=ident.author if ident.status == "identified" else "",
                edition=ident.edition, isbn=ident.isbn, publisher=ident.publisher or best.publisher,
                spine_height_cm=height, spine_thickness_cm=thickness,
                dimension_method="marker-plane homography (median over sightings)" if height else "",
                id_confidence=ident.confidence, id_source=ident.source, id_url=ident.url,
                replacement_cost=bp.replacement if bp else Price(),
                used_value=bp.used if bp else Price(),
                notes=notes + (bp.notes if bp else []),
                excluded=book.shelf in self.excluded_shelves,
            )
            out.append(record)
        return out

    async def _price_items(self, prices, locale) -> list[Item]:
        t0 = time.monotonic()
        results = []
        for index, entry in enumerate(self.item_sightings, start=1):
            is_print = self.art_answers.get(index - 1, False)
            ip = await price_item(category=entry["category"], description=entry["description"], material=entry["material"],
                                  brand_model=entry["brand_model"], is_artwork=entry["is_artwork"],
                                  user_says_print=is_print, locale=locale, client=prices)
            dims = entry["dims_cm"]
            results.append(Item(
                id=f"I{index:03d}", category=entry["category"], description=entry["description"], material=entry["material"],
                brand_model=entry["brand_model"], frame_ref=f"frames/{entry['frame_id']}.jpg",
                dimensions_cm=Dimensions(w=dims[0], h=dims[1]) if dims else Dimensions(),
                dimension_method="marker-plane projection (approximate: object may not lie on the shelf plane)" if dims else "",
                status=ip.status, replacement_cost=ip.price,
                confidence=0.8 if entry["brand_model"] else 0.7,
                notes=ip.notes + (["policyholder: print"] if is_print else []),
            ))
        self.clock.add("pricing_items", time.monotonic() - t0)
        return results

    async def _locale_comparison(self, books: list[Book], prices: PriceClient, other) -> dict:
        """The same sample of identified books priced in the comparison market."""
        t0 = time.monotonic()
        sample = [b for b in books if b.status == "identified"][:10]
        rows = []
        for book in sample:
            bp = await price_book(title=book.title, author=book.author, isbn=book.isbn, edition_year="", notes=[],
                                  spine_text="", visual_flags=[], locale=other, fallback=other,
                                  threshold=float("inf"), client=prices)
            rows.append({"book_id": book.id, "title": book.title,
                         "home": book.replacement_cost.model_dump(), "other": bp.replacement.model_dump()})
        self.clock.add("pricing_second_locale", time.monotonic() - t0)
        return {"country": other.country, "currency": other.currency, "rows": rows}

    def _room(self) -> Room:
        corners = [p["position"] for p in self.ar_points if p["kind"] == "floor_corner"]
        ceilings = [p["position"][1] for p in self.ar_points if p["kind"] == "ceiling"]
        if len(corners) < 3:
            return Room(scale_method="ARCore WebXR hit tests (not captured in this sweep)", confidence=0)
        geometry = measure_room([tuple(c) for c in corners], float(np.median(ceilings)) if ceilings else None)
        shelved = self._shelved_wall_area()
        return Room(
            length_m=geometry.length_m, width_m=geometry.width_m, height_m=geometry.height_m,
            floor_area_m2=geometry.floor_area_m2, wall_area_m2=geometry.wall_area_m2, shelved_wall_area_m2=shelved,
            floor_area_ft2=to_ft2(geometry.floor_area_m2), wall_area_ft2=to_ft2(geometry.wall_area_m2),
            shelved_wall_area_ft2=to_ft2(shelved), shape=geometry.shape,
            scale_method=f"ARCore WebXR hit tests ({len(corners)} floor corners, {len(ceilings)} ceiling point(s)); "
                         "spine scale from ArUco marker plane",
            confidence=0.85 if ceilings else 0.6,
            evidence=[p["frame_ref"] for p in self.ar_points],
        )

    def _shelved_wall_area(self) -> float | None:
        """Sum over metric units of the area spanned by their books' outer extent (m²)."""
        total = 0.0
        for plane in self.planes:
            boxes = [b.box for b in self.inventory.books if b.plane_id == plane.id]
            if not plane.metric or not boxes:
                continue
            width = max(b[2] for b in boxes) - min(b[0] for b in boxes)
            height = max(b[3] for b in boxes) - min(b[1] for b in boxes)
            total += width * height / 10_000
        return round(total, 2) if total else None

    def _stage_report(self, after_sweep_s: float) -> dict:
        usage = self.clock.usage
        tokens_in = sum(u.get("input_tokens", 0) for u in usage.values())
        tokens_out = sum(u.get("output_tokens", 0) for u in usage.values())
        searches = usage.get("pricing_searches", {}).get("searches", 0)
        vision_cost = tokens_in / 1e6 * MODEL_PRICES_USD["input"] + tokens_out / 1e6 * MODEL_PRICES_USD["output"]
        return {
            "latency_s": self.clock.timings,
            "time_to_packet_s": round(after_sweep_s, 1),
            "usage": usage,
            "frames": len(self.frame_log),
            "loop_closures": self.merges,
            "units": len([p for p in self.planes if any(b.plane_id == p.id for b in self.inventory.books)]),
            "frames_sent_to_vision": usage.get("vision_spines", {}).get("calls", 0) + usage.get("vision_items", {}).get("calls", 0),
            "cost_usd_estimate": {
                "vision": round(vision_cost, 4),
                "price_search": round(searches * SERPAPI_USD_PER_SEARCH, 4),
                "note": "Live voice session billed separately by Gemini Live; free tier used during development.",
            },
        }
