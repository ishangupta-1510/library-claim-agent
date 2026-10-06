"""Merge per-frame detections into one inventory where each book appears once.

Detections are placed in the coordinates of their shelving unit's plane
(centimetres when a marker anchored the unit, otherwise the pixels of the
unit's first frame). The same spine seen in several overlapping frames then
lands in the same place.

Frames often show only part of a spine (its top in one frame, its bottom in
the next). So:
- Identity is decided along the spine's thickness axis: two upright boxes in
  the same column whose heights overlap are the same spine, even if each shows
  a different part of it. Plain box IoU split such spines into duplicates.
- Each sighting records which of its edges were cut off by the frame border.
  A spine's length is measured only from ends seen uncut (top from one frame,
  bottom from another), and its thickness only from sightings whose sides were
  both inside the frame. An end never seen uncut leaves the dimension unknown.

Each book keeps every sighting, so its frame_ref, reading and dimensions are
traceable to specific frames.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from rapidfuzz import fuzz

SAME_SPINE_OVERLAP = 0.5  # 1-D overlap along the thickness axis to be the same spine
SHELF_GAP_FRACTION = 0.5  # a new shelf row starts when a box bottom is this far (in book lengths) below the row

Box = tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class Sighting:
    frame_id: str
    box_plane: Box  # in plane units
    metric: bool  # plane units are centimetres
    orientation: str  # "upright" | "flat"
    title: str
    author: str
    publisher: str
    all_text: str
    legible: bool
    age_cues: list[str]
    sharpness: float = 0.0
    # Edges of the box that touched the frame border, in plane axes: "left", "right", "top", "bottom".
    cut: frozenset[str] = frozenset()


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    """Intersection over the shorter interval (so a thin sliver inside a wide box counts fully)."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = min(a1 - a0, b1 - b0)
    return inter / shorter if shorter > 0 else 0.0


def _axes(orientation: str) -> tuple[int, int]:
    """Indices (thickness_lo, length_lo) into a box: upright spines are thin in x, flat ones in y."""
    return (1, 0) if orientation == "flat" else (0, 1)


@dataclass
class InventoryBook:
    plane_id: str
    sightings: list[Sighting] = field(default_factory=list)
    shelf: str = ""
    position: int = 0

    @property
    def orientation(self) -> str:
        flat = sum(s.orientation == "flat" for s in self.sightings)
        return "flat" if flat > len(self.sightings) / 2 else "upright"

    @property
    def box(self) -> Box:
        """Thickness axis: median of sightings. Length axis: the union of what was seen."""
        t, l = _axes(self.orientation)
        boxes = [s.box_plane for s in self.sightings]
        out = [0.0] * 4
        out[t] = statistics.median(b[t] for b in boxes)
        out[t + 2] = statistics.median(b[t + 2] for b in boxes)
        out[l] = min(b[l] for b in boxes)
        out[l + 2] = max(b[l + 2] for b in boxes)
        return tuple(out)  # type: ignore[return-value]

    def matches(self, sighting: Sighting) -> bool:
        """Same spine: same column along the thickness axis, and overlapping along the length."""
        if sighting.orientation != self.orientation:
            return False
        t, l = _axes(self.orientation)
        box, other = self.box, sighting.box_plane
        same_column = _overlap_1d(box[t], box[t + 2], other[t], other[t + 2]) >= SAME_SPINE_OVERLAP
        overlapping = min(box[l + 2], other[l + 2]) > max(box[l], other[l])
        return same_column and overlapping

    @property
    def best(self) -> Sighting:
        """The reading to trust: legible first, then most text, then sharpest frame."""
        return max(self.sightings, key=lambda s: (s.legible, len(s.title) + len(s.author) + len(s.publisher), s.sharpness))

    @property
    def metric(self) -> bool:
        return any(s.metric for s in self.sightings)

    def dimensions_cm(self) -> tuple[float | None, float | None]:
        """(spine length, thickness) in cm from edges seen uncut; None where never seen."""
        metric = [s for s in self.sightings if s.metric]
        if not metric:
            return None, None
        flat = self.orientation == "flat"
        t, l = _axes(self.orientation)
        start_edge, end_edge = ("left", "right") if flat else ("top", "bottom")
        side_edges = {"top", "bottom"} if flat else {"left", "right"}

        starts = [s.box_plane[l] for s in metric if start_edge not in s.cut]
        ends = [s.box_plane[l + 2] for s in metric if end_edge not in s.cut]
        length = round(max(ends) - min(starts), 1) if starts and ends else None

        whole_width = [s.box_plane[t + 2] - s.box_plane[t] for s in metric if not (s.cut & side_edges)]
        thickness = round(statistics.median(whole_width), 1) if whole_width else None
        return length, thickness

    def frame_ref(self) -> str:
        """The frame to show an adjuster: the best reading, preferring a whole, uncut view."""
        return max(self.sightings, key=lambda s: (s.legible, not s.cut, s.sharpness)).frame_id

    def readings_disagree(self) -> bool:
        """Two legible readings that name different titles: the merge or the reading is suspect."""
        titles = [s.title for s in self.sightings if s.legible and s.title]
        return any(fuzz.ratio(a.lower(), b.lower()) < 70 for i, a in enumerate(titles) for b in titles[i + 1:])


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Inventory:
    def __init__(self) -> None:
        self.books: list[InventoryBook] = []

    def add(self, plane_id: str, sighting: Sighting) -> tuple[InventoryBook, bool]:
        """Add a sighting; returns the book it belongs to and whether the book is new."""
        candidates = [b for b in self.books if b.plane_id == plane_id and b.matches(sighting)]
        if candidates:
            # Several columns can touch a wide, oblique box: take the best-aligned one.
            t, _ = _axes(sighting.orientation)
            book = max(candidates, key=lambda b: _overlap_1d(b.box[t], b.box[t + 2], sighting.box_plane[t], sighting.box_plane[t + 2]))
            book.sightings.append(sighting)
            return book, False
        book = InventoryBook(plane_id, [sighting])
        self.books.append(book)
        return book, True

    def merge_overlaps(self, plane_id: str) -> int:
        """Merge books on one plane that turn out to be the same spine (after fragments were joined)."""
        merged = 0
        books = [b for b in self.books if b.plane_id == plane_id]
        for i, book in enumerate(books):
            if book not in self.books:
                continue
            for other in books[i + 1:]:
                if other in self.books and all(book.matches(s) for s in other.sightings[:1]):
                    book.sightings.extend(other.sightings)
                    self.books.remove(other)
                    merged += 1
        return merged

    def assign_shelves(self, unit_names: dict[str, str] | None = None) -> None:
        """Group each unit's books into shelf rows (top to bottom) and order left to right."""
        unit_names = unit_names or {}
        planes = sorted({b.plane_id for b in self.books})
        for index, plane_id in enumerate(planes):
            unit = unit_names.get(plane_id) or f"Unit {chr(ord('A') + index)}"
            books = sorted((b for b in self.books if b.plane_id == plane_id), key=lambda b: b.box[3])
            rows: list[list[InventoryBook]] = []
            for book in books:
                x0, y0, x1, y1 = book.box
                if rows:
                    row_bottom = statistics.median(b.box[3] for b in rows[-1])
                    row_height = statistics.median(b.box[3] - b.box[1] for b in rows[-1])
                    if y1 - row_bottom <= SHELF_GAP_FRACTION * row_height:
                        rows[-1].append(book)
                        continue
                rows.append([book])
            for row_number, row in enumerate(rows, start=1):
                for position, book in enumerate(sorted(row, key=lambda b: b.box[0]), start=1):
                    book.shelf = f"{unit} · Shelf {row_number}"
                    book.position = position
