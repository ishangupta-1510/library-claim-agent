"""Merge per-frame detections into one inventory where each book appears once.

Detections are placed in the coordinates of their shelving unit's plane
(centimetres when a marker anchored the unit, otherwise the pixels of the
unit's first frame). The same spine seen in several overlapping frames then
lands in the same place, and overlapping boxes are merged.

Each merged book keeps every sighting, so its frame_ref, best reading and
median dimensions are all traceable to specific frames.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from rapidfuzz import fuzz

MERGE_IOU = 0.35
SHELF_GAP_FRACTION = 0.5  # a new shelf row starts when a box top is this far (in book heights) below the row


@dataclass
class Sighting:
    frame_id: str
    box_plane: tuple[float, float, float, float]  # x0, y0, x1, y1 in plane units
    metric: bool  # plane units are centimetres
    orientation: str
    title: str
    author: str
    publisher: str
    all_text: str
    legible: bool
    age_cues: list[str]
    sharpness: float = 0.0


@dataclass
class InventoryBook:
    plane_id: str
    sightings: list[Sighting] = field(default_factory=list)
    shelf: str = ""
    position: int = 0

    @property
    def box(self) -> tuple[float, float, float, float]:
        boxes = [s.box_plane for s in self.sightings]
        return tuple(statistics.median(b[i] for b in boxes) for i in range(4))  # type: ignore[return-value]

    @property
    def best(self) -> Sighting:
        """The reading to trust: legible first, then most text, then sharpest frame."""
        return max(self.sightings, key=lambda s: (s.legible, len(s.title) + len(s.author) + len(s.publisher), s.sharpness))

    @property
    def metric(self) -> bool:
        return any(s.metric for s in self.sightings)

    def dimensions_cm(self) -> tuple[float | None, float | None]:
        """(height, thickness) in cm: median over metric sightings; None without scale."""
        metric = [s.box_plane for s in self.sightings if s.metric]
        if not metric:
            return None, None
        widths = [b[2] - b[0] for b in metric]
        heights = [b[3] - b[1] for b in metric]
        w, h = statistics.median(widths), statistics.median(heights)
        # Upright: spine height is vertical. Flat (lying down): the spine runs horizontally.
        if self.best.orientation == "flat":
            return round(w, 1), round(h, 1)
        return round(h, 1), round(w, 1)

    def frame_ref(self) -> str:
        return self.best.frame_id

    def readings_disagree(self) -> bool:
        """Two legible readings that name different titles: the merge or the reading is suspect."""
        titles = [s.title for s in self.sightings if s.legible and s.title]
        return any(fuzz.ratio(a.lower(), b.lower()) < 70 for i, a in enumerate(titles) for b in titles[i + 1:])


def iou(a, b) -> float:
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
        best, best_iou = None, 0.0
        for book in self.books:
            if book.plane_id != plane_id:
                continue
            overlap = iou(book.box, sighting.box_plane)
            if overlap > best_iou:
                best, best_iou = book, overlap
        if best is not None and best_iou >= MERGE_IOU:
            best.sightings.append(sighting)
            return best, False
        book = InventoryBook(plane_id, [sighting])
        self.books.append(book)
        return book, True

    def merge_overlaps(self, plane_id: str) -> int:
        """Merge books on one plane whose boxes now overlap (after two fragments were joined)."""
        merged = 0
        books = [b for b in self.books if b.plane_id == plane_id]
        for i, book in enumerate(books):
            if book not in self.books:
                continue
            for other in books[i + 1:]:
                if other in self.books and iou(book.box, other.box) >= MERGE_IOU:
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
