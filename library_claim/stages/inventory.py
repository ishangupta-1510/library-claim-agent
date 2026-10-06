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

ANCHORS_MIN = 3  # read spines that must agree before a frame is moved
ANCHOR_SHIFT_MIN = 0.3  # ...by at least this share of a spine's thickness
ANCHOR_SPREAD_MAX = 0.25  # ...and agree to within this share
SAME_SPINE_OVERLAP = 0.5  # 1-D overlap along the thickness axis to be the same spine
SAME_ROW_SHARE = 0.5  # a book is on the current shelf row when this share of its height lies in the row's band

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
    # False when the reading placed this sighting on a book its box missed (a mislocated detection):
    # it counts as a sighting of the book, but its box is not used for position or dimensions.
    box_trusted: bool = True

    @property
    def partial(self) -> bool:
        """An end of the spine was outside the frame, so its text may be truncated ("MELUHA").

        Spine text runs along the spine's length, so only a cut end truncates
        it; a cut side does not.
        """
        ends = {"left", "right"} if self.orientation == "flat" else {"top", "bottom"}
        return bool(self.cut & ends)


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    """Intersection over the shorter interval (so a thin sliver inside a wide box counts fully)."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = min(a1 - a0, b1 - b0)
    return inter / shorter if shorter > 0 else 0.0


ORIENTATION_ASPECT = 1.5  # a box this much longer one way than the other shows its orientation by itself


def orientation_of(box: Box, label: str, cut: frozenset[str] = frozenset()) -> str:
    """Upright or flat from the box's shape on the shelf plane; the vision label when the shape cannot say.

    Boxes are on the rectified plane, so a spine is a long thin rectangle
    along its length. The model's label is less reliable than that: in one
    frame it called three upright spines "flat", which split each into a
    second book with height and thickness swapped. A side cut by the frame
    border only bounds that dimension from below, so a box cut at the top
    or bottom can prove "upright" (tall even when cut) but never "flat"
    (the top of a spine peeking in at the frame's edge is short and wide).
    """
    width, height = box[2] - box[0], box[3] - box[1]
    height_whole = not cut & {"top", "bottom"}
    width_whole = not cut & {"left", "right"}
    if height >= ORIENTATION_ASPECT * width and width_whole:
        return "upright"
    if width >= ORIENTATION_ASPECT * height and height_whole:
        return "flat"
    return label


def _axes(orientation: str) -> tuple[int, int]:
    """Indices (thickness_lo, length_lo) into a box: upright spines are thin in x, flat ones in y."""
    return (1, 0) if orientation == "flat" else (0, 1)


@dataclass
class InventoryBook:
    plane_id: str
    sightings: list[Sighting] = field(default_factory=list)
    shelf: str = ""
    position: int = 0
    # What the policyholder said about this book ("this one is signed"); kept on the book so merges carry it.
    statements: list[str] = field(default_factory=list)
    excluded: str = ""  # why the policyholder excluded it (not theirs); empty when claimed

    def absorb(self, other: "InventoryBook") -> None:
        """Take over another record of the same spine: its sightings and what was said about it."""
        self.sightings.extend(other.sightings)
        self.statements.extend(s for s in other.statements if s not in self.statements)
        self.excluded = self.excluded or other.excluded

    @property
    def orientation(self) -> str:
        # Sightings cut by the frame border have unreliable shape and label; they vote only if nothing else does.
        voters = [s for s in self.sightings if not s.cut] or self.sightings
        flat = sum(s.orientation == "flat" for s in voters)
        return "flat" if flat > len(voters) / 2 else "upright"

    @property
    def placed(self) -> list[Sighting]:
        """Sightings whose boxes locate the book (all of them if none is trusted)."""
        return [s for s in self.sightings if s.box_trusted] or self.sightings

    @property
    def box(self) -> Box:
        """Thickness axis: median of sightings. Length axis: from the spine's ends (see `ends`)."""
        t, l = _axes(self.orientation)
        boxes = [s.box_plane for s in self.placed]
        out = [0.0] * 4
        out[t] = statistics.median(b[t] for b in boxes)
        out[t + 2] = statistics.median(b[t + 2] for b in boxes)
        (out[l], _), (out[l + 2], _) = self.ends(self.placed)
        return tuple(out)  # type: ignore[return-value]

    def ends(self, sightings: list[Sighting]) -> tuple[tuple[float, bool], tuple[float, bool]]:
        """Each end of the spine along its length, and whether it was ever seen uncut.

        An end seen uncut is the median over the sightings that saw it: one
        box that ran into the next shelf row must not stretch the spine (a
        single max stretched a 27 cm spine to 41 cm and let it swallow the
        book below). An end never seen uncut is the furthest it was seen.
        """
        flat = self.orientation == "flat"
        _, l = _axes(self.orientation)
        start_edge, end_edge = ("left", "right") if flat else ("top", "bottom")
        starts = [s.box_plane[l] for s in sightings if start_edge not in s.cut]
        stops = [s.box_plane[l + 2] for s in sightings if end_edge not in s.cut]
        start = (statistics.median(starts), True) if starts else (min(s.box_plane[l] for s in sightings), False)
        stop = (statistics.median(stops), True) if stops else (max(s.box_plane[l + 2] for s in sightings), False)
        return start, stop

    def matches(self, sighting: Sighting) -> bool:
        """Same spine: same column along the thickness axis, and overlapping along the length.

        A sighting whose orientation disagrees can still be this spine when it
        was cut by the frame border (its shape and label are then unreliable);
        it is judged in this book's axes.
        """
        if sighting.orientation != self.orientation and not sighting.cut:
            return False
        t, l = _axes(self.orientation)
        box, other = self.box, sighting.box_plane
        same_column = _overlap_1d(box[t], box[t + 2], other[t], other[t + 2]) >= SAME_SPINE_OVERLAP
        overlapping = min(box[l + 2], other[l + 2]) > max(box[l], other[l])
        return same_column and overlapping

    @property
    def best(self) -> Sighting:
        """The reading to trust: legible first, then a whole spine over a partial one, then most text, then sharpest."""
        return max(self.sightings, key=lambda s: (s.legible, not s.partial, len(s.title) + len(s.author) + len(s.publisher), s.sharpness))

    @property
    def metric(self) -> bool:
        return any(s.metric for s in self.sightings)

    def dimensions_cm(self) -> tuple[float | None, float | None]:
        """(spine length, thickness) in cm from edges seen uncut; None where never seen."""
        metric = [s for s in self.placed if s.metric]
        if not metric:
            return None, None
        t, _ = _axes(self.orientation)
        side_edges = {"top", "bottom"} if self.orientation == "flat" else {"left", "right"}

        (start, start_seen), (stop, stop_seen) = self.ends(metric)
        length = round(stop - start, 1) if start_seen and stop_seen else None

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


NEIGHBOUR_GAP = 0.25  # columns closer than this share of the thinner one's width are neighbours
SPAN_COVER = 0.8  # a sighting spans a column when it covers this share of the column's thickness


def _spans(sighting: Sighting, book: "InventoryBook") -> bool:
    t, _ = _axes(book.orientation)
    col0, col1 = book.box[t], book.box[t + 2]
    inter = max(0.0, min(col1, sighting.box_plane[t + 2]) - max(col0, sighting.box_plane[t]))
    return col1 > col0 and inter / (col1 - col0) >= SPAN_COVER


def _same_reading(a: str, b: str) -> bool:
    return fuzz.ratio(a.lower(), b.lower()) >= 90


def _reads_otherwise(book: "InventoryBook", sighting: Sighting) -> bool:
    """The book has legible readings, and none of them is this sighting's title."""
    titles = [s.title for s in book.sightings if s.legible and s.title]
    return bool(titles) and not any(_same_reading(t, sighting.title) for t in titles)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Inventory:
    def __init__(self) -> None:
        self.books: list[InventoryBook] = []
        # Shelf rows the policyholder excluded, by position on their unit (labels are renumbered as rows
        # are discovered): (plane id, top, bottom, reason). Books seen there later are excluded too.
        self.excluded_rows: list[tuple[str, float, float, str]] = []

    def exclude_row_of(self, book: "InventoryBook", reason: str) -> tuple[float, float]:
        """Exclude the shelf row this book stands on; returns the row's extent on its unit."""
        row = [b for b in self.books if b.plane_id == book.plane_id and b.shelf == book.shelf] or [book]
        top, bottom = min(b.box[1] for b in row), max(b.box[3] for b in row)
        self.excluded_rows.append((book.plane_id, top, bottom, reason))
        self.apply_exclusions()
        return top, bottom

    def apply_exclusions(self) -> None:
        for book in self.books:
            centre = (book.box[1] + book.box[3]) / 2
            for plane_id, top, bottom, reason in self.excluded_rows:
                if book.plane_id == plane_id and top <= centre <= bottom:
                    book.excluded = book.excluded or reason

    def add(self, plane_id: str, sighting: Sighting) -> tuple[InventoryBook, bool]:
        """Add a sighting; returns the book it belongs to and whether the book is new."""
        candidates = [b for b in self.books if b.plane_id == plane_id and b.matches(sighting)]
        if candidates:
            # Several columns can touch a wide, oblique box: take the best-aligned one.
            def alignment(b: InventoryBook) -> float:
                t, _ = _axes(b.orientation)
                return _overlap_1d(b.box[t], b.box[t + 2], sighting.box_plane[t], sighting.box_plane[t + 2])

            book = max(candidates, key=alignment)
            reader = self._same_reading_elsewhere(plane_id, sighting)
            if reader is not None and reader is not book and _reads_otherwise(book, sighting):
                # The box landed on a neighbour that reads as another title; the reading places it.
                sighting.box_trusted = False
                book = reader
            book.sightings.append(sighting)
            return book, False
        book = self._same_reading_elsewhere(plane_id, sighting)
        if book is not None:
            sighting.box_trusted = False
            book.sightings.append(sighting)
            return book, False
        book = InventoryBook(plane_id, [sighting])
        self.books.append(book)
        return book, True

    def add_frame(self, plane_id: str, sightings: list[Sighting]) -> tuple[int, tuple[float, float]]:
        """Add one frame's sightings, first correcting the frame's placement by the spines it read.

        Returns (new books, the shift applied in plane units).
        """
        dx, dy = self._text_anchor_shift(plane_id, sightings)
        if dx or dy:
            for s in sightings:
                x0, y0, x1, y1 = s.box_plane
                s.box_plane = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        new = 0
        for s in sightings:
            new += self.add(plane_id, s)[1]
        return new, (dx, dy)

    def _readers(self, plane_id: str, sighting: Sighting) -> list["InventoryBook"]:
        """Books placed from other frames whose reading is this sighting's title, in the same shelf row."""
        if not (sighting.legible and sighting.title):
            return []
        out = []
        for book in self.books:
            if book.plane_id != plane_id or any(s.frame_id == sighting.frame_id for s in book.sightings):
                continue  # the same title twice in one frame is two copies
            if not any(s.legible and s.title and _same_reading(s.title, sighting.title) for s in book.sightings):
                continue
            _, l = _axes(book.orientation)
            box, other = book.box, sighting.box_plane
            if min(box[l + 2], other[l + 2]) > max(box[l], other[l]):
                out.append(book)
        return out

    def _same_reading_elsewhere(self, plane_id: str, sighting: Sighting) -> "InventoryBook | None":
        """A legible title already placed in this row from other frames: the same spine, mislocated by its box."""
        readers = self._readers(plane_id, sighting)
        return readers[0] if len(readers) == 1 else None

    def _text_anchor_shift(self, plane_id: str, sightings: list[Sighting]) -> tuple[float, float]:
        """The frame's offset from the plane, measured on spines it read that are already placed.

        Chained registration can drift by a spine's width, which shifts every
        box in the frame onto its neighbour's column. Titles are landmarks: if
        at least ANCHORS_MIN of them agree on one offset larger than a fraction
        of a spine, the frame is moved by it.
        """
        offsets: dict[int, list[float]] = {0: [], 1: []}
        widths: dict[int, list[float]] = {0: [], 1: []}
        for s in sightings:
            readers = self._readers(plane_id, s)
            if len(readers) != 1 or not s.box_trusted:
                continue
            book = readers[0]
            t, _ = _axes(book.orientation)
            col = book.box
            offsets[t].append((col[t] + col[t + 2]) / 2 - (s.box_plane[t] + s.box_plane[t + 2]) / 2)
            widths[t].append(col[t + 2] - col[t])
        shift = [0.0, 0.0]
        for axis in (0, 1):
            if len(offsets[axis]) < ANCHORS_MIN:
                continue
            median = statistics.median(offsets[axis])
            spread = statistics.median(abs(o - median) for o in offsets[axis])
            spine = statistics.median(widths[axis])
            if abs(median) >= ANCHOR_SHIFT_MIN * spine and spread <= ANCHOR_SPREAD_MAX * spine:
                shift[axis] = median
        return shift[0], shift[1]

    def merge_overlaps(self, plane_id: str) -> int:
        """Merge books on one plane that turn out to be the same spine (after fragments were joined)."""
        merged = 0
        books = [b for b in self.books if b.plane_id == plane_id]
        for i, book in enumerate(books):
            if book not in self.books:
                continue
            for other in books[i + 1:]:
                if other in self.books and all(book.matches(s) for s in other.sightings[:1]):
                    book.absorb(other)
                    self.books.remove(other)
                    merged += 1
        return merged

    def resolve_splits(self) -> int:
        """Merge neighbouring columns that are one spine split in two by a detection.

        A detector sometimes draws two boxes on one wide spine, and sometimes
        one box across two thin ones. Neither is trusted on its own: two books
        are merged when more frames saw a single box spanning both columns
        than frames that saw the two separately.
        """
        merged = 0
        changed = True
        while changed:
            changed = False
            for a, b in self._neighbour_pairs():
                spanning = {s.frame_id for s in a.sightings + b.sightings if _spans(s, a) and _spans(s, b)}
                separate = {s.frame_id for s in a.sightings} & {s.frame_id for s in b.sightings}
                if len(spanning) > len(separate - spanning):
                    a.absorb(b)
                    self.books.remove(b)
                    merged += 1
                    changed = True
                    break
        return merged

    def _neighbour_pairs(self):
        for i, a in enumerate(self.books):
            for b in self.books[i + 1:]:
                if a.plane_id != b.plane_id or a.orientation != b.orientation:
                    continue
                t, l = _axes(a.orientation)
                ba, bb = a.box, b.box
                gap = max(ba[t], bb[t]) - min(ba[t + 2], bb[t + 2])  # negative when the columns overlap
                thinner = min(ba[t + 2] - ba[t], bb[t + 2] - bb[t])
                if min(ba[l + 2], bb[l + 2]) > max(ba[l], bb[l]) and gap < NEIGHBOUR_GAP * thinner:
                    yield a, b  # overlapping along the length and touching along the thickness axis

    def assign_shelves(self, unit_names: dict[str, str] | None = None) -> None:
        """Group each unit's books into shelf rows (top to bottom) and order left to right."""
        unit_names = unit_names or {}
        planes = sorted({b.plane_id for b in self.books})
        for index, plane_id in enumerate(planes):
            unit = unit_names.get(plane_id) or f"Unit {chr(ord('A') + index)}"
            # A shelf row is the vertical band its books occupy. A book joins the current row when most of
            # its height lies inside that band. Comparing bottoms instead split each book of a flat stack
            # (2-3 cm tall, resting on the one below) into a "shelf" of its own: a 4-shelf unit got 8.
            books = sorted((b for b in self.books if b.plane_id == plane_id), key=lambda b: (b.box[1] + b.box[3]) / 2)
            rows: list[list[InventoryBook]] = []
            band = (0.0, 0.0)
            for book in books:
                top, bottom = book.box[1], book.box[3]
                inside = max(0.0, min(bottom, band[1]) - max(top, band[0]))
                if rows and inside >= SAME_ROW_SHARE * (bottom - top):
                    rows[-1].append(book)
                    band = (min(band[0], top), max(band[1], bottom))
                    continue
                rows.append([book])
                band = (top, bottom)
            for row_number, row in enumerate(rows, start=1):
                for position, book in enumerate(sorted(row, key=lambda b: b.box[0]), start=1):
                    book.shelf = f"{unit} · Shelf {row_number}"
                    book.position = position
