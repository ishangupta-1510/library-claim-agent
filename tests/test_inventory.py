from library_claim.stages.inventory import Inventory, Sighting


def S(frame, box, title="", legible=False, orientation="upright", metric=True, sharp=100.0):
    return Sighting(frame, box, metric, orientation, title, "", "", title, legible, [], sharp)


def test_same_spine_in_overlapping_frames_counts_once():
    inv = Inventory()
    inv.add("A", S("f1", (10, 0, 13, 24), "Sapiens", True))
    _, new = inv.add("A", S("f2", (10.3, 0.2, 13.2, 24.1), "Sapiens", True))
    assert not new and len(inv.books) == 1
    assert len(inv.books[0].sightings) == 2


def test_neighbouring_spines_stay_separate():
    inv = Inventory()
    inv.add("A", S("f1", (10, 0, 13, 24)))
    inv.add("A", S("f1", (13.1, 0, 16, 24)))
    assert len(inv.books) == 2


def test_different_units_never_merge():
    inv = Inventory()
    inv.add("A", S("f1", (10, 0, 13, 24)))
    inv.add("B", S("f9", (10, 0, 13, 24)))
    assert len(inv.books) == 2


def test_best_reading_prefers_legible_then_sharpest():
    inv = Inventory()
    book, _ = inv.add("A", S("blurry", (0, 0, 3, 24), "", False, sharp=20))
    inv.add("A", S("sharp", (0, 0, 3, 24), "Dune Messiah", True, sharp=300))
    assert book.best.frame_id == "sharp" and book.frame_ref() == "sharp"


def test_dimensions_follow_orientation_and_need_metric_scale():
    inv = Inventory()
    upright, _ = inv.add("A", S("f", (0, 0, 3.0, 24.0)))
    flat, _ = inv.add("A", S("f", (20, 30, 44.0, 33.5), orientation="flat"))
    unscaled, _ = inv.add("B", S("g", (0, 0, 30, 240), metric=False))
    assert upright.dimensions_cm() == (24.0, 3.0)
    assert flat.dimensions_cm() == (24.0, 3.5)
    assert unscaled.dimensions_cm() == (None, None)


def test_shelves_assigned_top_to_bottom_left_to_right():
    inv = Inventory()
    for x in (20, 0, 10):
        inv.add("A", S("f", (x, 0, x + 3, 24)))  # top shelf, out of order
    for x in (5, 0):
        inv.add("A", S("f", (x, 35, x + 3, 58)))  # shelf below
    inv.assign_shelves()
    layout = sorted((b.shelf, b.position, b.box[0]) for b in inv.books)
    assert layout == [
        ("Unit A · Shelf 1", 1, 0), ("Unit A · Shelf 1", 2, 10), ("Unit A · Shelf 1", 3, 20),
        ("Unit A · Shelf 2", 1, 0), ("Unit A · Shelf 2", 2, 5),
    ]


def test_conflicting_readings_are_detected():
    inv = Inventory()
    book, _ = inv.add("A", S("f1", (0, 0, 3, 24), "Sapiens", True))
    inv.add("A", S("f2", (0, 0, 3, 24), "Homo Deus", True))
    assert book.readings_disagree()


def test_spine_cut_off_by_the_frame_edge_is_not_read():
    from library_claim.sweep import _cut_off

    assert _cut_off((0, 100, 40, 600), None, (720, 1280, 3))  # touches the left edge
    assert not _cut_off((300, 100, 340, 600), None, (720, 1280, 3))
