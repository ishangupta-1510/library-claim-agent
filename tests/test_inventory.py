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


def test_cut_edges_name_the_sides_touching_the_frame():
    from library_claim.sweep import _cut_edges

    assert _cut_edges((0, 100, 40, 600), None, (720, 1280, 3)) == {"left"}
    assert _cut_edges((300, 0, 340, 720), None, (720, 1280, 3)) == {"top", "bottom"}
    assert not _cut_edges((300, 100, 340, 600), None, (720, 1280, 3))


def C(frame, box, cut=(), title="", legible=False, orientation="upright"):
    return Sighting(frame, box, True, orientation, title, "", "", title, legible, [], 1.0, frozenset(cut))


def test_top_and_bottom_halves_of_one_spine_merge_and_combine_into_its_height():
    inv = Inventory()
    inv.add("A", C("f1", (10, 0, 13, 14), cut={"bottom"}))   # sees the top 14 cm, bottom cut off
    _, new = inv.add("A", C("f2", (10.2, 9, 13.1, 24), cut={"top"}))  # sees the bottom part
    assert not new and len(inv.books) == 1
    height, thickness = inv.books[0].dimensions_cm()
    assert height == 24.0 and thickness == 3.0


def test_length_unknown_if_an_end_was_never_seen():
    inv = Inventory()
    inv.add("A", C("f1", (10, 0, 13, 14), cut={"bottom"}))
    assert inv.books[0].dimensions_cm() == (None, 3.0)


def test_sideways_cut_does_not_give_thickness():
    inv = Inventory()
    inv.add("A", C("f1", (0, 0, 1.4, 24), cut={"left"}))  # only part of its width visible
    assert inv.books[0].dimensions_cm() == (24.0, None)


def test_neighbouring_columns_stay_separate_even_when_partial():
    inv = Inventory()
    inv.add("A", C("f1", (10, 0, 13, 14), cut={"bottom"}))
    inv.add("A", C("f2", (13.2, 9, 16, 24), cut={"top"}))
    assert len(inv.books) == 2
