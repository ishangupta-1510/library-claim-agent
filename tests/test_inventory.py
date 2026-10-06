import pytest

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


def test_a_cut_end_makes_the_reading_partial_but_a_cut_side_does_not():
    assert C("f", (0, 0, 3, 20), cut={"bottom"}).partial
    assert not C("f", (0, 0, 3, 20), cut={"left"}).partial
    assert C("f", (0, 0, 20, 3), cut={"left"}, orientation="flat").partial


def test_a_whole_reading_beats_a_partial_one():
    inv = Inventory()
    book, _ = inv.add("A", C("f1", (10, 0, 13, 24), cut={"bottom"}, title="TRAIN TO PAKISTAN AND MORE", legible=True))
    inv.add("A", C("f2", (10, 0, 13, 24), title="Train to Pakistan", legible=True))
    assert book.best.frame_id == "f2"


def test_box_shape_overrides_a_wrong_orientation_label():
    from library_claim.stages.inventory import orientation_of

    assert orientation_of((10, 0, 12, 20), "flat") == "upright"  # 2 cm wide, 20 cm tall
    assert orientation_of((0, 10, 20, 12.5), "upright") == "flat"
    assert orientation_of((0, 0, 3, 4), "flat") == "flat"  # near square (a heavily cut spine): keep the label


def test_mislabelled_sighting_merges_with_the_same_spine():
    from library_claim.stages.inventory import orientation_of

    inv = Inventory()
    inv.add("A", C("f1", (10, 0, 12, 20), title="The Namesake", legible=True))
    box = (10.1, 0.2, 12.1, 20.1)
    _, new = inv.add("A", C("f2", box, title="The Namesake", legible=True, orientation=orientation_of(box, "flat")))
    assert not new and len(inv.books) == 1


def test_a_cut_box_cannot_prove_flat():
    from library_claim.stages.inventory import orientation_of

    # The top 1.6 cm of an upright spine at the frame's bottom edge: short and wide, labelled flat.
    assert orientation_of((9.3, 84.9, 12.6, 86.5), "upright", frozenset({"bottom"})) == "upright"
    assert orientation_of((9.3, 84.9, 12.6, 86.5), "flat", frozenset({"bottom"})) == "flat"
    assert orientation_of((10, 0, 12, 20), "flat", frozenset({"top"})) == "upright"  # tall even when cut


def test_a_cut_sliver_labelled_flat_joins_its_upright_column():
    inv = Inventory()
    book, _ = inv.add("A", C("f1", (9.2, 64.0, 12.5, 86.6), title="Sapiens", legible=True))
    for frame in ("f2", "f3"):
        _, new = inv.add("A", C(frame, (9.3, 64.1, 12.6, 65.7), cut={"bottom"}, orientation="flat"))  # its top, at the frame edge
        assert not new
    assert len(inv.books) == 1 and book.orientation == "upright"
    assert book.dimensions_cm() == (22.5, 3.3)


def test_a_weakly_seen_repeat_of_a_title_goes_to_review():
    from library_claim.schemas import Book
    from library_claim.stages.inventory import InventoryBook
    from library_claim.sweep import _flag_repeated_titles

    def book(*sightings):
        b = InventoryBook("A")
        b.sightings = list(sightings)
        return b

    strong = book(C("f0", (24, 13, 27, 32), title="The God of Small Things", legible=True),
                  C("f1", (24, 13, 27, 32), title="The God of Small Things", legible=True))
    weak = book(C("f4", (7, 87, 9, 101), cut={"bottom"}, title="The God of Small Things", legible=True))
    records = [Book(id=f"B{i}", status="identified", title="The God of Small Things", id_confidence=1.0) for i in (5, 28)]
    _flag_repeated_titles(records, [strong, weak])
    assert records[0].id_confidence == 1.0
    assert records[1].id_confidence == 0.6 and "B5" in records[1].notes[-1]


def test_one_spine_split_by_one_detection_is_merged():
    inv = Inventory()
    inv.add("A", C("f4", (-0.5, 87.5, 1.9, 100.6), cut={"bottom"}, title="ANCE OF LOSS", legible=True))
    inv.add("A", C("f4", (2.0, 87.9, 3.3, 100.9), cut={"bottom"}))
    for frame in ("f5", "f6", "f7"):
        inv.add("A", C(frame, (-0.5, 87.5, 3.35, 106.0), title="THE INHERITANCE OF LOSS", legible=True))
    assert len(inv.books) == 2
    assert inv.resolve_splits() == 1 and len(inv.books) == 1


def test_two_spines_joined_by_one_detection_stay_apart():
    inv = Inventory()
    for frame in ("f1", "f2", "f3"):
        inv.add("A", C(frame, (4.8, 41.3, 6.6, 69.1)))
        inv.add("A", C(frame, (6.7, 41.3, 9.0, 69.1)))
    inv.add("A", C("f4", (4.8, 50.3, 9.1, 69.1), cut={"top"}))
    assert inv.resolve_splits() == 0 and len(inv.books) == 2


def _shelf_row(frame, xs, titles, shift=0.0):
    return [C(frame, (x + shift, 2, x + 2.5 + shift, 30), title=t, legible=True) for x, t in zip(xs, titles)]


def test_a_frame_offset_by_a_spine_is_moved_back_by_the_titles_it_read():
    inv = Inventory()
    xs, titles = [10, 12.6, 15.2, 17.8], ["Shantaram", "The Alchemist", "Dune", "The Hobbit"]
    inv.add_frame("A", _shelf_row("f1", xs, titles))
    new, (dx, dy) = inv.add_frame("A", _shelf_row("f2", xs, titles, shift=-1.7))
    assert new == 0 and dx == pytest.approx(1.7) and dy == 0
    assert len(inv.books) == 4 and all(len(b.sightings) == 2 for b in inv.books)


def test_a_mislocated_box_is_placed_by_its_reading_but_not_measured():
    inv = Inventory()
    inv.add_frame("A", _shelf_row("f1", [30.1], ["The Midnight Library"]))
    wide = C("f2", (39.4, 4.6, 50.6, 32.2), cut={"top"}, title="THE MIDNIGHT LIBRARY", legible=True)
    new, _ = inv.add_frame("A", [wide])
    assert new == 0 and len(inv.books) == 1
    assert not wide.box_trusted and inv.books[0].dimensions_cm() == (28.0, 2.5)


def test_the_same_title_twice_in_one_frame_is_two_copies():
    inv = Inventory()
    new, _ = inv.add_frame("A", _shelf_row("f1", [10, 30], ["Dune", "Dune"]))
    assert new == 2 and len(inv.books) == 2


def test_a_box_on_a_neighbour_that_reads_otherwise_follows_its_reading():
    inv = Inventory()
    inv.add_frame("A", _shelf_row("f1", [10, 12.6], ["Shantaram", "The Alchemist"]))
    stray = C("f2", (10.3, 2, 12.8, 30), title="The Alchemist", legible=True)
    inv.add_frame("A", [stray])
    alchemist = next(b for b in inv.books if b.best.title == "The Alchemist")
    assert stray in alchemist.sightings and not stray.box_trusted


def test_one_box_running_into_the_next_row_does_not_stretch_the_spine():
    inv = Inventory()
    for frame, box, cut in (("f3", (12.6, 78.9, 15.1, 86.6), {"bottom"}), ("f4", (12.6, 78.8, 15.0, 101.5), {"bottom"}),
                            ("f5", (12.6, 78.8, 15.0, 106.6), set()), ("f6", (12.6, 82.0, 15.0, 106.5), {"top"}),
                            ("f7", (12.6, 97.0, 15.0, 120.0), {"top"})):
        inv.add("A", C(frame, box, cut=cut))
    book = inv.books[0]
    assert len(inv.books) == 1 and book.dimensions_cm()[0] == pytest.approx(27.8, abs=0.1)
    assert book.box[3] < 107  # the row below is not inside this book's box
