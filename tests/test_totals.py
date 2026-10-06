from library_claim.schemas import Book, ClaimPacket, Item, Price, PriceRange, Sweep
from library_claim.totals import finalize


def _packet(books=(), items=()):
    return ClaimPacket(sweep=Sweep(id="s", captured_at="t", country="IN", currency="INR"), books=list(books), items=list(items))


def priced(i, amount, used=None, **kw):
    return Book(id=f"B{i}", status="identified", id_confidence=0.95, spine_height_cm=20, spine_thickness_cm=2.5,
                replacement_cost=Price(amount=amount, url="u", source="s", retrieved_at="d"),
                used_value=Price(amount=used, url="u" if used else "", source="s"), **kw)


def test_totals_add_up_from_lines_only():
    packet = finalize(_packet([priced(1, 300, 120), priced(2, 450), Book(id="B3", spine_thickness_cm=3.0)]))
    t = packet.totals
    assert (t.book_count, t.books_identified, t.books_unidentified) == (3, 2, 1)
    assert t.books_replacement_cost == 750 and t.books_used_value == 120
    assert t.excluded_from_totals == 1
    assert t.shelf_run_m == 0.08


def test_unsourced_amount_is_never_counted():
    sneaky = Book(id="B1", status="identified", replacement_cost=Price(amount=999, url=""))
    assert finalize(_packet([sneaky])).totals.books_replacement_cost == 0


def test_excluded_shelf_is_listed_but_not_claimed():
    packet = finalize(_packet([priced(1, 300, excluded=True), priced(2, 200)]))
    assert packet.totals.books_replacement_cost == 200
    assert packet.totals.books_excluded_by_policyholder == 1
    assert any("Excluded by the policyholder" in r.reason for r in packet.review_queue)


def test_items_ranges_and_appraisals():
    items = [
        Item(id="I1", category="lighting", status="range", confidence=0.8, replacement_cost=PriceRange(low=1500, high=3000, url="u")),
        Item(id="I2", category="art", status="needs_appraisal", confidence=0.9),
        Item(id="I3", category="decor", status="range", confidence=0.6),
    ]
    packet = finalize(_packet(items=items))
    assert (packet.totals.items_replacement_cost_low, packet.totals.items_replacement_cost_high) == (1500, 3000)
    assert packet.totals.excluded_from_totals == 2
    reasons = {r.ref_id: r.reason for r in packet.review_queue}
    assert "appraisal" in reasons["I2"]


def test_room_without_measurement_is_reviewed():
    packet = finalize(_packet())
    assert any(r.ref_id == "room" for r in packet.review_queue)
