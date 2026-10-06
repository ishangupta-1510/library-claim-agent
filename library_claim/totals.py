"""Totals and the review queue, computed in code from the line items.

No language model ever writes a total. Each rule for what counts is spelled
out here so an adjuster can reproduce every figure from the lines.
"""

from __future__ import annotations

from .schemas import Book, ClaimPacket, Item, ReviewEntry, Totals

# Lines with confidence below this go to the review queue.
REVIEW_CONFIDENCE = 0.75


def _money(value: float) -> float:
    return round(value + 0.0, 2)


def book_counts_in_total(book: Book) -> bool:
    """A book contributes money only when identified AND sourced in the claim currency."""
    return (
        book.status == "identified"
        and book.replacement_cost.amount is not None
        and bool(book.replacement_cost.url)
    )


def compute_totals(packet: ClaimPacket) -> Totals:
    books, items = packet.books, packet.items
    excluded = 0
    replacement = used = 0.0

    for book in books:
        if book_counts_in_total(book):
            replacement += book.replacement_cost.amount or 0
            if book.used_value.amount is not None and book.used_value.url:
                used += book.used_value.amount
        else:
            excluded += 1

    low = high = 0.0
    for item in items:
        price = item.replacement_cost
        if item.status == "needs_appraisal" or price.low is None or not price.url:
            excluded += 1
            continue
        low += price.low
        high += price.high if price.high is not None else price.low

    thickness = [b.spine_thickness_cm for b in books if b.spine_thickness_cm is not None]
    return Totals(
        book_count=len(books),
        books_identified=sum(b.status == "identified" for b in books),
        books_unidentified=sum(b.status == "unidentified" for b in books),
        books_needs_appraisal=sum(b.status == "needs_appraisal" for b in books),
        # Shelf run = summed spine thickness of measured books, in metres.
        shelf_run_m=round(sum(thickness) / 100, 2),
        books_replacement_cost=_money(replacement),
        books_used_value=_money(used),
        items_replacement_cost_low=_money(low),
        items_replacement_cost_high=_money(high),
        excluded_from_totals=excluded,
        currency=packet.sweep.currency,
    )


def _book_reasons(book: Book) -> list[str]:
    reasons = []
    if book.status == "unidentified":
        reasons.append("Spine not readable enough to identify; logged with dimensions only")
    if book.status == "needs_appraisal":
        reasons.append("Flagged for human appraisal (" + "; ".join(book.notes or ["rare or high value"]) + ")")
    if book.status == "identified" and book.id_confidence < REVIEW_CONFIDENCE:
        reasons.append(f"Identification confidence {book.id_confidence:.2f} below {REVIEW_CONFIDENCE}")
    if book.status == "identified" and book.replacement_cost.amount is None:
        reasons.append("No retrievable price found; excluded from totals")
    if book.replacement_cost.converted:
        reasons.append("Replacement cost converted from another market")
    if book.spine_height_cm is None or book.spine_thickness_cm is None:
        reasons.append("Spine dimensions not measurable (no metric scale in frame)")
    return reasons


def _item_reasons(item: Item) -> list[str]:
    reasons = []
    if item.status == "needs_appraisal":
        reasons.append("Art or unique item: needs appraisal, not auto-priced")
    if item.status != "needs_appraisal" and item.replacement_cost.low is None:
        reasons.append("No retrievable price found; excluded from totals")
    if item.status == "range" and item.replacement_cost.low is not None:
        reasons.append("Brand/model unknown: priced as a sourced range")
    if item.confidence < REVIEW_CONFIDENCE:
        reasons.append(f"Detection confidence {item.confidence:.2f} below {REVIEW_CONFIDENCE}")
    return reasons


def build_review_queue(packet: ClaimPacket) -> list[ReviewEntry]:
    queue: list[ReviewEntry] = []
    for book in packet.books:
        queue += [ReviewEntry(ref_id=book.id, reason=r) for r in _book_reasons(book)]
    for item in packet.items:
        queue += [ReviewEntry(ref_id=item.id, reason=r) for r in _item_reasons(item)]
    room = packet.room
    if room.floor_area_m2 is None:
        queue.append(ReviewEntry(ref_id="room", reason="Room not measured: no metric room geometry captured"))
    elif room.confidence < REVIEW_CONFIDENCE:
        queue.append(ReviewEntry(ref_id="room", reason=f"Room measurement confidence {room.confidence:.2f}"))
    return queue


def finalize(packet: ClaimPacket) -> ClaimPacket:
    """Recompute everything derived; safe to call any number of times."""
    packet.totals = compute_totals(packet)
    packet.review_queue = build_review_queue(packet)
    return packet
