"""The claim packet contract.

Every field the brief requires is here; extra fields carry evidence. Rule of
thumb used throughout: an empty value (None / "") means "the system does not
know". Nothing is filled without evidence behind it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Sweep(BaseModel):
    id: str
    captured_at: str
    device: str = ""
    duration_s: float = 0
    country: str
    currency: str


class Room(BaseModel):
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    floor_area_m2: float | None = None
    wall_area_m2: float | None = None
    shelved_wall_area_m2: float | None = None
    # Imperial copies, derived in code from the metric figures.
    floor_area_ft2: float | None = None
    wall_area_ft2: float | None = None
    shelved_wall_area_ft2: float | None = None
    shape: str = ""
    scale_method: str = ""
    confidence: float = 0
    evidence: list[str] = Field(default_factory=list)


class Price(BaseModel):
    """One priced figure. Without a URL it is not a price, so `amount` stays None."""

    amount: float | None = None
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False
    # When converted: what it was converted from, and the dated rate used.
    original_amount: float | None = None
    original_currency: str = ""
    fx_rate: float | None = None
    fx_source: str = ""
    condition_assumed: str = ""
    # How the figure was picked from the listings (e.g. "median of 4 new listings").
    basis: str = ""


BookStatus = Literal["identified", "unidentified", "needs_appraisal"]


class Book(BaseModel):
    id: str
    shelf: str = ""
    position: int = 0
    frame_ref: str = ""
    status: BookStatus = "unidentified"
    title: str = ""
    author: str = ""
    edition: str = ""
    isbn: str = ""
    publisher: str = ""
    orientation: Literal["upright", "flat", ""] = ""
    spine_text: str = ""
    spine_height_cm: float | None = None
    spine_thickness_cm: float | None = None
    dimension_method: str = ""
    id_confidence: float = 0
    id_source: str = ""
    id_url: str = ""
    replacement_cost: Price = Field(default_factory=Price)
    used_value: Price = Field(default_factory=Price)
    # The policyholder said these are not theirs: listed for the record, not claimed.
    excluded: bool = False
    notes: list[str] = Field(default_factory=list)


class Dimensions(BaseModel):
    w: float | None = None
    h: float | None = None
    d: float | None = None


class PriceRange(BaseModel):
    low: float | None = None
    high: float | None = None
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    basis: str = ""
    listings: list[dict] = Field(default_factory=list)


ItemStatus = Literal["priced", "range", "needs_appraisal"]


class Item(BaseModel):
    id: str
    category: str
    description: str = ""
    material: str = ""
    brand_model: str = ""
    frame_ref: str = ""
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    dimension_method: str = ""
    status: ItemStatus = "range"
    replacement_cost: PriceRange = Field(default_factory=PriceRange)
    confidence: float = 0
    notes: list[str] = Field(default_factory=list)


class Totals(BaseModel):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    books_needs_appraisal: int = 0
    books_excluded_by_policyholder: int = 0
    shelf_run_m: float = 0
    books_replacement_cost: float = 0
    books_used_value: float = 0
    # Priced books whose identification awaits review (low confidence): shown, not in the claim total.
    books_pending_review_cost: float = 0
    items_replacement_cost_low: float = 0
    items_replacement_cost_high: float = 0
    excluded_from_totals: int = 0
    currency: str = ""


class ReviewEntry(BaseModel):
    ref_id: str
    reason: str


class ClaimPacket(BaseModel):
    sweep: Sweep
    room: Room = Field(default_factory=Room)
    books: list[Book] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    review_queue: list[ReviewEntry] = Field(default_factory=list)
    # Second-locale pricing of the same sample, to show locale is a setting.
    locale_comparison: dict | None = None
    stages: dict = Field(default_factory=dict)
