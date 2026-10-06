"""Replacement cost for non-book items, from retrievable listings only.

- Brand/model legible  -> "priced": median of listings for that model (low = high).
- Brand/model unknown  -> "range":  interquartile range of listings for a
                          description + material query, so the range reflects
                          the market rather than one merchant.
- Art / portraits      -> "needs_appraisal", unless the policyholder said it is a print.
- No listings          -> price left empty; the line is excluded from totals.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..config import Locale
from ..schemas import PriceRange
from .pricing import Listing, now


@dataclass
class ItemPricing:
    status: str  # priced | range | needs_appraisal
    price: PriceRange
    notes: list[str] = field(default_factory=list)


def _trim_outliers(listings: list[Listing]) -> list[Listing]:
    """Drop listings far outside the pack (accessories, bulk lots, mislisted items)."""
    if len(listings) < 4:
        return listings
    amounts = sorted(l.amount for l in listings)
    q1, _, q3 = statistics.quantiles(amounts, n=4)
    spread = q3 - q1
    return [l for l in listings if q1 - 1.5 * spread <= l.amount <= q3 + 1.5 * spread]


def _range(listings: list[Listing], basis: str, retrieved_at: str) -> PriceRange:
    amounts = sorted(l.amount for l in listings)
    if len(amounts) >= 4:
        low, _, high = statistics.quantiles(amounts, n=4)
    else:
        low, high = amounts[0], amounts[-1]
    closest = min(listings, key=lambda l: abs(l.amount - statistics.median(amounts)))
    return PriceRange(
        low=round(low, 2), high=round(high, 2), currency=listings[0].currency,
        source=", ".join(sorted({l.merchant for l in listings if l.merchant})) or "Google Shopping",
        url=closest.url, retrieved_at=closest.retrieved_at or retrieved_at, basis=basis,
        listings=[{"title": l.title, "amount": l.amount, "merchant": l.merchant, "url": l.url} for l in listings[:10]],
    )


async def price_item(
    *, category: str, description: str, material: str, brand_model: str, is_artwork: bool,
    user_says_print: bool, locale: Locale, client,
) -> ItemPricing:
    if is_artwork and not user_says_print:
        return ItemPricing("needs_appraisal", PriceRange(), ["art: original or print not confirmed by the policyholder"])

    retrieved_at = now()
    if brand_model:
        listings = _trim_outliers([l for l in await client.shopping(brand_model, locale) if l.condition == "new"])
        if listings:
            median = round(statistics.median(l.amount for l in listings), 2)
            price = _range(listings, f"median of {len(listings)} new listing(s) for '{brand_model}'", retrieved_at)
            price.low = price.high = median
            return ItemPricing("priced", price)

    query = " ".join(part for part in (material, description or category) if part)
    if is_artwork:
        query += " print"
    listings = _trim_outliers([l for l in await client.shopping(query, locale) if l.condition == "new"])
    if not listings:
        return ItemPricing("range", PriceRange(), [f"no listings found for '{query}'"])
    basis = f"interquartile range of {len(listings)} new listing(s) for '{query}'"
    notes = ["brand/model not legible: priced as a market range"]
    if brand_model:
        notes.insert(0, f"no listings for brand/model '{brand_model}'")
    return ItemPricing("range", _range(listings, basis, retrieved_at), notes)
