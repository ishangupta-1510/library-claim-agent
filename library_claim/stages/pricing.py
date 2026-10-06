"""Book prices from retrievable listings only.

Every figure this stage returns carries the listing it came from (merchant,
URL, retrieval time) and how it was picked. A language model never produces a
number here. Outcomes per figure, in order of preference:

1. Local listings in the claim currency           -> converted = False
2. Listings from the comparison market, converted  -> converted = True, with dated FX rate
3. Nothing found                                   -> amount None, excluded from totals

Rare / signed / antiquarian books, and anything at or above the appraisal
threshold, are flagged for a human and not priced.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from rapidfuzz import fuzz

from .. import net

from ..config import Locale
from ..schemas import Price

SERPAPI = "https://serpapi.com/search.json"
FX_API = "https://api.frankfurter.dev/v1/latest"

LISTING_TITLE_MATCH = 70  # listing titles are noisy ("1984 (Penguin Modern Classics) Paperback ...")
ANTIQUARIAN_BEFORE = 1950
RARITY_WORDS = ("signed", "first edition", "1st edition", "first printing", "limited edition", "inscribed", "antiquarian")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Listing:
    title: str
    amount: float
    currency: str
    merchant: str
    url: str
    condition: str  # "new" | "used" | ""
    retrieved_at: str = ""  # when the search that returned this listing ran


@dataclass
class Appraisal:
    needed: bool
    reasons: list[str] = field(default_factory=list)


def appraisal_check(title: str, edition_year: str, notes: list[str], spine_text: str, visual_flags: list[str]) -> Appraisal:
    """Decide before pricing whether a human must value this copy.

    Evidence must be about the copy, not the work: a modern paperback of a
    1949 novel is not antiquarian. So the year used is the pinned edition's,
    and age cues come from what the camera saw or what the policyholder said.
    """
    reasons = []
    text = " ".join([title, spine_text, *notes]).lower()
    for word in RARITY_WORDS:
        if word in text:
            reasons.append(f"'{word}' noted")
    if edition_year.isdigit() and int(edition_year) < ANTIQUARIAN_BEFORE:
        reasons.append(f"identified edition is from {edition_year}")
    reasons += [f"looks antiquarian: {flag}" for flag in visual_flags]
    return Appraisal(bool(reasons), reasons)


def relevant(listing_title: str, title: str, author: str) -> bool:
    """A listing is about this book if its title contains the book's title (fuzzily)."""
    score = fuzz.partial_ratio(title.lower(), listing_title.lower())
    if score < LISTING_TITLE_MATCH:
        return False
    # Guard against study guides, summaries and box sets that share the title.
    lowered = listing_title.lower()
    return not any(w in lowered for w in ("summary", "study guide", "cliffsnotes", "box set", "workbook"))


def pick(listings: list[Listing], condition: str) -> tuple[float, list[Listing]] | None:
    """Median of matching listings, so one outlier merchant cannot set the price."""
    chosen = [l for l in listings if l.condition == condition and l.amount > 0]
    if not chosen:
        return None
    return round(statistics.median(l.amount for l in chosen), 2), chosen


def parse_google_shopping(payload: dict, currency: str, retrieved_at: str = "") -> list[Listing]:
    out = []
    for result in payload.get("shopping_results", []) or []:
        amount = result.get("extracted_price")
        if not isinstance(amount, (int, float)):
            continue
        condition = (result.get("second_hand_condition") or "").lower()
        out.append(Listing(
            title=result.get("title", ""),
            amount=float(amount),
            currency=currency,
            merchant=result.get("source", ""),
            url=result.get("product_link") or result.get("link") or "",
            condition="used" if condition in ("used", "pre-owned", "refurbished") else "new",
            retrieved_at=retrieved_at,
        ))
    return out


def parse_ebay(payload: dict, currency: str, retrieved_at: str = "") -> list[Listing]:
    out = []
    for result in payload.get("organic_results", []) or []:
        price = result.get("price") or {}
        amount = price.get("extracted") if isinstance(price, dict) else None
        if not isinstance(amount, (int, float)):
            continue
        condition = (result.get("condition") or "").lower()
        out.append(Listing(
            title=result.get("title", ""),
            amount=float(amount),
            currency=currency,
            merchant="eBay",
            url=result.get("link", ""),
            condition="new" if "brand new" in condition else "used",
            retrieved_at=retrieved_at,
        ))
    return out


@dataclass
class FxRate:
    rate: float
    date: str
    source: str = "European Central Bank via frankfurter.dev"


CACHE_DAYS = 7


class PriceClient:
    """Fetches listings and FX. Keeps every raw response for the audit trail.

    Searches are cached on disk for CACHE_DAYS, keyed by their exact
    parameters. A cached result keeps its original retrieval time, so prices
    are never presented as fresher than they are, and re-running a sweep does
    not spend the search quota again.
    """

    def __init__(
        self, http: httpx.AsyncClient, serpapi_key: str, cache_dir: Path | None = Path(".cache/serpapi"),
        max_live_searches: int | None = None,
    ):
        self.http = http
        self.serpapi_key = serpapi_key
        self.cache_dir = cache_dir
        # Optional cap on paid searches per packet (development runs); None = unlimited.
        self.max_live_searches = max_live_searches
        self.budget_exhausted = False
        self.raw: list[dict] = []  # every search used for this packet, with its response
        self.live_searches = 0
        self.failed_searches = 0
        self._fx: dict[tuple[str, str], FxRate] = {}

    def _cache_path(self, params: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]
        return self.cache_dir / f"{key}.json"

    async def _serpapi(self, params: dict) -> tuple[dict, str]:
        if not self.serpapi_key:
            return {}, ""
        path = self._cache_path(params)
        if path and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            age = datetime.now(timezone.utc) - datetime.fromisoformat(cached["retrieved_at"])
            if age < timedelta(days=CACHE_DAYS):
                self.raw.append({**cached, "from_cache": True})
                return cached["response"], cached["retrieved_at"]
        if self.max_live_searches is not None and self.live_searches >= self.max_live_searches:
            self.budget_exhausted = True
            return {}, ""
        response = await net.get(self.http, SERPAPI, params={**params, "api_key": self.serpapi_key}, timeout=45)
        self.live_searches += 1
        retrieved_at = now()
        if response is None:
            # Kept failing (timeouts): the line goes unpriced and is flagged, the packet still builds.
            self.failed_searches += 1
            self.raw.append({"params": params, "retrieved_at": retrieved_at, "status": "failed", "response": {}})
            return {}, ""
        payload = response.json() if response.status_code == 200 else {"error": response.text[:200]}
        entry = {"params": params, "retrieved_at": retrieved_at, "status": response.status_code, "response": payload}
        self.raw.append(entry)
        if path and response.status_code == 200:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(entry), encoding="utf-8")
        return payload, retrieved_at

    async def shopping(self, query: str, locale: Locale) -> list[Listing]:
        payload, retrieved_at = await self._serpapi({
            "engine": "google_shopping", "q": query, "gl": locale.google_gl, "hl": locale.google_hl,
            "google_domain": locale.google_domain,
        })
        return parse_google_shopping(payload, locale.currency, retrieved_at)

    async def ebay_used(self, query: str, locale: Locale) -> list[Listing]:
        if not locale.ebay_domain:
            return []
        payload, retrieved_at = await self._serpapi(
            {"engine": "ebay", "_nkw": query, "ebay_domain": locale.ebay_domain, "LH_ItemCondition": "3000"})
        return parse_ebay(payload, locale.currency, retrieved_at)

    async def fx(self, base: str, quote: str) -> FxRate | None:
        if (base, quote) not in self._fx:
            response = await net.get(self.http, FX_API, params={"from": base, "to": quote})
            if response is None or response.status_code != 200:
                return None
            payload = response.json()
            self._fx[(base, quote)] = FxRate(float(payload["rates"][quote]), payload["date"])
        return self._fx[(base, quote)]


def _price_from(result: tuple[float, list[Listing]], condition: str, retrieved_at: str) -> Price:
    amount, listings = result
    closest = min(listings, key=lambda l: abs(l.amount - amount))
    merchants = sorted({l.merchant for l in listings if l.merchant})
    return Price(
        amount=amount, currency=listings[0].currency, source=", ".join(merchants) or "Google Shopping",
        url=closest.url, retrieved_at=closest.retrieved_at or retrieved_at, condition_assumed=condition,
        basis=f"median of {len(listings)} {condition} listing(s)",
    )


async def convert(price: Price, target: str, client: PriceClient) -> Price | None:
    rate = await client.fx(price.currency, target)
    if rate is None:
        return None
    return price.model_copy(update={
        "amount": round(price.amount * rate.rate, 2), "currency": target, "converted": True,
        "original_amount": price.amount, "original_currency": price.currency,
        "fx_rate": rate.rate, "fx_source": f"{rate.source}, rate dated {rate.date}",
    })


@dataclass
class BookPrices:
    replacement: Price
    used: Price
    appraisal: Appraisal
    notes: list[str] = field(default_factory=list)


async def price_book(
    *, title: str, author: str, isbn: str, edition_year: str, notes: list[str], spine_text: str,
    visual_flags: list[str], locale: Locale, fallback: Locale, threshold: float, client: PriceClient,
) -> BookPrices:
    appraisal = appraisal_check(title, edition_year, notes, spine_text, visual_flags)
    if appraisal.needed:
        return BookPrices(Price(), Price(), appraisal, ["not auto-priced: " + "; ".join(appraisal.reasons)])

    query = f"{isbn}" if isbn else f"{title} {author} book"
    retrieved_at = now()
    failed_before = getattr(client, "failed_searches", 0)
    listings = [l for l in await client.shopping(query, locale) if relevant(l.title, title, author)]
    new = pick(listings, "new")
    used = pick(listings, "used")
    replacement = _price_from(new, "new", retrieved_at) if new else Price()
    used_value = _price_from(used, "used, good", retrieved_at) if used else Price()
    out_notes = []

    if used_value.amount is None and fallback.ebay_domain:
        foreign = [l for l in await client.ebay_used(query, fallback) if relevant(l.title, title, author)]
        picked = pick(foreign, "used")
        if picked:
            converted = await convert(_price_from(picked, "used, good", now()), locale.currency, client)
            if converted:
                used_value = converted
                out_notes.append(f"used value converted from {fallback.country} eBay listings")
    if replacement.amount is None and getattr(client, "failed_searches", 0) > failed_before:
        out_notes.append("not priced: the price search failed (network); retry before settling")
    elif replacement.amount is None and getattr(client, "budget_exhausted", False):
        out_notes.append("not priced: price-search budget for this run was used up")
    elif replacement.amount is None:
        out_notes.append("no new listing found in local market")
    if used_value.amount is None:
        out_notes.append("no used listing found")

    if replacement.amount is not None and replacement.amount >= threshold:
        appraisal = Appraisal(True, [f"replacement {replacement.amount:.0f} {locale.currency} at or above appraisal threshold {threshold:.0f}"])
        return BookPrices(Price(), Price(), appraisal, ["not auto-priced: " + appraisal.reasons[0]])
    return BookPrices(replacement, used_value, appraisal, out_notes)
