"""Book prices from retrievable listings only.

Every figure this stage returns carries the listing it came from (merchant,
URL, retrieval time) and how it was picked. A language model never produces a
number here. Outcomes per figure, in order of preference:

1. Local listings in the claim currency           -> converted = False
2. Listings from the comparison market, converted  -> converted = True, with dated FX rate
3. Nothing found                                   -> amount None, excluded from totals

Rare / signed / antiquarian books, and anything at or above the appraisal
threshold, are flagged for a human and not priced.

Only like-kind listings count: the same title as a whole phrase, a physical
copy, one item, not a collectible copy, with a link. A used value is never
above the replacement cost; used listings above it are collectible copies.
"""

from __future__ import annotations

import hashlib
import json
import re
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


def _words(text: str) -> str:
    """Lowercase words separated by single spaces, padded so phrases match on word boundaries."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


# A listing for one of these is not a like-kind copy of the book on the shelf.
NOT_THE_BOOK = (
    # other works that share the title
    "summary", "study guide", "cliffsnotes", "workbook", "critical perspective", "companion", "discussion guide",
    # other formats and products
    "audiobook", "audio cd", "audio book", "ebook", "e book", "kindle edition", "mp3", "poster", "bookmark",
    # several items in one listing
    "box set", "lot", "set of", "books set", "bundle", "combo", "collection",
    # translations (the spine was read in the language it is printed in)
    "hindi", "marathi", "telugu", "tamil", "malayalam", "kannada", "bengali", "gujarati", "urdu", "punjabi",
    "spanish", "french", "german", "farsi", "persian", "arabic", "translated",
)
# Collectible copies: priced for the copy, not the text, so never a replacement or a used reading copy.
COLLECTIBLE = (*RARITY_WORDS, "1st ed", "first uk edition", "first us edition", "autographed", "rare", "collectible",
               "leather bound")

# What may follow the title in a listing for the same book: a subtitle, a bracket, a dash, "by <author>",
# the author's name (eBay: "The Selfish Gene Richard Dawkins Book") or a format.
_SEPARATOR = r"\s*$|\s*[:(\[\-–—|,/.]"
_FOLLOWERS = ("by", "paperback", "hardcover", "hardback", "mass market", "edition", "novel", "book")


def relevant(listing_title: str, title: str, author: str) -> bool:
    """A listing is a like-kind copy of this book.

    The listing must start with the book's main title (before any subtitle,
    leading article optional), followed only by a subtitle, bracket, dash,
    "by <author>" or format. Containing the title anywhere was not enough:
    "Deep Work" matched "JDM Deep rim WORK" wheels, and "The Midnight Library"
    matched "Tales from the Midnight Library" and two-book combos. A "by"
    naming someone else is another book with the same title. Other formats,
    translations, multi-item lots and collectible copies are rejected.
    """
    main = re.sub(r"^\s*(the|a|an)\s+", "", title.split(":")[0].strip(), flags=re.I)
    words = re.findall(r"[a-z0-9]+", main.lower())
    if not words:
        return False
    followers = "|".join(map(re.escape, (*_FOLLOWERS, *author.lower().split())))
    head = (r"^\W*(?:(?:the|a|an)\W+)?" + r"[\W_]+".join(map(re.escape, words))
            + rf"(?={_SEPARATOR}|\s+(?:{followers})\b)")
    if not re.search(head, listing_title, flags=re.I):
        return False
    by = re.search(r"\bby\s+([^,(\[|:]+)", listing_title, flags=re.I)
    surname = author.split()[-1].lower() if author.split() else ""
    if by and surname and fuzz.partial_ratio(surname, by.group(1).lower()) < 80:
        return False
    listing, own = _words(listing_title), _words(title)  # own: "The Cambridge Companion to ..." is the book itself
    return not any(_words(w) in listing and _words(w) not in own for w in (*NOT_THE_BOOK, *COLLECTIBLE))


def pick(listings: list[Listing], condition: str, ceiling: float | None = None) -> tuple[float, list[Listing]] | None:
    """Median of matching listings, so one outlier merchant cannot set the price.

    Listings without a link are not evidence and are dropped. `ceiling` drops
    listings above it: a used reading copy cannot cost more than a new one, so
    used listings above the replacement price are collectible copies.
    """
    chosen = [l for l in listings if l.condition == condition and l.amount > 0 and l.url
              and (ceiling is None or l.amount <= ceiling)]
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
    replacement = _price_from(new, "new", retrieved_at) if new else Price()
    used_value = Price()
    out_notes = []

    # A used copy's value is bounded by the replacement cost, so it is only
    # priced when there is a replacement to bound it; listings above that
    # bound are collectible copies, not this one.
    if replacement.amount is not None:
        used = pick(listings, "used", ceiling=replacement.amount)
        if used:
            used_value = _price_from(used, "used, good", retrieved_at)
        elif fallback.ebay_domain:
            rate = await client.fx(fallback.currency, locale.currency)
            foreign = [l for l in await client.ebay_used(query, fallback) if relevant(l.title, title, author)]
            picked = pick(foreign, "used", ceiling=replacement.amount / rate.rate) if rate else None
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
    if used_value.amount is None and replacement.amount is not None:
        out_notes.append("no used listing at or below the replacement price")

    if replacement.amount is not None and replacement.amount >= threshold:
        appraisal = Appraisal(True, [f"replacement {replacement.amount:.0f} {locale.currency} at or above appraisal threshold {threshold:.0f}"])
        return BookPrices(Price(), Price(), appraisal, ["not auto-priced: " + appraisal.reasons[0]])
    return BookPrices(replacement, used_value, appraisal, out_notes)
