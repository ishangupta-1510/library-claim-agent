"""Settings from the environment. Locale is data here, never hardcoded in stages."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Locale:
    country: str  # ISO 3166-1 alpha-2
    currency: str  # ISO 4217
    google_gl: str  # Google Shopping country
    google_hl: str
    google_domain: str
    ebay_domain: str  # for used listings, where a local eBay exists


# Words that confirm a country when the policyholder says them (country, people, currency).
LOCALE_WORDS: dict[str, tuple[str, ...]] = {
    "IN": ("india", "indian", "bharat", "rupee", "rupees", "inr"),
    "US": ("united states", "america", "american", "usa", "dollar", "dollars", "usd"),
    "GB": ("united kingdom", "uk", "britain", "british", "england", "scotland", "wales", "pound", "pounds", "gbp"),
}


def confirms_country(code: str, words: str) -> bool:
    """The policyholder's words name this country or its currency (whole words, any case)."""
    said = " " + " ".join(re.findall(r"[a-z]+", words.lower())) + " "
    if code == "US" and re.search(r"\b(US|U\.S\.?)(?![a-z])", words):
        return True  # "the US", "U.S." -- but not the pronoun "us"
    return any(f" {w} " in said for w in LOCALE_WORDS.get(code, ()))


LOCALES: dict[str, Locale] = {
    "IN": Locale("IN", "INR", "in", "en", "google.co.in", ""),
    "US": Locale("US", "USD", "us", "en", "google.com", "ebay.com"),
    "GB": Locale("GB", "GBP", "uk", "en", "google.co.uk", "ebay.co.uk"),
}


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    serpapi_key: str
    live_model: str
    vision_model: str
    country: str
    compare_country: str
    marker_size_cm: float
    appraisal_threshold: float
    appraisal_threshold_currency: str  # the threshold is converted at the day's rate for claims in other currencies
    sweeps_dir: Path
    price_search_budget: int | None
    record_conversation: bool = False  # save the spoken conversation as conversation.wav (off unless chosen)

    @property
    def locale(self) -> Locale:
        return locale_for(self.country)


def locale_for(country: str) -> Locale:
    try:
        return LOCALES[country.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported country {country!r}; add it to LOCALES in config.py") from exc


@lru_cache
def settings() -> Settings:
    cfg = Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        serpapi_key=os.getenv("SERPAPI_KEY", ""),
        live_model=os.getenv("LIVE_MODEL", "gemini-3.8-live"),
        vision_model=os.getenv("VISION_MODEL", "gemini-3.5-flash-lite"),
        country=os.getenv("COUNTRY", "IN"),
        compare_country=os.getenv("COMPARE_COUNTRY", "US"),
        marker_size_cm=float(os.getenv("MARKER_SIZE_CM", "15.0")),
        appraisal_threshold=float(os.getenv("APPRAISAL_THRESHOLD", "10000")),
        appraisal_threshold_currency=os.getenv("APPRAISAL_THRESHOLD_CURRENCY", "INR").upper(),
        sweeps_dir=Path(os.getenv("SWEEPS_DIR", "sweeps")),
        price_search_budget=int(os.environ["PRICE_SEARCH_BUDGET"]) if os.getenv("PRICE_SEARCH_BUDGET") else None,
        record_conversation=os.getenv("RECORD_CONVERSATION", "") == "1",
    )
    # Fail at startup, not after a policyholder has walked the whole room.
    locale_for(cfg.country)
    locale_for(cfg.compare_country)
    return cfg
