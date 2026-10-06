"""Settings from the environment. Locale is data here, never hardcoded in stages."""

from __future__ import annotations

import os
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
    sweeps_dir: Path
    price_search_budget: int | None

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
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        serpapi_key=os.getenv("SERPAPI_KEY", ""),
        live_model=os.getenv("LIVE_MODEL", "gemini-3.8-live"),
        vision_model=os.getenv("VISION_MODEL", "gemini-3.5-flash"),
        country=os.getenv("COUNTRY", "IN"),
        compare_country=os.getenv("COMPARE_COUNTRY", "US"),
        marker_size_cm=float(os.getenv("MARKER_SIZE_CM", "15.0")),
        appraisal_threshold=float(os.getenv("APPRAISAL_THRESHOLD", "10000")),
        sweeps_dir=Path(os.getenv("SWEEPS_DIR", "sweeps")),
        price_search_budget=int(os.environ["PRICE_SEARCH_BUDGET"]) if os.getenv("PRICE_SEARCH_BUDGET") else None,
    )
