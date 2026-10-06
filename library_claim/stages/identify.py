"""Resolve a read spine to a specific work (and edition only where evidenced).

Input is only what was actually read off the spine. Candidates come from
open catalogs (Google Books, Open Library). A match is accepted only when the
catalog record agrees with the spine text strongly; otherwise the book stays
"unidentified". A blank beats a confident wrong answer.

Edition/ISBN are filled only when the spine shows a publisher that matches the
candidate edition's publisher. Otherwise the work is identified, but the
edition is left empty.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

import httpx
from rapidfuzz import fuzz

from .. import net

logger = logging.getLogger("library_claim.identify")

GOOGLE_BOOKS = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY = "https://openlibrary.org/search.json"

# Acceptance thresholds (0-100 fuzzy scores). Chosen to favour blanks over
# wrong answers; see tests for the cases they separate.
TITLE_ACCEPT = 88
AUTHOR_ACCEPT = 80
TITLE_ONLY_ACCEPT = 95  # no author legible: demand a near-exact, distinctive title
MIN_TITLE_WORDS_WITHOUT_AUTHOR = 2


@dataclass
class SpineReading:
    """What the vision stage read. Empty strings mean 'not legible'."""

    title: str = ""
    author: str = ""
    publisher: str = ""
    raw_text: str = ""
    # Read from a spine partly outside the frame: the title may be truncated.
    partial: bool = False


@dataclass
class Candidate:
    title: str
    authors: list[str]
    publisher: str
    isbn: str
    year: str
    source: str
    url: str
    subtitle: str = ""
    # Other spellings of the authors' names. Open Library often lists only the
    # native script ("村上春樹"), with the Latin spelling printed on the spine here.
    author_aliases: list[str] = field(default_factory=list)


@dataclass
class Identification:
    status: str  # "identified" | "unidentified"
    confidence: float = 0.0
    title: str = ""
    author: str = ""
    publisher: str = ""
    edition: str = ""
    isbn: str = ""
    year: str = ""
    source: str = ""
    url: str = ""
    reasons: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    # Fold accents ("Héctor García" -> "hector garcia"): spines and catalogs disagree on them,
    # and dropping the letters outright broke author matching.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_leading_article(text: str) -> str:
    return re.sub(r"^(the|a|an) ", "", text)


def title_score(spine_title: str, candidate: Candidate) -> float:
    spine = _strip_leading_article(_norm(spine_title))
    # Catalogs often fold the subtitle or a bracketed note into the title
    # ("Man's Search for Meaning : An Introduction to Logotherapy"); spines print the main title.
    options = [candidate.title, _main_title(candidate.title), f"{candidate.title} {candidate.subtitle}".strip()]
    return max(
        max(fuzz.ratio(spine, _strip_leading_article(_norm(o))), fuzz.token_sort_ratio(spine, _norm(o)))
        for o in options
    )


def _author_match(spine_author: str, name: str) -> float:
    spine, name = _norm(spine_author), _norm(name)
    if not name:
        return 0.0
    surname = name.split(" ")[-1]
    # Spines often print only the surname ("ORWELL"); accept surname-exact as strong.
    surname_hit = 100.0 if spine.split(" ")[-1:] == [surname] else 0.0
    return max(fuzz.token_set_ratio(spine, name), surname_hit)


def author_score(spine_author: str, candidate: Candidate) -> float:
    names = [*candidate.authors, *candidate.author_aliases]
    return max((_author_match(spine_author, n) for n in names), default=0.0)


def _main_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", "", title).split(":")[0]).strip()


def display_title(candidate: Candidate) -> str:
    """The title as printed on the book: without a subtitle or catalog note folded into it."""
    return _main_title(candidate.title) or candidate.title


def display_authors(reading_author: str, candidate: Candidate) -> str:
    """Authors as the packet shows them: a name with no Latin letters is replaced by the alias the spine matched."""
    shown = []
    for author in candidate.authors:
        if not _norm(author) and candidate.author_aliases:
            # Best match, then the spelling closest to the spine's word order, then not all-caps.
            author = max(candidate.author_aliases, key=lambda a: (
                _author_match(reading_author, a), fuzz.ratio(_norm(reading_author), _norm(a)), not a.isupper()))
        shown.append(author)
    return ", ".join(shown)


def score(reading: SpineReading, candidate: Candidate) -> tuple[bool, float, list[str]]:
    """Whether to accept a candidate, the confidence, and why."""
    t = title_score(reading.title, candidate)
    reasons = [f"title match {t:.0f}"]
    if reading.author.strip():
        a = author_score(reading.author, candidate)
        reasons.append(f"author match {a:.0f}")
        accepted = t >= TITLE_ACCEPT and a >= AUTHOR_ACCEPT
        confidence = round(min(t, a) / 100, 2)
    else:
        words = len(_norm(reading.title).split())
        # A truncated title can itself be a real title, so a partial reading needs the author too.
        accepted = t >= TITLE_ONLY_ACCEPT and words >= MIN_TITLE_WORDS_WITHOUT_AUTHOR and not reading.partial
        reasons.append("no author legible; title-only match")
        # Title-only matches are capped so they always get human review.
        confidence = round(min(t / 100, 0.7), 2)
    if reading.partial:
        reasons.append("read from a spine partly outside the frame")
    return accepted, confidence, reasons


def _google_candidates(payload: dict) -> list[Candidate]:
    out = []
    for item in payload.get("items", []) or []:
        info = item.get("volumeInfo", {})
        isbns = {i.get("type"): i.get("identifier") for i in info.get("industryIdentifiers", []) or []}
        out.append(Candidate(
            title=info.get("title", ""),
            subtitle=info.get("subtitle", ""),
            authors=info.get("authors", []) or [],
            publisher=info.get("publisher", ""),
            isbn=isbns.get("ISBN_13") or isbns.get("ISBN_10") or "",
            year=(info.get("publishedDate") or "")[:4],
            source="Google Books",
            url=info.get("canonicalVolumeLink") or f"https://books.google.com/books?id={item.get('id', '')}",
        ))
    return out


def _openlibrary_candidates(payload: dict) -> list[Candidate]:
    out = []
    for doc in payload.get("docs", []) or []:
        out.append(Candidate(
            title=doc.get("title", ""),
            subtitle=doc.get("subtitle", ""),
            authors=doc.get("author_name", []) or [],
            author_aliases=doc.get("author_alternative_name", []) or [],
            publisher=(doc.get("publisher") or [""])[0],
            isbn="",  # Open Library work records span editions; no edition-level ISBN here
            year=str(doc.get("first_publish_year") or ""),
            source="Open Library",
            url=f"https://openlibrary.org{doc.get('key', '')}",
        ))
    return out


class Catalogs:
    """The catalog lookups for one packet, with what the run learned about each catalog.

    Google Books is dropped for the rest of the run once it is unusable (its
    key rejected and keyless access rate-limited, or it keeps failing), so a
    sweep of 60 books does not pay the retry backoff 60 times. Open Library
    is always tried. `events` records what happened, for the stage report.
    """

    def __init__(self, http: httpx.AsyncClient, google_api_key: str = ""):
        self.http = http
        # Keyless access shares a global daily quota that is routinely exhausted (HTTP 429), so use a key if given.
        self.google_key = google_api_key
        self.google_available = True
        self.events: list[str] = []

    async def google(self, query: str) -> list[Candidate]:
        if not self.google_available:
            return []
        params = {"q": query, "maxResults": 10, "printType": "books"}
        if self.google_key:
            # The key goes in a header, not the URL, so it never appears in logs or recorded URLs.
            response = await net.get(self.http, GOOGLE_BOOKS, params=params, headers={"x-goog-api-key": self.google_key})
            if response is not None and response.status_code in (400, 401, 403):
                # The key is not enabled for the Books API (e.g. a Gemini-only key).
                self.google_key = ""
                self._event(f"Google Books rejected the API key (HTTP {response.status_code}); using keyless access")
            else:
                return self._google_result(response)
        return self._google_result(await net.get(self.http, GOOGLE_BOOKS, params=params))

    def _google_result(self, response: httpx.Response | None) -> list[Candidate]:
        if response is not None and response.status_code == 200:
            return _google_candidates(response.json())
        if self.google_available:
            # net.get already retried; a catalog that is still failing is skipped for the rest of the run.
            status = "no response" if response is None else f"HTTP {response.status_code}"
            self.google_available = False
            self._event(f"Google Books unavailable ({status}); identified from Open Library only for this run")
        return []

    async def open_library(self, title: str, author: str) -> list[Candidate]:
        params = {"title": title, "limit": 10,
                  "fields": "key,title,subtitle,author_name,author_alternative_name,publisher,first_publish_year"}
        if author:
            params["author"] = author
        response = await net.get(self.http, OPEN_LIBRARY, params=params)
        return _openlibrary_candidates(response.json()) if response is not None and response.status_code == 200 else []

    def _event(self, message: str) -> None:
        if message not in self.events:
            logger.warning(message)
            self.events.append(message)


async def fetch_candidates(reading: SpineReading, catalogs: Catalogs) -> list[Candidate]:
    """Candidates from both catalogs. A catalog that errors is skipped, not fatal."""
    terms = [f'intitle:"{reading.title}"'] + ([f'inauthor:"{reading.author}"'] if reading.author else [])
    return [*await catalogs.google(" ".join(terms)), *await catalogs.open_library(reading.title, reading.author)]


def choose(reading: SpineReading, candidates: list[Candidate]) -> Identification:
    if not reading.title.strip():
        return Identification("unidentified", reasons=["no legible title on spine"])
    scored = []
    for cand in candidates:
        accepted, confidence, reasons = score(reading, cand)
        if accepted:
            scored.append((confidence, cand, reasons))
    if not scored:
        return Identification("unidentified", reasons=["no catalog record matched the spine text closely enough"])

    scored.sort(key=lambda s: s[0], reverse=True)
    confidence, best, reasons = scored[0]
    result = Identification(
        "identified", confidence, display_title(best), display_authors(reading.author, best), source=best.source, url=best.url,
        year=best.year, reasons=reasons,
    )
    # Edition only when the spine's publisher matches a specific edition's record.
    if reading.publisher.strip():
        for conf, cand, _ in scored:
            if cand.isbn and cand.publisher and fuzz.token_set_ratio(_norm(reading.publisher), _norm(cand.publisher)) >= 85:
                result.publisher, result.isbn, result.year = cand.publisher, cand.isbn, cand.year
                result.edition = f"{cand.publisher} edition" + (f" ({cand.year})" if cand.year else "")
                result.source, result.url = cand.source, cand.url
                result.reasons.append("edition matched on spine publisher")
                break
    return result


@dataclass
class Edition:
    publisher: str
    isbn: str
    year: str
    language: str
    url: str


def _publisher_matches(spine_publisher: str, publisher: str) -> bool:
    return fuzz.token_set_ratio(_norm(spine_publisher), _norm(publisher)) >= 85


def pin_edition(result: Identification, spine_publisher: str, editions: list[Edition]) -> Identification:
    """Narrow an identified work to an edition using the publisher read off the spine.

    One matching English edition: fill ISBN. Several: record the publisher but
    leave the ISBN empty, because the spine cannot tell the printings apart.
    """
    matches = [e for e in editions if _publisher_matches(spine_publisher, e.publisher) and e.language in ("", "eng")]
    if not matches:
        result.reasons.append(f"spine publisher {spine_publisher!r} matched no catalogued edition")
        return result
    result.publisher = matches[0].publisher
    if len({e.isbn for e in matches if e.isbn}) == 1:
        only = next(e for e in matches if e.isbn)
        result.isbn, result.year, result.url = only.isbn, only.year, only.url
        result.edition = f"{only.publisher} edition" + (f" ({only.year})" if only.year else "")
        result.reasons.append("edition pinned: the only edition from the spine's publisher")
    else:
        result.edition = f"{result.publisher} (exact printing not visible on spine)"
        result.reasons.append(f"{len(matches)} editions from {result.publisher}; ISBN left empty")
    return result


async def fetch_editions(work_url: str, client: httpx.AsyncClient) -> list[Edition]:
    """All editions of an Open Library work (publisher, ISBN-13, year, language)."""
    response = await net.get(
        client, f"{work_url}/editions.json",
        params={"limit": 1000, "fields": "key,publishers,isbn_13,publish_date,languages"},
    )
    if response is None or response.status_code != 200:
        return []
    editions = []
    for entry in response.json().get("entries", []):
        languages = [lang.get("key", "").split("/")[-1] for lang in entry.get("languages", []) or []]
        year = re.search(r"\d{4}", entry.get("publish_date") or "")
        for publisher in entry.get("publishers", []) or [""]:
            editions.append(Edition(
                publisher=publisher,
                isbn=(entry.get("isbn_13") or [""])[0],
                year=year.group(0) if year else "",
                language=languages[0] if languages else "",
                url=f"https://openlibrary.org{entry.get('key', '')}",
            ))
    return editions


async def identify(reading: SpineReading, catalogs: Catalogs) -> Identification:
    if not reading.title.strip():
        return Identification("unidentified", reasons=["no legible title on spine"])
    result = choose(reading, await fetch_candidates(reading, catalogs))
    if result.status == "identified" and reading.publisher.strip() and not result.isbn and result.source == "Open Library":
        result = pin_edition(result, reading.publisher, await fetch_editions(result.url, catalogs.http))
    return result
