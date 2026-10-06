"""Vision: detect and transcribe spines, detect non-book items.

The model's job is narrow on purpose: find boxes and copy the letters it can
see. It does not identify works, estimate sizes or prices. Identification,
measurement and pricing are separate stages that check its output.

Boxes come back on Gemini's 0-1000 grid as [ymin, xmin, ymax, xmax] and are
converted to pixels here.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Protocol

SPINE_PROMPT = """You are cataloguing books on a shelf for an insurance record.

Find EVERY book whose spine is visible: upright books, books lying flat, and books in stacks.
For each one return its bounding box around the spine, its orientation, and the text printed on it.

Rules for text (this record is used for a claim, so accuracy beats completeness):
- Copy only letters you can actually read on the spine. Do not use knowledge of books to fill in or correct words.
- If part of a word is unreadable, leave that field empty rather than completing it.
- title: the title as printed. author: the author name as printed. publisher: publisher name or logo text if printed.
- all_text: every legible piece of text on the spine, in reading order.
- legible: true only if you could read the title clearly.
- age_cues: visible signs of an old or valuable copy (e.g. "cloth binding with gilt lettering", "leather spine",
  "signed"), or an empty list. Do not guess.
- Count a book even when nothing on it is readable (legible false, empty text).
"""

ITEM_PROMPT = """You are listing the contents of a room for a home contents insurance claim.

List every object that is NOT a book: shelving units, furniture, lamps, rugs, electronics, appliances
(e.g. coffee machine), framed art and portraits, decor, plants in pots.
For each: its bounding box, category, a short description, the main material, and brand_model ONLY if a brand
or model is legibly printed or clearly shown by a logo (otherwise empty). is_artwork is true for paintings,
portraits, framed prints, posters and sculptures. Do not list people, walls, floors, windows or doors.
"""

SPINE_SCHEMA = {
    "type": "object",
    "properties": {
        "books": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                    "orientation": {"type": "string", "enum": ["upright", "flat"]},
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                    "publisher": {"type": "string"},
                    "all_text": {"type": "string"},
                    "legible": {"type": "boolean"},
                    "age_cues": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["box_2d", "orientation", "title", "author", "publisher", "all_text", "legible", "age_cues"],
            },
        }
    },
    "required": ["books"],
}

ITEM_CATEGORIES = [
    "shelving", "furniture", "lighting", "rug", "electronics", "appliance", "art", "decor", "plant", "other",
]

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                    "category": {"type": "string", "enum": ITEM_CATEGORIES},
                    "description": {"type": "string"},
                    "material": {"type": "string"},
                    "brand_model": {"type": "string"},
                    "is_artwork": {"type": "boolean"},
                },
                "required": ["box_2d", "category", "description", "material", "brand_model", "is_artwork"],
            },
        }
    },
    "required": ["items"],
}


class VisionModel(Protocol):
    async def generate_json(self, image_jpeg: bytes, prompt: str, schema: dict) -> tuple[dict, dict]:
        """Returns (parsed JSON, usage info with token counts and latency)."""


@dataclass
class SpineDetection:
    box_px: tuple[float, float, float, float]  # x0, y0, x1, y1 in the image that was sent
    orientation: str
    title: str
    author: str
    publisher: str
    all_text: str
    legible: bool
    age_cues: list[str] = field(default_factory=list)


@dataclass
class ItemDetection:
    box_px: tuple[float, float, float, float]
    category: str
    description: str
    material: str
    brand_model: str
    is_artwork: bool


def box_to_px(box_2d: list[int], width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(box_2d, list) or len(box_2d) != 4:
        return None
    ymin, xmin, ymax, xmax = (max(0, min(1000, int(v))) for v in box_2d)
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin * width / 1000, ymin * height / 1000, xmax * width / 1000, ymax * height / 1000


def parse_spines(payload: dict, width: int, height: int) -> list[SpineDetection]:
    out = []
    for book in payload.get("books", []) or []:
        box = box_to_px(book.get("box_2d"), width, height)
        if box is None:
            continue
        legible = bool(book.get("legible"))
        # If the model says the title is not legible, discard any title it wrote anyway.
        title = (book.get("title") or "").strip() if legible else ""
        out.append(SpineDetection(
            box_px=box,
            orientation=book.get("orientation") or "upright",
            title=title,
            author=(book.get("author") or "").strip() if legible else "",
            publisher=(book.get("publisher") or "").strip(),
            all_text=(book.get("all_text") or "").strip(),
            legible=legible and bool(title),
            age_cues=[c for c in book.get("age_cues") or [] if isinstance(c, str) and c.strip()],
        ))
    return out


def parse_items(payload: dict, width: int, height: int) -> list[ItemDetection]:
    out = []
    for item in payload.get("items", []) or []:
        box = box_to_px(item.get("box_2d"), width, height)
        if box is None or item.get("category") not in ITEM_CATEGORIES:
            continue
        out.append(ItemDetection(
            box_px=box, category=item["category"], description=(item.get("description") or "").strip(),
            material=(item.get("material") or "").strip(), brand_model=(item.get("brand_model") or "").strip(),
            is_artwork=bool(item.get("is_artwork")) or item["category"] == "art",
        ))
    return out


class GeminiVision:
    """google-genai implementation with a simple per-minute rate limit for the free tier."""

    def __init__(self, api_key: str, model: str, max_per_minute: int = 9):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_per_minute = max_per_minute
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def _throttle(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < 60]
                if len(self._calls) < self.max_per_minute:
                    self._calls.append(now)
                    return
                await asyncio.sleep(60 - (now - self._calls[0]) + 0.1)

    async def generate_json(self, image_jpeg: bytes, prompt: str, schema: dict) -> tuple[dict, dict]:
        from google.genai import types

        await self._throttle()
        started = time.monotonic()
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=schema, temperature=0,
            ),
        )
        usage = response.usage_metadata
        info = {
            "model": self.model,
            "latency_s": round(time.monotonic() - started, 2),
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        }
        return json.loads(response.text or "{}"), info
