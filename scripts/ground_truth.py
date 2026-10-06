"""Turn the hand-filled CSV sheets in ground_truth/ into ground_truth.json for scripts/evaluate.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "ground_truth"


def _rows(name: str) -> list[dict]:
    path = ROOT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]


def _num(value: str) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def build() -> dict:
    books = [{
        "unit": r["unit"], "shelf": r["shelf"], "position": r["position"], "title": r["title"].strip(),
        "author": r["author"].strip(), "legible": r["legible"].strip().lower() in ("yes", "y", "true", "1"),
        "height_cm": _num(r["height_cm"]), "thickness_cm": _num(r["thickness_cm"]),
    } for r in _rows("books.csv")]
    prices = [{"title": r["title"], "replacement": _num(r["replacement_inr"]), "used": _num(r["used_inr"]),
               "source": r["source"], "url": r["url"], "checked_on": r["checked_on"]}
              for r in _rows("prices.csv") if _num(r["replacement_inr"])]
    items = [{"category": r["category"], "description": r["description"], "brand_model": r["brand_model"]} for r in _rows("items.csv")]
    room_rows = _rows("room.csv")
    room = {k: _num(v) for k, v in room_rows[0].items() if k != "shape_notes"} if room_rows else {}
    return {"book_count": len(books), "legible": sum(b["legible"] for b in books), "books": books,
            "prices": prices, "items": items, "room": room}


if __name__ == "__main__":
    data = build()
    (ROOT / "ground_truth.json").write_text(json.dumps(data, indent=1))
    print(f"{data['book_count']} books ({data['legible']} legible), {len(data['prices'])} prices, "
          f"{len(data['items'])} items -> {ROOT / 'ground_truth.json'}")
