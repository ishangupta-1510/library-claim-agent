"""Score a claim packet against hand-collected ground truth, one line per pass bar.

Ground truth JSON (dev_data/synthetic/ground_truth.json has the same shape;
for the real capture, ground_truth/ground_truth.json is filled in by hand):

{
  "book_count": 63,
  "books": [ {"title": "...", "author": "...", "legible": true,
              "height_cm": 23.4, "thickness_cm": 2.1}, ... ],      # legible = a human can read it
  "prices": [ {"title": "...", "replacement": 399, "used": 180, "source": "Amazon.in"} ],   # optional
  "items":  [ {"category": "lighting", "description": "floor lamp"} ],                       # optional
  "room":   {"floor_area_m2": 12.1, "wall_area_m2": 38.0}                                     # optional
}

Usage: python scripts/evaluate.py <sweep_dir> <ground_truth.json>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from rapidfuzz import fuzz

TITLE_MATCH = 85


def _same_title(a: str, b: str) -> bool:
    return bool(a and b) and fuzz.token_sort_ratio(a.lower(), b.lower()) >= TITLE_MATCH


def evaluate(packet: dict, truth: dict) -> dict:
    books = packet["books"]
    true_books = truth["books"]
    results: dict = {}

    # Book count: within 5% of the true count.
    true_count = truth.get("book_count", len(true_books))
    got = len(books)
    error = abs(got - true_count) / true_count
    results["book_count"] = {"truth": true_count, "system": got, "error_pct": round(100 * error, 1), "pass": error <= 0.05}

    # Titles: >= 70% of human-legible spines correct, <= 3% confidently wrong.
    legible_truth = [b for b in true_books if b.get("legible") and b.get("title")]
    identified = [b for b in books if b["status"] in ("identified", "needs_appraisal") and b.get("title")]
    unmatched_truth = list(legible_truth)
    correct, wrong = 0, []
    for book in identified:
        match = next((t for t in unmatched_truth if _same_title(book["title"], t["title"])), None)
        if match:
            correct += 1
            unmatched_truth.remove(match)
        else:
            wrong.append(book["title"])
    confident_wrong = [b for b in identified if b["title"] in wrong and b.get("id_confidence", 0) >= 0.75]
    rate = correct / len(legible_truth) if legible_truth else 0
    wrong_rate = len(confident_wrong) / max(1, len(identified))
    results["titles"] = {
        "legible_truth": len(legible_truth), "correct": correct, "correct_pct": round(100 * rate, 1),
        "confidently_wrong": len(confident_wrong), "confidently_wrong_pct": round(100 * wrong_rate, 1),
        "wrong_titles": wrong, "pass": rate >= 0.70 and wrong_rate <= 0.03,
    }

    # Spine dimensions: within 15% on the hand-measured sample (matched by title).
    errors = []
    for t in true_books:
        if not t.get("height_cm") or not t.get("title"):
            continue
        match = next((b for b in books if _same_title(b.get("title", ""), t["title"]) and b.get("spine_height_cm")), None)
        if match:
            errors.append({
                "title": t["title"],
                "height_err_pct": round(100 * abs(match["spine_height_cm"] - t["height_cm"]) / t["height_cm"], 1),
                "thickness_err_pct": round(100 * abs(match["spine_thickness_cm"] - t["thickness_cm"]) / t["thickness_cm"], 1),
            })
    sample = errors[:20]
    within = [e for e in sample if e["height_err_pct"] <= 15 and e["thickness_err_pct"] <= 15]
    results["spine_dimensions"] = {
        "sample": len(sample), "within_15pct": len(within),
        "median_height_err_pct": statistics.median([e["height_err_pct"] for e in sample]) if sample else None,
        "median_thickness_err_pct": statistics.median([e["thickness_err_pct"] for e in sample]) if sample else None,
        "worst": sorted(sample, key=lambda e: -max(e["height_err_pct"], e["thickness_err_pct"]))[:5],
        "pass": bool(sample) and len(within) == len(sample),
    }

    # Prices: within 25% of the hand-checked price (same source type).
    price_rows = []
    for p in truth.get("prices", []):
        match = next((b for b in books if _same_title(b.get("title", ""), p["title"])), None)
        amount = match and match["replacement_cost"]["amount"]
        if amount:
            price_rows.append({"title": p["title"], "truth": p["replacement"], "system": amount,
                               "err_pct": round(100 * abs(amount - p["replacement"]) / p["replacement"], 1)})
    if truth.get("prices"):
        ok = [r for r in price_rows if r["err_pct"] <= 25]
        results["book_prices"] = {"sample": len(truth["prices"]), "priced": len(price_rows), "within_25pct": len(ok),
                                  "rows": price_rows, "pass": len(ok) == len(truth["prices"])}

    # Non-book items: >= 80% found with the right category.
    if truth.get("items"):
        remaining = list(packet["items"])
        found = 0
        for item in truth["items"]:
            match = next((i for i in remaining if i["category"] == item["category"]
                          and fuzz.token_set_ratio(i["description"], item["description"]) >= 60), None)
            if match:
                found += 1
                remaining.remove(match)
        share = found / len(truth["items"])
        results["items"] = {"truth": len(truth["items"]), "found": found, "found_pct": round(100 * share, 1), "pass": share >= 0.8}

    # Room areas.
    room, true_room = packet["room"], truth.get("room") or {}
    for key, bar in (("floor_area_m2", 0.10), ("wall_area_m2", 0.15)):
        if true_room.get(key):
            value = room.get(key)
            err = abs(value - true_room[key]) / true_room[key] if value else None
            results[key] = {"truth": true_room[key], "system": value,
                            "error_pct": round(100 * err, 1) if err is not None else None,
                            "pass": err is not None and err <= bar}

    # Time to packet.
    seconds = packet.get("stages", {}).get("time_to_packet_s")
    results["time_to_packet_s"] = {"system": seconds, "pass": seconds is not None and seconds < 300}
    return results


def main() -> None:
    sweep_dir, truth_path = Path(sys.argv[1]), Path(sys.argv[2])
    packet = json.loads((sweep_dir / "claim_packet.json").read_text(encoding="utf-8"))
    results = evaluate(packet, json.loads(truth_path.read_text(encoding="utf-8")))
    (sweep_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, r in results.items():
        flag = "PASS" if r.get("pass") else "FAIL"
        detail = {k: v for k, v in r.items() if k not in ("pass", "rows", "worst", "wrong_titles")}
        print(f"{flag:4}  {name:18} {detail}")


if __name__ == "__main__":
    main()
