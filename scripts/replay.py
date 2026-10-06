"""Run the measuring pipeline on a folder of frames (no voice), then optionally score it.

Useful for tuning and for re-running a recorded sweep with a changed stage.

    python scripts/replay.py dev_data/synthetic/frames --truth dev_data/synthetic/ground_truth.json
    python scripts/replay.py <frames_dir> --marker-cm 18.0 --ar-points ar_points.json

Uses the real Gemini vision model when GOOGLE_API_KEY is set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

from library_claim.config import settings
from library_claim.stages.vision import GeminiVision
from library_claim.sweep import SweepSession


async def run(args) -> Path:
    cfg = settings()
    if args.marker_cm:
        cfg = replace(cfg, marker_size_cm=args.marker_cm)
    if not cfg.google_api_key:
        raise SystemExit("GOOGLE_API_KEY is not set in .env")

    async def emit(event: dict) -> None:
        if event["type"] == "inventory":
            print(f"  [{time.strftime('%H:%M:%S')}] {event['frame_id']}: {event['detected']} spines "
                  f"({event['legible']} legible), {event['new_books']} new, total {event['book_count']}")
        elif event["type"] in ("stage_error", "packet", "items"):
            print("  ", {k: v for k, v in event.items() if k not in ("books", "added")})

    sweep = SweepSession(cfg, GeminiVision(cfg.google_api_key, cfg.vision_model), emit, device="replay")
    frames = sorted(Path(args.frames).glob("*.jpg"))
    print(f"Replaying {len(frames)} frames into sweep {sweep.id}")
    for path in frames:
        feedback = await sweep.add_frame(path.read_bytes())
        if feedback["problems"]:
            print(f"  {feedback['frame_id']} ({path.name}): {feedback['problems']}")
    if args.ar_points:
        for point in json.loads(Path(args.ar_points).read_text()):
            sweep.add_ar_point(point["kind"], point["position"])
    packet = await sweep.finish()
    print(f"Packet: {packet.totals.book_count} books, {packet.totals.books_identified} identified -> {sweep.dir}")
    return sweep.dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames")
    parser.add_argument("--truth")
    parser.add_argument("--marker-cm", type=float)
    parser.add_argument("--ar-points")
    args = parser.parse_args()
    sweep_dir = asyncio.run(run(args))
    if args.truth:
        from scripts.evaluate import evaluate

        packet = json.loads((sweep_dir / "claim_packet.json").read_text())
        results = evaluate(packet, json.loads(Path(args.truth).read_text()))
        (sweep_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
        for name, r in results.items():
            print(f"{'PASS' if r.get('pass') else 'FAIL':4}  {name:18} "
                  f"{ {k: v for k, v in r.items() if k not in ('pass', 'rows', 'worst', 'wrong_titles')} }")


if __name__ == "__main__":
    main()
