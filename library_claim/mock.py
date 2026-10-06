"""The offline mock flow: the whole user journey with no API keys and no network.

The browser plays the built-in demo footage; the server runs the real
pipeline on those frames, answers vision from a recorded run
(RecordedVision), catalog/FX/price lookups from recordings (OfflineLookups),
and a scripted agent stands in for Gemini Live. It speaks the same capture
directions the live agent is given, from the same Narrator, so what a tester
sees is the real flow, with only the model calls replayed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from .config import LOCALES, Settings, confirms_country
from .live import Narrator
from .stages.vision import RecordedVision
from .sweep import OfflineLookups, SweepSession

Send = Callable[[dict], Awaitable[None]]
MAX_TEXT = 500
VISION_DELAY_S = 0.6  # recorded answers arrive at a pace like the live model's, so the inventory fills in visibly
CURRENCY_SYMBOL = {"INR": "₹", "USD": "$", "GBP": "£"}


def available(demo: Path) -> bool:
    recorded = demo / "recorded"
    return (demo / "frames").is_dir() and (recorded / "vision").is_dir() and (recorded / "http.json").exists()


class ScriptedAgent:
    """Turns the Narrator's [system] directions into what the live agent would say."""

    LINES = (
        ("motion-blurred", "That was a little blurry. Please slow down, or hold still for a second."),
        ("Glare", "There's glare on the spines. Could you tilt the camera slightly?"),
        ("too dark", "It's too dark to read the spines. Could you turn on a light?"),
        ("No size marker", "I can't see the size marker yet. Hold it flat against the shelf for a moment."),
        ("step closer", "I can't read those spines. Please step a little closer."),
    )

    def __init__(self) -> None:
        self.announced = 0
        self.art_pending: str | None = None

    def direction(self, system_text: str) -> str | None:
        art = re.search(r"Artwork found: (.+?)\. Ask", system_text)
        if art:
            self.art_pending = art.group(1)
            return f"I can see {art.group(1)}. Is it an original or a print?"
        return next((line for key, line in self.LINES if key in system_text), None)

    def progress(self, book_count: int) -> str | None:
        """A short count every 15 books, as the live agent paces its updates."""
        if book_count // 15 > self.announced // 15:
            self.announced = book_count
            return f"{book_count} books logged so far."
        return None


async def run(websocket: WebSocket, send: Send, cfg: Settings, demo: Path) -> None:
    recorded = demo / "recorded"
    frames = sorted((demo / "frames").glob("*.jpg"))
    marker_cm = json.loads((demo / "ground_truth.json").read_text(encoding="utf-8")).get("marker_cm", cfg.marker_size_cm)
    narrator, agent = Narrator(), ScriptedAgent()
    finishing: dict = {}

    async def say(text: str) -> None:
        await send({"type": "transcript", "speaker": "agent", "text": text})
        await send({"type": "turn_complete"})

    async def on_pipeline_event(event: dict) -> None:
        await send(event)
        if event.get("type") == "inventory":
            line = agent.progress(event["book_count"])
            if line:
                await say(line)
        if event.get("type") == "packet":
            await say(_packet_summary(event))
        said = narrator.on_event(event)
        if said and said[1]:
            line = agent.direction(said[0])
            if line:
                await say(line)

    session_cfg = replace(cfg, marker_size_cm=float(marker_cm), google_api_key="", serpapi_key="")
    sweep = SweepSession(session_cfg, RecordedVision(recorded / "vision", VISION_DELAY_S), on_pipeline_event,
                         device="mock flow (demo footage, recorded answers)",
                         offline=OfflineLookups(recorded / "http.json", recorded / "prices"))
    await send({"type": "ready", "sweep_id": sweep.id, "countries": sorted(LOCALES), "country": sweep.country, "mock": True})
    await say("Hello! I'm your claim assistant. I'll guide you while you film your library once. "
              "First, which country are you in, so I price everything in your currency?")

    async def end() -> None:
        if "task" not in finishing:
            finishing["task"] = asyncio.create_task(sweep.finish())
            await send({"type": "phase", "phase": "processing"})
            await say("That's everything. I'm building your claim packet now.")

    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > 10_000:
                continue
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "keyframe" and "task" not in finishing:
                index = message.get("demo_index")
                if not isinstance(index, int) or not 0 <= index < len(frames):
                    continue  # frames come only from the demo footage, by index
                feedback = await sweep.add_frame(frames[index].read_bytes())
                await send({"type": "feedback", **feedback})
                said = narrator.on_frame(feedback)
                if said and said[1]:
                    line = agent.direction(said[0])
                    if line:
                        await say(line)
            elif kind == "text":
                await _answer(str(message.get("text", ""))[:MAX_TEXT], sweep, agent, send, say)
            elif kind == "ar_point":
                result = sweep.add_ar_point(str(message.get("kind")), list(message.get("position", []))[:3])
                await send({"type": "ar_point", **result})
                if result.get("applied") and message.get("kind") == "ceiling":
                    await say("Ceiling height recorded. That gives me the room's floor and wall areas.")
                elif result.get("floor_corners") == 1:
                    await say("Now the room: I'm marking each floor corner as you reach it.")
            elif kind == "end":
                await end()
    except WebSocketDisconnect:
        pass
    finally:
        # Same rule as a live sweep: a dropped connection still finishes the packet from what was captured.
        if "task" not in finishing and sweep.frame_log:
            finishing["task"] = asyncio.create_task(sweep.finish())
        if "task" in finishing:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(finishing["task"]), timeout=600)


async def _answer(text: str, sweep: SweepSession, agent: ScriptedAgent, send: Send, say) -> None:
    await send({"type": "transcript", "speaker": "you", "text": text})
    if agent.art_pending and re.search(r"\b(print|poster|copy|original)\b", text, re.I):
        is_print = not re.search(r"\boriginal\b", text, re.I)
        result = sweep.answer_art_question(is_print)
        await send({"type": "tool", "name": "answer_art_question", "args": {"is_print": is_print}, "result": result})
        agent.art_pending = None
        await say("Noted, a print." if is_print else "Noted. An original goes to an appraiser rather than being auto-priced.")
        return
    code = next((c for c in LOCALES if confirms_country(c, text)), None)
    if code is None:
        await say("Sorry, which country are you in? India, the US or the UK?")
        return
    sweep.country = code
    currency = LOCALES[code].currency
    await send({"type": "locale", "country": code, "currency": currency})
    await say(f"Thanks, prices will be in {currency}. Now walk slowly along each shelving unit from top to bottom, "
              "with the size marker in view at the start of each unit.")


def _packet_summary(event: dict) -> str:
    totals = event.get("totals", {})
    symbol = CURRENCY_SYMBOL.get(totals.get("currency", ""), totals.get("currency", "") + " ")
    return (f"Your claim packet is ready: {totals.get('book_count', 0)} books, {totals.get('books_identified', 0)} identified, "
            f"replacement value {symbol}{totals.get('books_replacement_cost', 0):,.0f}. "
            f"{event.get('review_count', 0)} lines need a human review. Open the claim packet to see every source.")
