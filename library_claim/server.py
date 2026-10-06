"""FastAPI app: the sweep UI, the marker page, saved sweeps, and the live WebSocket.

Run:  uvicorn library_claim.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
from dataclasses import replace
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import LOCALES, confirms_country, settings
from . import mock
from .conversation import ConversationRecorder
from .live import Narrator, live_config
from .stages.scale import render_marker
from .stages.vision import GeminiVision
from .sweep import SweepSession

logger = logging.getLogger("library_claim")
logging.basicConfig(level=logging.INFO)
# httpx logs every request URL at INFO, and SerpAPI takes its key as a query parameter: keep keys out of logs.
for noisy in ("httpx", "httpcore", "google_genai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

STATIC = Path(__file__).parent / "static"
# Demo camera: the synthetic sweep, played in the browser as the camera so the live flow can be tried
# without a bookshelf. Present only in a development checkout (dev_data/synthetic.py generates it).
DEMO = Path(__file__).parent.parent / "dev_data" / "synthetic"
MAX_MESSAGE = 4_000_000  # a high-res keyframe as base64

app = FastAPI(title="Library contents claim agent")


@app.middleware("http")
async def revalidate_app_shell(request, call_next):
    """The page and its scripts are revalidated on every load (ETag, so usually a 304).

    Without this a browser kept running a cached app.js after an update, with
    a page that no longer matched the server.
    """
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response
cfg = settings()
cfg.sweeps_dir.mkdir(parents=True, exist_ok=True)
app.mount("/sweeps", StaticFiles(directory=cfg.sweeps_dir, html=True), name="sweeps")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
if (DEMO / "frames").is_dir():
    app.mount("/demo/frames", StaticFiles(directory=DEMO / "frames"), name="demo-frames")


def _demo_manifest() -> dict | None:
    frames = sorted(p.name for p in (DEMO / "frames").glob("*.jpg")) if (DEMO / "frames").is_dir() else []
    truth = DEMO / "ground_truth.json"
    if not frames or not truth.exists():
        return None
    marker_cm = json.loads(truth.read_text(encoding="utf-8")).get("marker_cm")
    return {"frames": [f"/demo/frames/{name}" for name in frames], "marker_cm": marker_cm}


@app.get("/api/demo")
def demo() -> dict:
    """The demo camera's frames and the size of the marker printed in them (unavailable outside dev)."""
    return {**(_demo_manifest() or {"frames": []}), "mock": mock.available(DEMO)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/marker")
def marker_page() -> FileResponse:
    return FileResponse(STATIC / "marker.html")


@app.get("/marker.png")
def marker_png(id: int = 0) -> Response:
    ok, png = cv2.imencode(".png", render_marker(id, 800))
    return Response(png.tobytes(), media_type="image/png")


@app.get("/api/health")
def health() -> dict:
    return {"gemini_key": bool(cfg.google_api_key), "serpapi_key": bool(cfg.serpapi_key),
            "live_model": cfg.live_model, "vision_model": cfg.vision_model, "country": cfg.country,
            "marker_size_cm": cfg.marker_size_cm}


def _marker_cm(value: str | None) -> float | None:
    """A marker side length from the page, in cm, if it is a plausible one."""
    try:
        size = float(value) if value else None
    except ValueError:
        return None
    return size if size is not None and 1.0 <= size <= 60.0 else None


@app.websocket("/ws/sweep")
async def sweep_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            with contextlib.suppress(Exception):
                await websocket.send_json(payload)

    if websocket.query_params.get("mock") == "1":
        # Offline mock flow: demo footage, recorded model answers, scripted agent; no keys, no network.
        if not mock.available(DEMO):
            await send({"type": "error", "message": "Mock flow needs dev_data/synthetic (frames and recorded/)"})
            await websocket.close()
            return
        await mock.run(websocket, send, cfg, DEMO)
        return

    if not cfg.google_api_key:
        await send({"type": "error", "message": "GOOGLE_API_KEY is missing from .env"})
        await websocket.close()
        return

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.google_api_key)
    narrator = Narrator()
    live_ref: dict = {}
    finishing: dict = {}

    async def tell_agent(text: str, speak: bool) -> None:
        live = live_ref.get("session")
        if live is None:
            return
        await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=speak)

    async def on_pipeline_event(event: dict) -> None:
        await send(event)
        said = narrator.on_event(event)
        if said:
            await tell_agent(*said)

    vision = GeminiVision(cfg.google_api_key, cfg.vision_model)
    device = websocket.headers.get("user-agent", "")[:120]
    session_cfg = cfg
    marker_cm = _marker_cm(websocket.query_params.get("marker_cm"))
    if marker_cm:
        # The marker shown in this sweep (a phone screen, a print): set on the page, no restart needed.
        session_cfg = replace(cfg, marker_size_cm=marker_cm)
    manifest = _demo_manifest() if websocket.query_params.get("demo") == "1" else None
    if manifest:
        # The demo footage carries its own marker; its size is part of the footage, not this room's setting.
        session_cfg = replace(cfg, marker_size_cm=float(manifest["marker_cm"]))
        device = "demo camera (synthetic sweep) / " + device
    sweep = SweepSession(session_cfg, vision, on_pipeline_event, device=device)
    recorder = ConversationRecorder() if cfg.record_conversation else None
    await send({"type": "ready", "sweep_id": sweep.id, "countries": sorted(LOCALES), "country": sweep.country})

    async def end_sweep() -> dict:
        if "task" not in finishing:
            finishing["task"] = asyncio.create_task(sweep.finish_or_report())
            await send({"type": "phase", "phase": "processing"})
        return {"started": True, "message": "Building the claim packet now; a summary will follow."}

    async def run_tool(call) -> dict:
        args = dict(call.args or {})
        if call.name == "set_locale":
            code = str(args.get("country_code", "")).upper()
            if code not in LOCALES:
                return {"ok": False, "supported": sorted(LOCALES)}
            if not confirms_country(code, str(args.get("policyholder_words", ""))):
                # The live model once set USD after hearing "Sh": the currency must come from what they said.
                return {"ok": False, "reason": "their words do not name this country or currency; ask them to say it"}
            sweep.country = code
            await send({"type": "locale", "country": code, "currency": LOCALES[code].currency})
            return {"ok": True, "country": code, "currency": LOCALES[code].currency}
        if call.name == "note_book_in_view":
            result = sweep.note_book_in_view(str(args.get("statement", "")))
        elif call.name == "exclude_shelf_in_view":
            result = sweep.exclude_shelf_in_view(str(args.get("reason", "")))
        elif call.name == "answer_art_question":
            result = sweep.answer_art_question(bool(args.get("is_print")))
        elif call.name == "end_sweep":
            result = await end_sweep()
        else:
            result = {"error": f"unknown tool {call.name}"}
        await send({"type": "tool", "name": call.name, "args": args, "result": result})
        return result

    opening = "[system] The policyholder opened the app. Greet them and start the opening."
    if manifest:
        opening += (" This is a demo: the camera is pre-recorded footage of a two-unit bookshelf with its marker, "
                    "played at a steady pace. The policyholder cannot move the camera, so do not give capture "
                    "directions; confirm their country, then describe progress from the [system] updates.")
    try:
        async with client.aio.live.connect(model=cfg.live_model, config=live_config()) as live:
            live_ref["session"] = live
            await live.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=opening)]),
                turn_complete=True,
            )

            async def on_message(message: dict) -> None:
                kind = message.get("type")
                if kind == "audio":
                    pcm = base64.b64decode(message["data"])
                    if recorder:
                        recorder.add_mic(pcm)
                    await live.send_realtime_input(audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000"))
                elif kind == "video":
                    await live.send_realtime_input(video=types.Blob(data=base64.b64decode(message["data"]), mime_type="image/jpeg"))
                elif kind == "keyframe":
                    if "task" in finishing:
                        return
                    feedback = await sweep.add_frame(base64.b64decode(message["data"]))
                    await send({"type": "feedback", **feedback})
                    said = narrator.on_frame(feedback)
                    if said:
                        await tell_agent(*said)
                elif kind == "ar_point":
                    result = sweep.add_ar_point(str(message.get("kind")), list(message.get("position", []))[:3])
                    await send({"type": "ar_point", **result})
                    if result.get("applied"):
                        await tell_agent(f"[system] Room point recorded ({message.get('kind')}). Floor corners so far: {result['floor_corners']}.", False)
                elif kind == "text":
                    await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=str(message.get("text", ""))[:2000])]), turn_complete=True)
                elif kind == "end":
                    await end_sweep()
                    await tell_agent("[system] The policyholder pressed End sweep. Tell them the packet is being built.", True)

            async def browser_to_gemini() -> None:
                while True:
                    raw = await websocket.receive_text()
                    if len(raw) > MAX_MESSAGE:
                        # Always answer a keyframe: the page sends the next one only after feedback arrives.
                        await send({"type": "feedback", "problems": ["frame too large to process"]})
                        continue
                    try:
                        await on_message(json.loads(raw))
                    except (ValueError, KeyError, TypeError) as exc:  # one malformed message must not end the sweep
                        logger.warning("Ignored a malformed %s message: %s", type(exc).__name__, exc)
                        await send({"type": "feedback", "problems": ["unreadable message"]})

            async def gemini_to_browser() -> None:
                while True:
                    async for response in live.receive():
                        if response.tool_call and response.tool_call.function_calls:
                            replies = [types.FunctionResponse(id=c.id, name=c.name, response=await run_tool(c))
                                       for c in response.tool_call.function_calls]
                            await live.send_tool_response(function_responses=replies)
                        content = response.server_content
                        if not content:
                            continue
                        if content.interrupted:
                            await send({"type": "interrupted"})
                            if recorder:
                                recorder.interrupt()
                        for speaker, chunk in (("you", content.input_transcription), ("agent", content.output_transcription)):
                            if chunk and chunk.text:
                                await send({"type": "transcript", "speaker": speaker, "text": chunk.text})
                        if content.model_turn and not content.interrupted:
                            for part in content.model_turn.parts or []:
                                if part.inline_data and (part.inline_data.mime_type or "").startswith("audio/"):
                                    if recorder:
                                        rate = re.search(r"rate=(\d+)", part.inline_data.mime_type or "")
                                        recorder.add_agent(part.inline_data.data, int(rate.group(1)) if rate else 24_000)
                                    await send({"type": "audio", "data": base64.b64encode(part.inline_data.data).decode(),
                                                "mime": part.inline_data.mime_type})
                        if content.turn_complete:
                            await send({"type": "turn_complete"})

            tasks = [asyncio.create_task(browser_to_gemini()), asyncio.create_task(gemini_to_browser())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Live session failed")
        await send({"type": "error", "message": f"Live session ended: {type(exc).__name__}: {exc}"[:300]})
    finally:
        # A dropped connection must not lose the sweep: finish the packet from what was captured.
        if "task" not in finishing and sweep.frame_log:
            finishing["task"] = asyncio.create_task(sweep.finish_or_report())
        if "task" in finishing:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(finishing["task"]), timeout=600)
        else:
            sweep.close()
        if recorder:
            with contextlib.suppress(Exception):
                recorder.save(sweep.dir)
