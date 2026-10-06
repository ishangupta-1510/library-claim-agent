"""Record an unattended demo video of a sweep on the built-in synthetic library.

    python -m scripts.record_demo --mode live   # real Gemini agent; a synthesized voice plays the policyholder
    python -m scripts.record_demo --mode mock   # the offline mock flow (no keys)

What it does (Windows; needs Chrome and ffmpeg on PATH):
1. Builds a camera file from dev_data/synthetic/frames (each view held 2.5 s) and, for live mode,
   a microphone file in which a synthesized voice says the policyholder's lines at set times.
2. Starts the app on its own port with RECORD_CONVERSATION=1.
3. Runs a separate Chrome window (fresh profile) whose camera and microphone are those files,
   records the page itself through DevTools' screencast, presses Start, ends the sweep when the
   footage is done, waits for the packet and scrolls through the report.
4. Muxes the recorded conversation (agent voice + policyholder voice) under the page recording.

The output is labelled in its file name: it shows the synthetic library, not a real room.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import websockets

FRAMES = Path("dev_data/synthetic/frames")
WORK = Path(".cache/demo")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PORT, DEBUG_PORT = 8020, 9333
HOLD_S = 2.5
# The policyholder's lines and when they are spoken (seconds after the microphone opens).
LINES = [
    (9.0, "I'm in India."),
    (32.0, "That blue book on the top shelf is a signed copy."),
]


def run(cmd: list[str], **kw) -> None:
    subprocess.run(cmd, check=True, **kw)


def build_camera() -> Path:
    """Y4M (Chrome's fake camera keeps its frame rate): 2 fps, each synthetic view held HOLD_S seconds."""
    out = WORK / "camera.y4m"
    if not out.exists():
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(1 / HOLD_S),
             "-i", str(FRAMES / "%04d.jpg"), "-vf", "fps=2,format=yuv420p", str(out)])
    return out


def build_microphone() -> Path:
    """A 16 kHz WAV: silence, with each line synthesized by Windows speech at its time."""
    out = WORK / "microphone.wav"
    rate, total_s = 16_000, 240
    track = np.zeros(rate * total_s, dtype=np.int16)
    for i, (at, text) in enumerate(LINES):
        line = WORK / f"line{i}.wav"
        script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.SetOutputToWaveFile('{line.resolve()}'); $s.Speak(\"{text}\"); $s.Dispose()")
        run(["powershell", "-NoProfile", "-Command", script])
        tmp16 = WORK / f"line{i}-16k.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(line), "-ar", str(rate), "-ac", "1",
             "-sample_fmt", "s16", str(tmp16)])
        with wave.open(str(tmp16)) as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        start = int(at * rate)
        track[start:start + len(samples)] = samples[: len(track) - start]
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(track.tobytes())
    return out


def wait_for(url: str, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"{url} did not come up")


class Page:
    """Just enough of the Chrome DevTools protocol: run JavaScript, and record the page as it renders.

    Recording uses DevTools' screencast: Chrome sends each rendered frame of the page itself, so the
    video holds exactly the app (no other windows or notifications), even if the window is covered.
    Screen grabbing was tried first and recorded only black: Chrome's window is GPU-composited.
    """

    def __init__(self, ws, frames_dir: Path):
        self.ws, self.next_id = ws, 0
        self.pending: dict[int, asyncio.Future] = {}
        self.frames_dir = frames_dir
        self.frames: list[tuple[float, Path]] = []  # (wall-clock time, jpeg)
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for raw in self.ws:
            message = json.loads(raw)
            if message.get("id") in self.pending:
                self.pending.pop(message["id"]).set_result(message)
            elif message.get("method") == "Page.screencastFrame":
                params = message["params"]
                path = self.frames_dir / f"{len(self.frames):06d}.jpg"
                path.write_bytes(base64.b64decode(params["data"]))
                self.frames.append((params["metadata"]["timestamp"], path))
                await self._send("Page.screencastFrameAck", {"sessionId": params["sessionId"]}, wait=False)

    async def _send(self, method: str, params: dict, wait: bool = True):
        self.next_id += 1
        future = asyncio.get_running_loop().create_future()
        if wait:
            self.pending[self.next_id] = future
        await self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        return await future if wait else None

    async def js(self, expression: str):
        reply = await self._send("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
        return reply.get("result", {}).get("result", {}).get("value")

    async def until(self, expression: str, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(expression):
                return True
            await asyncio.sleep(1)
        return False


async def drive(mode: str, frames_dir: Path) -> tuple[str, list[tuple[float, Path]]]:
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json").read())
    page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
    async with websockets.connect(page_ws, max_size=None) as ws:
        page = Page(ws, frames_dir)
        await page.until("document.readyState === 'complete' && !!document.getElementById('start')", 30)
        await page._send("Page.startScreencast", {"format": "jpeg", "quality": 85})
        await asyncio.sleep(3)
        if mode == "mock":
            await page.until("document.getElementById('start-mock').style.display !== 'none'", 10)
            await page.js("document.getElementById('start-mock').click()")
        else:
            await page.js("document.getElementById('marker-cm').value = '10'")  # the marker printed in the footage
            await page.js("document.getElementById('start').click()")
            await asyncio.sleep(len(list(FRAMES.glob("*.jpg"))) * HOLD_S + 6)
            await page.js("document.getElementById('end').click()")
        ready = await page.until("document.getElementById('packet').style.display === 'inline'", 420)
        await asyncio.sleep(6)  # the agent's summary
        report = await page.js("document.getElementById('packet').href") if ready else ""
        if report:
            await page.js(f"location.href = {json.dumps(report)}")
            await asyncio.sleep(4)
            for _ in range(12):
                await page.js("window.scrollBy(0, 300)")
                await asyncio.sleep(1.2)
        await page._send("Page.stopScreencast", {})
        await asyncio.sleep(1)
        page.reader.cancel()
        return report, page.frames


def encode(frames: list[tuple[float, Path]], out: Path, fps: int = 15) -> None:
    """Frames arrive only when the page changes; hold each one until the next, at a constant output rate."""
    listing = WORK / "frames.txt"
    lines = []
    following = [t for t, _ in frames[1:]] + [frames[-1][0] + 2.0]
    for (t, path), t_next in zip(frames, following):
        lines += [f"file '{path.resolve().as_posix()}'", f"duration {max(0.001, t_next - t):.3f}"]
    lines.append(f"file '{frames[-1][1].resolve().as_posix()}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-vf", f"fps={fps},crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
         str(out)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "mock"], default="live")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not shutil.which("ffmpeg") or not CHROME.exists():
        raise SystemExit("needs ffmpeg on PATH and Chrome at " + str(CHROME))
    WORK.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    out = Path(args.out or Path.home() / "Desktop" / f"demo-{args.mode}-synthetic-library-{stamp}.mp4")
    camera = build_camera()
    microphone = build_microphone() if args.mode == "live" else None
    frames_dir = WORK / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir()

    sweeps = Path("sweeps")
    before = set(sweeps.glob("*")) if sweeps.exists() else set()
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "library_claim.server:app", "--host", "127.0.0.1",
                               "--port", str(PORT), "--log-level", "warning"], env={**os.environ, "RECORD_CONVERSATION": "1"})
    profile = Path(tempfile.mkdtemp(prefix="claim-demo-chrome-"))
    chrome = None
    try:
        wait_for(f"http://127.0.0.1:{PORT}/api/health")
        flags = [f"--user-data-dir={profile}", f"--remote-debugging-port={DEBUG_PORT}", "--no-first-run",
                 # A visible window: headless Chrome does not repaint <video> into the screencast, so the
                 # camera panel looked frozen on its first frame.
                 "--no-default-browser-check", "--window-position=40,40", "--window-size=1600,900",
                 "--hide-scrollbars", "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                 # Keep painting when other windows cover it (Windows occlusion tracking would pause the page).
                 "--disable-features=CalculateNativeWinOcclusion",
                 "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                 f"--use-file-for-fake-video-capture={camera.resolve()}", "--autoplay-policy=no-user-gesture-required",
                 f"--app=http://localhost:{PORT}/"]
        if microphone:
            flags.append(f"--use-file-for-fake-audio-capture={microphone.resolve()}%noloop")
        chrome = subprocess.Popen([str(CHROME), *flags], stderr=subprocess.DEVNULL)
        wait_for(f"http://127.0.0.1:{DEBUG_PORT}/json")
        time.sleep(2)
        report, frames = asyncio.run(drive(args.mode, frames_dir))
    finally:
        if chrome:
            chrome.terminate()
            chrome.wait(timeout=30)
        time.sleep(4)  # the server saves the conversation when the page disconnects
        server.terminate()
        server.wait(timeout=30)
        shutil.rmtree(profile, ignore_errors=True)

    if not frames:
        raise SystemExit("no frames were recorded")
    screen = WORK / "screen.mp4"
    encode(frames, screen)
    new = sorted(set(sweeps.glob("*")) - before)
    sweep = new[-1] if new else None
    conversation = sweep / "conversation.wav" if sweep else None
    if conversation and conversation.exists():
        started = json.loads((sweep / "conversation.json").read_text())["started_unix"]
        offset = max(0.0, started - frames[0][0])
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(screen), "-itsoffset", f"{offset:.3f}",
             "-i", str(conversation), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(out)])
    else:
        shutil.copy(screen, out)
    print(f"video: {out} ({len(frames)} frames)\nsweep: {sweep}\nreport: {report or 'not reached'}")


if __name__ == "__main__":
    main()
