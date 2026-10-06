"""Optional recording of the spoken conversation, as one WAV per sweep (RECORD_CONVERSATION=1).

Off by default: a policyholder's voice is recorded only when the operator
chooses to. The timeline mirrors what the browser plays: microphone audio at
the moment it arrives; agent audio queued back to back, starting no earlier
than it arrived (the model sends it faster than real time, and the page plays
it sequentially). An interruption cuts the agent's queued audio, as the page
does.
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np

RATE = 24_000  # output rate (the agent's rate); the 16 kHz microphone is resampled up


class ConversationRecorder:
    def __init__(self) -> None:
        self.started_wall = time.time()
        self.started = time.monotonic()
        self.mic: list[tuple[float, np.ndarray]] = []
        self.agent: list[tuple[float, np.ndarray]] = []
        self.agent_end = 0.0  # where the agent's queued audio ends, in seconds from the start

    def _now(self) -> float:
        return time.monotonic() - self.started

    def add_mic(self, pcm16: bytes, rate: int = 16_000) -> None:
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        self.mic.append((self._now(), _resample(samples, rate)))

    def add_agent(self, pcm16: bytes, rate: int = RATE) -> None:
        samples = _resample(np.frombuffer(pcm16, dtype=np.int16).astype(np.float32), rate)
        start = max(self._now(), self.agent_end)
        self.agent.append((start, samples))
        self.agent_end = start + len(samples) / RATE

    def interrupt(self) -> None:
        """The page stops agent playback: drop what had not been played yet."""
        now = self._now()
        kept = []
        for start, samples in self.agent:
            if start < now:
                kept.append((start, samples[: int((now - start) * RATE)]))
        self.agent = kept
        self.agent_end = now

    def save(self, directory: Path) -> Path | None:
        if not self.mic and not self.agent:
            return None
        end = max([s + len(a) / RATE for s, a in self.mic + self.agent])
        track = np.zeros(int(end * RATE) + 1, dtype=np.float32)
        for start, samples in self.mic + self.agent:
            i = int(start * RATE)
            track[i:i + len(samples)] += samples[: len(track) - i]
        path = directory / "conversation.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes(np.clip(track, -32768, 32767).astype(np.int16).tobytes())
        # When the recording starts, so it can be lined up with a screen recording of the same sweep.
        (directory / "conversation.json").write_text(
            json.dumps({"started_unix": self.started_wall, "rate": RATE, "seconds": round(end, 2)}), encoding="utf-8")
        return path


def _resample(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE or not len(samples):
        return samples
    positions = np.arange(0, len(samples), rate / RATE)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
