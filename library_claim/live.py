"""The live sweep agent: Gemini Live voice + camera, wired to the sweep pipeline.

Kept from the reference: one WebSocket per session relaying microphone PCM and
low-res camera frames into a Gemini Live session, transcripts and audio back,
and function calls executed server-side.

Changed: the agent does not describe what it sees. Measured facts from the
pipeline (blur scores, marker found, spines logged, art detected) are pushed
into the conversation as system updates, and the agent turns them into short
spoken directions. Inventory comes only from the pipeline, never from the
model's narration.
"""

from __future__ import annotations

import time
from datetime import datetime

from google.genai import types

SYSTEM_INSTRUCTION = """You are the live intake agent for a home-contents insurance claim. The policyholder is
walking their home library once with the phone camera while you talk. A separate measuring system logs the books
and objects; you direct the capture and handle the conversation.

Opening (keep it to two short sentences after the greeting):
1. Greet them, then confirm their country and currency. Call set_locale when they confirm.
2. Explain the sweep: walk slowly along each shelving unit from top to bottom. For measurements, a printed
   or on-screen marker (the black-and-white square from the marker page, shown on a laptop or phone screen or on
   paper) should be in view at the start of each unit. The device itself is not the marker. Then pan across the
   walls and floor. Ask them to tap "Mark corner" at each floor corner and
   "Mark ceiling" once at the top of a wall.

During the sweep:
- You receive [system] updates with measured facts. Turn them into short directions, at most one sentence, e.g.
  "That shelf was blurry, could you go back and slow down?" or "Step closer, I can't read those spines."
- Never state a book title, count, size or price yourself unless a [system] update gave it to you.
- When a [system] update says artwork was found, ask briefly whether it is an original or a print and call
  answer_art_question with their answer.
- If they say something about the book or shelf in view ("that's a first edition", "this one is signed"),
  call note_book_in_view with their exact words, then confirm what was logged.
- If they say a shelf is not theirs, call exclude_shelf_in_view.
- Do not ask them to pull books out, scan barcodes, type ISBNs or photograph items one by one.
- Keep everything short. They are walking and holding a phone.

When they say they are done, call end_sweep. Then tell them the claim packet is being built, and when a [system]
update brings the summary, read back: books counted, how many were identified, the totals with currency, and how
many lines need human review. Read only figures from the update.
"""

TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="set_locale",
        description="Record the policyholder's country once they confirm it. Prices use this country's market and currency.",
        parameters={"type": "object", "properties": {"country_code": {"type": "string", "description": "ISO 3166-1 alpha-2, e.g. IN, US, GB"}}, "required": ["country_code"]},
    ),
    types.FunctionDeclaration(
        name="note_book_in_view",
        description="Attach the policyholder's statement (e.g. first edition, signed, damaged) to the book currently centred on screen.",
        parameters={"type": "object", "properties": {"statement": {"type": "string"}}, "required": ["statement"]},
    ),
    types.FunctionDeclaration(
        name="exclude_shelf_in_view",
        description="Exclude the shelf currently on screen from the claim because the policyholder says the books are not theirs.",
        parameters={"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    ),
    types.FunctionDeclaration(
        name="answer_art_question",
        description="Record whether the most recently detected artwork is a print (true) or an original (false).",
        parameters={"type": "object", "properties": {"is_print": {"type": "boolean"}}, "required": ["is_print"]},
    ),
    types.FunctionDeclaration(
        name="end_sweep",
        description="The policyholder has finished walking the room. Starts building the claim packet.",
        parameters={"type": "object", "properties": {}},
    ),
])]


def live_config(voice: str = "Kore") -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_INSTRUCTION + "\nCurrent time: " + datetime.now().astimezone().isoformat(timespec="minutes"),
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS),
        tools=TOOLS,
    )


class Narrator:
    """Turns pipeline events into [system] updates without flooding the conversation.

    Actionable problems (blur, glare, no marker on a new unit, artwork) are
    spoken; routine progress is added as silent context. The same problem is
    not raised again within REPEAT_S seconds.
    """

    REPEAT_S = 8.0
    MIN_GAP_S = 5.0
    MARKER_ASKS = 2
    MARKER_GAP_S = 60.0

    def __init__(self) -> None:
        self.last_said: dict[str, float] = {}
        self.last_spoken_at = 0.0
        self.blur_streak = 0
        self.marker_asks = 0

    def _may_say(self, key: str) -> bool:
        now = time.monotonic()
        if now - self.last_said.get(key, -1e9) < self.REPEAT_S or now - self.last_spoken_at < self.MIN_GAP_S:
            return False
        self.last_said[key] = now
        self.last_spoken_at = now
        return True

    def on_frame(self, feedback: dict) -> tuple[str, bool] | None:
        """Returns (text, speak_now) or None."""
        problems = feedback.get("problems", [])
        self.blur_streak = self.blur_streak + 1 if "blur" in problems else 0
        if self.blur_streak >= 2 and self._may_say("blur"):
            return "[system] The last frames are motion-blurred. Ask them to slow down or hold still for a second.", True
        if "glare" in problems and self._may_say("glare"):
            return "[system] Glare is washing out part of the frame. Ask them to tilt the phone slightly.", True
        if "dark" in problems and self._may_say("dark"):
            return "[system] The frame is too dark to read spines. Ask them to turn on a light.", True
        if feedback.get("plane") and not feedback.get("metric") and self._may_ask_for_marker():
            return ("[system] No size marker seen yet, so books are counted but not measured in cm. Ask them once to "
                    "hold the black-and-white square marker (on a screen or paper) flat against this shelf for a "
                    "moment; if they don't have it, carry on with the sweep."), True
        return None

    def _may_ask_for_marker(self) -> bool:
        """Ask for the size marker at most MARKER_ASKS times, MARKER_GAP_S apart; the sweep never waits on it.

        Asking per new unit every few seconds looped: each frame that did not
        chain to the last one counted as a new unit, so the agent asked again
        and again and the policyholder could not move on.
        """
        if self.marker_asks >= self.MARKER_ASKS:
            return False
        if time.monotonic() - self.last_said.get("marker", -1e9) < self.MARKER_GAP_S or not self._may_say("marker"):
            return False
        self.marker_asks += 1
        return True

    def on_event(self, event: dict) -> tuple[str, bool] | None:
        kind = event.get("type")
        if kind == "inventory":
            unread = event["detected"] - event["legible"]
            text = (f"[system] Logged {event['new_books']} new book(s) from that view; {event['book_count']} books so far. "
                    f"{unread} spine(s) in that view were not readable.")
            if unread >= 3 and unread >= event["detected"] / 2 and self._may_say(f"unreadable-{event['plane']}"):
                return text + " Ask them to step closer to this shelf so the spines are readable.", True
            return text, False
        if kind == "items":
            art = event.get("needs_question") or []
            listed = ", ".join(i["description"] for i in event["added"])
            if art:
                return f"[system] Logged: {listed}. Artwork found: {art[0]['description']}. Ask whether it is an original or a print.", True
            return f"[system] Logged other contents: {listed}.", False
        if kind == "packet":
            t = event["totals"]
            return ("[system] Claim packet ready. Read back this summary: "
                    f"{t['book_count']} books counted, {t['books_identified']} identified, {t['books_unidentified']} unidentified, "
                    f"{t['books_needs_appraisal']} sent for appraisal. Books replacement total {t['books_replacement_cost']:.0f} {t['currency']}, "
                    f"used value {t['books_used_value']:.0f} {t['currency']}. Other contents {t['items_replacement_cost_low']:.0f} to "
                    f"{t['items_replacement_cost_high']:.0f} {t['currency']}. {event['review_count']} lines need human review."), True
        return None
