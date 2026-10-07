# Submission

> **No real room was available.** Every run below is on the built-in **synthetic library** (two shelving
> units, 60 books, drawn with exact ground truth in `dev_data/synthetic/`). The policyholder's voice in
> the live demo is synthesized speech. Nothing here is presented as a real-room capture.

| Deliverable | Where |
|---|---|
| Repository | this repo (root `README.md`: setup and the ways to run) |
| Demo video (unedited) | [`live_demo_run/demo-video-live-agent-synthetic-library.mp4`](live_demo_run/demo-video-live-agent-synthetic-library.mp4) |
| Claim packet | [`live_demo_run/claim_packet.json`](live_demo_run/claim_packet.json) and [`report.html`](live_demo_run/report.html) |
| Ground truth and results | `dev_data/synthetic/ground_truth.json`, [`live_demo_run/evaluation.json`](live_demo_run/evaluation.json), root README "Results" |
| Architecture note | [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) |
| Failure log | [`docs/FAILURE_LOG.md`](../docs/FAILURE_LOG.md) |

## `live_demo_run/`: the live agent on the synthetic library

One unattended run of `python -m scripts.record_demo --mode live` (2026-10-07):

- The camera is the synthetic sweep (Chrome's file-backed camera); the microphone is a synthesized voice
  that says "I'm in India" and, half-way, "That blue book on the top shelf is a signed copy".
- The **real** Gemini Live agent hears it and answers; spines are read live by Gemini Flash-Lite;
  books are identified on Open Library; prices come from SerpAPI results (cached from earlier runs).
- The video is the app page itself (Chrome's screencast), with the recorded conversation as its sound
  track; `conversation.m4a` is that conversation on its own.

| Pass bar | Target | This run |
|---|---|---|
| Book count | within 5% | 61 for 60 (1.7%) ✅ |
| Titles | ≥ 70% right, ≤ 3% confidently wrong | 98.2% right, 1 confidently wrong (1.7%) ✅ |
| Spine dimensions | 20 books within 15% | 17 / 20 (median error: height 0.65%, thickness 5.4%) ❌ |
| Time to packet | under 5 minutes | 118 s ✅ |

The spoken "signed copy" was attached to the book at the centre of the view when it was said
(B044, *Norwegian Wood*), which was then sent to appraisal instead of being priced. Prices, items and
room areas have no real-world truth here; the room is a 4.2 × 3.6 m example in the mock flow only.

## `sample_mock_run/`: the offline mock flow

The same library through the **Mock flow (no APIs)** button: recorded model, catalog and price answers,
scripted agent. Open `report.html`; reproduce it with no keys: `python -m library_claim`, then **Mock
flow (no APIs)** at http://localhost:8000.
