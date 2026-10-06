# Library contents claim agent

A live voice-and-camera agent that inventories a home library in one walk. The policyholder talks to
the agent while panning the phone across the shelves. Every book is counted from its spine, read,
identified, measured and priced in their currency. Other contents are listed and priced, and the room
is measured. Afterwards a claim packet (`claim_packet.json` plus an HTML report) is built where every
figure traces back to a saved frame and a dated price source.

> Status: the pipeline, the live agent and the tests are complete. Results on a real room are still
> to come. See [Results](#results).

## Run it (about 10 minutes)

Requirements: Python 3.12, Chrome. `ffmpeg` is needed only to regenerate the synthetic test video.

```bash
git clone <repo> && cd library-claim-agent
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # then fill in the keys below
uvicorn library_claim.server:app --host 0.0.0.0 --port 8000
```

| Key | Where | Used for |
|---|---|---|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey (free tier works) | live voice agent, spine reading, item detection |
| `SERPAPI_KEY` | https://serpapi.com (free plan: 100 searches/month) | Google Shopping and eBay listings for prices |

Then open http://localhost:8000 and press **Start sweep**.

**Size reference.** Open http://localhost:8000/marker on a laptop at full brightness, measure the black
square with a tape, set `MARKER_SIZE_CM` in `.env`, and stand the laptop on each shelving unit. A
printed A4 copy of `/marker.png` works too.

**Phone (room measurement).** WebXR needs HTTPS, so expose the server over an HTTPS tunnel (for example
`tailscale serve 8000`), open it in Chrome on an ARCore phone and press **Start AR sweep**. The agent
then asks you to tap **Mark corner** at each floor corner and **Mark ceiling** once.

**Mock camera (no phone, no books).** Chrome can play a video file as its webcam:

```bash
python -m dev_data.synthetic       # renders 2 units, 60 books, exact ground truth, sweep.mjpeg
chrome --use-fake-device-for-media-stream --use-file-for-fake-video-capture=dev_data/synthetic/sweep.mjpeg http://localhost:8000
```

**Without voice:** replay a folder of frames through the pipeline and score it:

```bash
python scripts/replay.py dev_data/synthetic/frames --truth dev_data/synthetic/ground_truth.json
python scripts/evaluate.py sweeps/<sweep_id> ground_truth/ground_truth.json
```

**Tests:** `pip install -r requirements-dev.txt && pytest`. They run fully offline, with network and models faked.

## How it works

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the one-page diagram. In short, there are two layers:

- **Live layer.** Gemini Live hears the policyholder and sees a 1 fps low-res feed. It directs the sweep
  and records corrections through tools (`note_book_in_view`, `exclude_shelf_in_view`,
  `answer_art_question`, `set_locale`, `end_sweep`). It never reports inventory itself. It only speaks
  facts the pipeline measured, which reach it as `[system]` updates.
- **Evidence layer.** Full-resolution keyframes are saved to disk; each one is a `frame_ref`. Separate
  stages run on them, each with its own module, tests and recorded output:

| Stage | Module | What it produces |
|---|---|---|
| Capture quality | `stages/quality.py` | blur, glare and darkness from image statistics; drives "slow down" prompts |
| Metric scale | `stages/scale.py` | ArUco marker → homography from pixels to cm on the shelf plane; head-on rectified view |
| Tracking | `stages/tracking.py`, `sweep.py` | frames chained onto each unit's plane, so scale and positions carry across frames |
| Spine reading | `stages/vision.py` | boxes and verbatim spine text (Gemini); unreadable spines kept with no title |
| Inventory | `stages/inventory.py` | each physical book once (merge in plane coordinates), shelf and position, median dimensions |
| Identification | `stages/identify.py` | match to Google Books / Open Library; edition only when the spine's publisher pins it |
| Book pricing | `stages/pricing.py` | median of matched listings, with source, URL and date; conversions labelled; rare copies to appraisal |
| Item pricing | `stages/item_pricing.py` | legible model: median; unknown brand: interquartile range; art: appraisal unless confirmed print |
| Room | `stages/room.py` | floor polygon from ARCore points (shoelace), minimum bounding rectangle, perimeter × height |
| Totals and review | `totals.py` | totals in code from the lines; review queue with a reason per line |
| Report | `report.py` | HTML built only from the packet, with links to frames and sources |

## Reference app: kept, changed, thrown away

Based on [insurance_claim_live_agent_team](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/voice_ai_agents/insurance_claim_live_agent_team),
which I read and ran first.

- **Kept:** the interaction pattern (one WebSocket relaying mic PCM and camera JPEGs into a Gemini Live
  session, transcripts and audio back, function calls run server-side, interruptions handled).
- **Changed:**
  - The reference's camera frames are 640 px at about 1 fps, which is enough to talk about but not to
    read spines. Here they still go to the agent, but a separate full-resolution keyframe stream feeds
    the measuring pipeline.
  - The agent's prompt is about directing capture, not describing it.
  - Background work is deterministic stages, not narrative agents.
- **Thrown away:** policy lookup, intake rules and mock policy records, sketch generation, the avatar,
  and the ADK graph. None of them help to count, measure or price.

## Decisions and assumptions

Where the brief is ambiguous I decided as follows:

- **Shelf run.** Reported as the summed measured thickness of the books (the length of shelf the
  collection occupies), not the length of the shelf boards.
- **Wall area.** Reported gross: perimeter × ceiling height, with doors and windows not subtracted.
  **Shelved wall area** is the outer extent of each unit's books on its measured plane.
- **Book count.** Includes shelves the policyholder excluded ("not mine"). Those books are listed but
  carry no money, and they're counted separately in `books_excluded_by_policyholder`.
- **Replacement cost.** The median of new listings in the claim country (Google Shopping via SerpAPI).
- **Used value.** Local used listings where they exist. Otherwise US eBay used listings converted at the
  ECB rate of the day, labelled `converted` with the rate and its date. Condition is assumed "good"
  unless the policyholder says otherwise.
- **Edition.** Filled only when the spine's publisher text matches exactly one English edition.
  Otherwise the work is identified and the ISBN stays empty.
- **Rare copies.** A book is sent to appraisal, not priced, when there is copy-level evidence:
  - the pinned edition predates 1950,
  - the camera saw age cues (cloth, gilt, leather),
  - "signed" or "first edition" was read on the spine or said by the policyholder, or
  - the replacement cost is at or above `APPRAISAL_THRESHOLD` (₹10,000).
  A modern copy of an old work is priced normally.
- **Locale.** Locale is a setting (`COUNTRY`, plus `set_locale` when the policyholder confirms). The
  packet's `locale_comparison` prices the same 10 books in `COMPARE_COUNTRY`.

## Results

To be filled in from the real capture: `ground_truth/` holds the hand-collected sheet, and
`scripts/evaluate.py` writes each sweep's numbers against every pass bar.

## Tools used

Gemini (Live API for voice, Flash for spine and item vision), Google Books and Open Library APIs,
SerpAPI (Google Shopping, eBay), frankfurter.dev (ECB exchange rates), OpenCV (ArUco, ORB,
homographies), WebXR with ARCore (room points), FastAPI. Code written with Claude Code (Anthropic) as
an AI coding assistant.
