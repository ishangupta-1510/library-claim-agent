# Library contents claim agent

A live voice-and-camera agent that inventories a home library in one walk. The policyholder talks to
the agent while panning the phone across the shelves. Every book is counted from its spine, read,
identified, measured and priced in their currency. Other contents are listed and priced, and the room
is measured. Afterwards a claim packet (`claim_packet.json` plus an HTML report) is built where every
figure traces back to a saved frame and a dated price source.

## Try it

Requirements: **Python 3.12** and **Chrome** (or Edge). Windows, macOS and Linux.

```bash
git clone <repo> && cd library-claim-agent
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python -m library_claim
```

Open **http://localhost:8000**. The page offers three ways to run, from no setup to the real thing:

| Button | Needs | What happens |
|---|---|---|
| **Mock flow (no APIs)** | nothing: no keys, no network, no camera, no microphone | The whole user journey, offline. Built-in shelf footage plays as the camera; a scripted agent talks (browser voice) and answers for you; the real pipeline counts, reads and measures the spines from **recorded** model answers and prices them from **recorded** searches; the room is marked; the claim packet and report are built. About 90 seconds. |
| **Start demo sweep** | `GOOGLE_API_KEY` (and `SERPAPI_KEY` for live prices) | The same built-in footage, but with the real Gemini Live agent (talk to it with your microphone), live spine reading and live price searches. |
| **Start sweep** | both keys, a webcam or phone, real shelves | The real thing. Show the size marker on each shelving unit (below). |

When the packet is ready, **Open claim packet** shows the HTML report; the JSON and every saved frame
and raw model/price response are in `sweeps/<sweep id>/`.

### Keys (for demo and live sweeps)

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

| Key | Where | Used for |
|---|---|---|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey (free tier works; vision is limited to ~500 calls/day) | live voice agent, spine reading, item detection |
| `SERPAPI_KEY` | https://serpapi.com (free plan) | Google Shopping and eBay listings for prices |

Restart `python -m library_claim` after editing `.env`; its first lines say which modes are ready.

### Size reference (live sweeps)

Open http://localhost:8000/marker on a second screen (laptop, tablet or phone) at full brightness, or
print `/marker.png`. Measure the black square with a ruler and type it in the **Marker … cm** box next
to Start sweep (its default comes from `MARKER_SIZE_CM` in `.env`). Hold or stand it flat against each
shelving unit at the start of that unit. Without it the sweep still counts
and reads books, but sizes stay empty.

### Phone (camera + room measurement)

Browsers only give the camera to `localhost` or HTTPS pages. Put an HTTPS proxy in front of the app,
for example with Tailscale on both devices:

```bash
tailscale serve --bg 8000      # prints https://<this-computer>.<tailnet>.ts.net
```

Open that address in Chrome on an ARCore phone. **Start AR sweep** adds room measurement: tap
**Mark corner** at each floor corner and **Mark ceiling** once.

### Live agent on recorded footage (OBS Virtual Camera)

To try the real camera path and the live agent without shelves: make a video of the built-in shelf,
play it in OBS as a Media Source and start OBS's **Virtual Camera**:

```bash
ffmpeg -framerate 0.4 -i dev_data/synthetic/frames/%04d.jpg -vf "fps=30,format=yuv420p" synthetic-shelf-sweep.mp4
```

On the page pick **OBS Virtual Camera** in the camera list, set **Marker** to **10** cm (the marker
printed in that footage) and press **Start sweep**. Results from this route are on the synthetic
library, not a real room.

### Without the browser

```bash
python -m scripts.replay dev_data/synthetic/frames --truth dev_data/synthetic/ground_truth.json --marker-cm 10
python -m scripts.evaluate sweeps/<sweep id> ground_truth/ground_truth.json
```

`replay` feeds a folder of frames through the pipeline (needs `GOOGLE_API_KEY`) and scores it against
every pass bar. Tests run fully offline: `pip install -r requirements-dev.txt && pytest`.

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

**Running it (2026-10-06, same free-tier key):**
- The live session ran on `gemini-3.8-live`, and typed and spoken turns worked.
- Its background claim team started up: the policy desk looked up the policy number and interrupted
  the agent with "no match".
- The claim writer failed every update. Its extraction model (`gemini-3.8-flash`, hardcoded) returned
  "503 high demand", and with no retry the notebook stayed at 0% collected.
- The sketch artist also failed, because there's no image-model quota on the free key.

Lesson carried over: every model call here retries on 429/5xx with backoff, and models are
configurable through `.env`.

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

**Synthetic library** (two units, 60 books, exact ground truth; `dev_data/synthetic`). Two live-model
replays at real capture pace (`python -m scripts.replay ... --truth ...`), Gemini Flash-Lite vision:

| Pass bar | Target | Run 1 | Run 2 |
|---|---|---|---|
| Book count | within 5% | 60 / 60 | 60 / 60 |
| Titles | ≥ 70% right, ≤ 3% confidently wrong | 98.2%, 0 wrong | 98.2%, 0 wrong |
| Spine dimensions | 20 books within 15% | 18 / 20 (median error: height 0.6%, thickness 4.5%) | 19 / 20 (0.7%, 3.9%) |
| Time to packet | under 5 minutes | 112 s | 122 s |

The dimension misses are the thinnest spines (1.5 cm): the model's boxes run about 0.3 cm wide, which
alone is 20% there. Snapping box sides to image edges was tried and made it worse (see the failure log).
Prices, items and room areas need a real room: prices against hand-checked listings, items against a
hand list, areas against a tape.

**Real room:** to be filled in from the capture. `ground_truth/` holds the hand-collected sheet, and
`python -m scripts.evaluate` scores a sweep against every pass bar.

## Tools used

Gemini (Live API for voice, Flash for spine and item vision), Google Books and Open Library APIs,
SerpAPI (Google Shopping, eBay), frankfurter.dev (ECB exchange rates), OpenCV (ArUco, SIFT,
homographies), WebXR with ARCore (room points), FastAPI. Code written with Claude Code (Anthropic) as
an AI coding assistant.
