# Architecture

```
 PHONE (Chrome)                          SERVER (FastAPI, one WebSocket per sweep)
 ───────────────                         ────────────────────────────────────────────────────────────────────
 mic 16 kHz PCM ───────────────────────► Gemini Live (native audio) ◄── [system] facts from the pipeline
 camera 640 px @1 fps ─────────────────►   speaks directions, asks questions, calls tools:
 agent voice 24 kHz ◄──────────────────    set_locale · note_book_in_view · exclude_shelf_in_view
                                           answer_art_question · end_sweep
 keyframes 1920 px (back-pressured) ───► DURING THE SWEEP (per keyframe, worker thread)
                                           save frame (= frame_ref) → quality (blur / glare / dark)
                                           → ArUco marker → plane homography (px → cm)
                                           → or chain to recent frames (SIFT + RANSAC); loop closure
                                           → ≥ 50 % unseen shelf, or end of a pass ? → vision
                                         VISION QUEUE (Gemini Flash-Lite, 3 in flight, JSON schema)
                                           spines on the head-on rectified view: box + verbatim text
                                           items on the raw frame: category, material, legible brand
                                           → inventory merge in plane coordinates → live UI + [system]
 ARCore WebXR hit tests ───────────────►   floor corners and ceiling point, metres
                                         AFTER THE SWEEP (concurrent, each stage timed)
                                           identify (Google Books / Open Library, fuzzy, conservative)
                                           → price books (SerpAPI Google Shopping IN; eBay US → INR, ECB)
                                           → price items (median or interquartile range; art → appraisal)
                                           → room (shoelace, min rectangle, perimeter × height)
                                           → totals + review queue in code → claim_packet.json → report.html
```

## Which model does what

| Job | Model / tool | Why |
|---|---|---|
| Conversation, directing the sweep | Gemini Live native-audio model | real-time voice in and out, barge-in, tool calls |
| Spine boxes and transcription | Gemini Flash-Lite on the rectified shelf view | reads small rotated text; boxes come back on a 0–1000 grid; strict "copy, don't guess" schema |
| Non-book items | Gemini Flash-Lite on raw frames | category, material, legible brand; never prices |
| Everything numeric | plain code | homographies, medians, areas and totals are deterministic and tested |

## Where metric scale comes from

- **Spines.** An ArUco marker of known, tape-measured size, shown on a laptop or printed, sits on each
  shelving unit. Its corners give a homography from image pixels to centimetres on the shelf plane.
  - Frames are re-projected head-on before spines are detected, so a box's width divided by the
    resolution (px per cm) is the true thickness.
  - Measuring boxes in the raw, oblique photo overstated a 3.2 cm spine as 7.3 cm. Rectification
    brought it to within 5%, and `tests/test_scale.py` checks this.
  - Frames without the marker inherit scale by chaining homographies to a frame that has it. That's
    exact for a plane, and the chain stops at a weak link rather than drifting.
- **Room.** ARCore motion tracking via WebXR hit tests gives metric 3D points in one world frame
  during the same walk.

## Where prices come from

- **Every figure is traceable.** It's the median of matched listings returned by SerpAPI (Google
  Shopping in the claim country, eBay for used copies). Each one carries its merchant, URL, retrieval
  time and the rule used to pick it.
- **Conversions are labelled.** Converted figures carry the ECB rate and its date (frankfurter.dev).
- **The model never produces a price.** With no listing, the amount stays empty, the line goes to the
  review queue and it's excluded from totals.
- **Raw responses are kept.** Every API response is saved in `sweeps/<id>/raw/` for audit.

## Mock flow (offline)

The **Mock flow** button runs the same pipeline with every external call replaced by a recording, so a
reviewer can see the whole journey with no keys, network, camera or microphone:

- **Vision.** `RecordedVision` answers each call by its id (`vision_spines-f0003`) from a real run over
  the built-in footage. The pipeline is deterministic on the same frames, so it asks for the same ids.
- **Catalogs and FX.** `cassette.py` replays recorded HTTP responses, keyed by URL with credentials
  removed. A request that wasn't recorded gets a 404, which the lookup code treats as "nothing found".
- **Prices.** `PriceClient(offline=True)` answers only from recorded searches.
- **Agent.** `mock.py` stands in for Gemini Live. It speaks the same capture directions the live agent is
  given (from the same `Narrator`), and the page answers for the policyholder and marks a 4.2 × 3.6 m room.

`python -m scripts.record_mock_fixtures <sweep>` rebuilds the recordings from a real replay without
spending quota. `tests/test_mock_flow.py` runs the mock pipeline with the network blocked.
