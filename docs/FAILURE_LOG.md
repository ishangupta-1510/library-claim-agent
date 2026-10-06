# Failure log

> The three worst errors on the real capture will be added after the sweep. Below are the failures
> found and fixed during development, each with its root cause and a measured before/after.

## Development failures (root cause → fix → measured effect)

### 1. Spine thickness doubled on oblique views
- **Symptom.** A 3.2 cm spine photographed at an angle measured 7.3 cm thick.
- **Root cause.** I measured the spine's upright bounding box in the raw photo. Under perspective a
  rectangle becomes a slanted quadrilateral, so its upright box is much wider than the spine.
- **Fix.** Warp each frame onto the shelf plane first, so the view is head-on, and detect spines there.
  In that view, box width divided by px/cm is the true size.
- **Measured.** At tilts of 0, 0.08 and 0.15: thickness within 5% and height within 3% after the fix,
  versus 7.3 cm for 3.2 cm (+128%) before. Covered by `tests/test_scale.py`.

### 2. Blur detector rejected 47 of 50 sharp frames
- **Symptom.** On the first real-model run, almost every frame was flagged "blur", so 14 books were
  found out of 60.
- **Root cause.** Variance of the Laplacian depends on content: plain shelves score low even when
  sharp. Calibration showed no threshold separates the cases, since a sharp plain shelf scored 11 and
  a blurred bookshop photo scored 34.
  - A second attempt, judging edge strength against the sweep's own recent frames, then flagged every
    frame that simply showed an emptier part of a shelf.
- **Fix.** The "blur effect" metric (Crete-Roffet et al., 2007): the share of detail lost when the
  frame is blurred again. It's a ratio, so it doesn't depend on how much texture the scene has.
  A frame counts as blurry if it's above 0.92, or 0.07 above the sweep's recent median.
- **Measured.** On calibration images, real sharp frames score 0.61–0.71 and real blurred ones
  0.78–0.92. On the synthetic sweep, false blur flags went from 47 to 0, and only the 4–6 frames the
  generator deliberately blurred are flagged.

### 3. Tracking fragmented each shelving unit into many pieces, and one sweep took over 10 minutes
- **Symptom.** Frames after the marker frame started new "units" with no scale. Their books got no
  dimensions and could be double-counted, and a 74-frame replay took over 10 minutes on CPU.
- **Root causes.**
  - Each frame was linked only to the previous frame, so it failed at every row change.
  - ORB features were weak on tilted views.
  - Features were recomputed for every candidate pair.
  - Bare-shelf frames with 2 features were registered as new units.
- **Fixes.**
  - SIFT features, computed once per frame and matched with FLANN.
  - Linking against the 6 most recent frames.
  - Frames with under 60 features aren't registered.
  - Loop closure: a young, unscaled fragment that overlaps an older plane is transformed into it.
- **Measured (synthetic sweep, 120 frames).**
  - Per-frame processing: from minutes for the sweep to 0.20 s median.
  - Unit A: from 10 scaled frames plus 9 unscaled fragments to all 47 frames on one scaled plane.
  - Unit B: 28 frames scaled, with two small fragments left for the agent to ask about.

### 4. Reference app's claim writer failed on the free key
- **Root cause.** Its hardcoded `gemini-3.8-flash` returned "503 high demand", and there's no retry.
- **Fix in this project.** Every model call retries 429/5xx with exponential backoff, and models are
  set in `.env`.

### 5. Free-tier vision quota: 20 requests a day
- **Root cause.** `gemini-3.5-flash` allows 20 free requests a day per project, and one sweep needs
  about 25–40.
- **Decision.** Vision runs on `gemini-3.5-flash-lite` (larger free quota). Coverage gating sends only
  frames that add 25%+ new shelf area. Price searches are cached and budgeted.

## Cost per sweep and latency per stage

Recorded in each packet under `stages` (`latency_s`, `usage`, `cost_usd_estimate`); figures from the
real sweep are to be added here.
