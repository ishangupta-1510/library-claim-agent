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

### 6. Used values summed to 50 times the replacement cost
- **Symptom.** Books' used value ₹949,372 against replacement ₹18,458 (*Long Walk to Freedom* used: ₹404,395).
- **Root causes.** Signed first editions from rare-book dealers were medianed as "used, good"; short titles
  matched unrelated products ("Deep Work" matched JDM wheel rims via fuzzy partial matching).
- **Fix.** A listing counts only if it starts with the book's title, any "by" names the author, and it is
  not another format, a translation, a lot or a collectible copy. Listings without a link are not
  evidence. A used value exists only beside a replacement and never above it.
- **Measured.** Used ₹2,256 against replacement ₹13,863; *The Midnight Library* ₹300 instead of ₹2,437.

### 7. Bottom shelves were never read
- **Symptom.** 16 of 17 missed titles were on the bottom shelf of each unit.
- **Root cause.** The frame that shows the last row whole adds little new area (32%) and was never sent to
  vision; earlier frames showed that row only cut off at their border.
- **Fix.** Coverage counts only a sent frame's interior; when a pass over a unit ends, its last unsent
  frame is sent if 20%+ of it is still unseen.
- **Measured.** Titles 59% → 98%; book count 54 → 60 of 60.

### 8. Catalog lookups failed for well-known books
- **Root causes.** Accented names lost their letters ("Héctor García" → "h ctor garc a"); Open Library
  lists some authors only in native script (村上春樹); subtitles folded into catalog titles.
- **Fix.** Accent folding, Open Library alternative names, matching on the main title.

### 9. One spine counted twice
- **Root causes.** The model labelled upright spines "flat" in one frame; one frame was offset by a spine
  width; one detection split a spine in two; one box ran into the next row and stretched a spine to 41 cm.
- **Fixes.** Orientation from box shape on the rectified plane; read titles used as landmarks to shift a
  misregistered frame; neighbouring columns merged when more frames saw one spine; spine ends as medians.
- **Measured.** Count 61–64 → 60 in both runs; confidently wrong titles 6 → 0.

### 10. Snapping box sides to image edges (reverted)
- **Idea.** Thin spines measure ~0.3 cm too thick because the model's boxes are loose; snap each side to
  the strongest nearby vertical edge.
- **Measured.** Dimension passes fell from 19/20 to 9–14/20: the strongest edge is often a band or
  lettering on the spine, not its boundary. Reverted.

### 11. Live agent: looped on the marker, and set the currency from noise
- **Symptoms.** The agent asked for "the laptop marker" every few seconds without saying what it is (the
  tester held up the laptop itself); separately it set USD after the microphone heard "Sh".
- **Fixes.** The opening explains the marker; the request is made at most twice, a minute apart, and
  the sweep never waits on it. `set_locale` must quote the policyholder's words, and the server refuses
  it unless they name the country or currency.

### 12. Late vision answers and policyholder notes landing on the wrong unit or book (code review)
- **Root causes.** A queued vision job kept the unit object it was queued with, after that unit had been
  merged into another; unit ids were reused; spoken notes were keyed by object id and shelf labels.
- **Fixes.** Stable unit ids; an answer is mapped into the frame's current unit geometry when it arrives;
  notes and exclusions live on the books (exclusions by position on the unit) and survive merges.

### 13. Daily vision quota ran out mid-sweep
- **Root cause.** Free tier: 500 Flash-Lite requests a day; development runs used them up, and the packet
  then waited on retries that could not succeed.
- **Fix.** A per-day 429 stops vision for the sweep at once; the packet builds from what was read and
  its stage report says how many frames were not read.

## Cost per sweep and latency per stage

Recorded in each packet under `stages` (`latency_s`, `usage`, `cost_usd_estimate`); figures from the
real sweep are to be added here.
