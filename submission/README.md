# Submission

| Deliverable | Where | Status |
|---|---|---|
| Repository | this archive (root `README.md`: setup and the three ways to run) | done |
| Architecture note | [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | done |
| Failure log | [`docs/FAILURE_LOG.md`](../docs/FAILURE_LOG.md) | development failures done; real-capture top 3 to come |
| Claim packet (sample) | [`sample_mock_run/`](sample_mock_run/): `claim_packet.json`, `report.html`, the frames it cites, `evaluation.json` | done (synthetic library, see below) |
| Ground truth and results | `ground_truth/` (templates), `dev_data/synthetic/ground_truth.json`, README "Results" | synthetic done; real room to come |
| Demo video (unedited) | to be added | real capture to come |

## The sample packet

`sample_mock_run/` is a packet built by the **Mock flow**: the built-in synthetic library (two units,
60 books with exact ground truth) run through the real pipeline, with the model, catalog and price
answers replayed from a recorded live run. Open `report.html` in a browser; every figure links to the
frame it came from and to its dated price source. `evaluation.json` scores it against the pass bars:

- Book count: 60 / 60
- Titles: 98.2% correct, none confidently wrong
- Spine dimensions: 19 / 20 within 15%
- Time to packet: about 11 s offline (110–125 s with live models, see the root README)

To reproduce it yourself with no keys: `python -m library_claim`, open http://localhost:8000 and press
**Mock flow (no APIs)**.
