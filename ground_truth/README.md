# Ground truth (collected by hand, before tuning)

Fill these in from the real room, then run `python scripts/ground_truth.py` to produce
`ground_truth.json` and `python scripts/evaluate.py sweeps/<id> ground_truth/ground_truth.json`.

- `books.csv` — every physical book (true count = number of rows), the title read by eye where legible,
  and tape-measured spine height/thickness for at least 20 of them (cm, to 0.1).
- `prices.csv` — 15 books priced by hand on the same kind of source the system uses (new: Amazon.in /
  Google Shopping; used: eBay or local used listings), with URL and date.
- `items.csv` — every non-book item in the room (category from: shelving, furniture, lighting, rug,
  electronics, appliance, art, decor, plant, other).
- `room.csv` — tape measurements of the room.
