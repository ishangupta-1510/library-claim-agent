

def test_marker_request_is_capped_and_never_blocks(monkeypatch):
    from library_claim import live

    clock = [0.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: clock[0])
    narrator = live.Narrator()
    asked = 0
    for i in range(300):  # five minutes of frames, each on a "new" unscaled unit
        clock[0] = i
        out = narrator.on_frame({"problems": [], "plane": f"P{i}", "metric": False})
        asked += bool(out)
    assert asked == live.Narrator.MARKER_ASKS
