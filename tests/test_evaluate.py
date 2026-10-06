"""The evaluator is checked on a hand-made packet before its numbers are trusted."""

from scripts.evaluate import evaluate


def b(t, h, th, status="identified", conf=0.9):
    return {"title": t, "status": status, "id_confidence": conf, "spine_height_cm": h, "spine_thickness_cm": th,
            "replacement_cost": {"amount": None}}


def test_evaluator_counts_correct_wrong_and_dimension_errors():
    truth = {"book_count": 4, "room": {"floor_area_m2": 12.0}, "books": [
        {"title": "Sapiens", "legible": True, "height_cm": 20, "thickness_cm": 3},
        {"title": "Dune", "legible": True, "height_cm": 18, "thickness_cm": 4},
        {"title": "Quiet", "legible": True, "height_cm": 22, "thickness_cm": 2},
        {"title": "", "legible": False, "height_cm": 21, "thickness_cm": 2}]}
    packet = {"books": [b("Sapiens", 21, 3.2), b("Dune", 25, 4), b("Homo Deus", 22, 2), b("", 21, 2, "unidentified", 0)],
              "items": [], "room": {"floor_area_m2": 12.9}, "stages": {"time_to_packet_s": 120}}
    r = evaluate(packet, truth)
    assert r["book_count"]["pass"]
    assert r["titles"]["correct"] == 2 and r["titles"]["confidently_wrong"] == 1 and not r["titles"]["pass"]
    assert r["spine_dimensions"]["sample"] == 2 and r["spine_dimensions"]["within_15pct"] == 1
    assert r["floor_area_m2"]["error_pct"] == 7.5 and r["floor_area_m2"]["pass"]
    assert r["time_to_packet_s"]["pass"]
