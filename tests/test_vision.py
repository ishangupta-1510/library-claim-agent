from library_claim.stages.vision import box_to_px, parse_items, parse_spines


def test_boxes_convert_from_gemini_grid():
    assert box_to_px([100, 200, 500, 300], 1000, 2000) == (200, 200, 300, 1000)
    assert box_to_px([500, 300, 100, 200], 1000, 1000) is None  # inverted
    assert box_to_px([1, 2, 3], 100, 100) is None


def test_illegible_spine_keeps_box_but_drops_any_written_title():
    payload = {"books": [
        {"box_2d": [0, 0, 900, 40], "orientation": "upright", "title": "Probably Dune", "author": "Herbert",
         "publisher": "", "all_text": "D?N", "legible": False, "age_cues": []},
        {"box_2d": [0, 50, 900, 90], "orientation": "upright", "title": "Sapiens", "author": "Harari",
         "publisher": "Vintage", "all_text": "SAPIENS HARARI VINTAGE", "legible": True, "age_cues": [" "]},
    ]}
    spines = parse_spines(payload, 1000, 1000)
    assert len(spines) == 2  # unreadable books still count
    assert spines[0].title == "" and spines[0].author == "" and not spines[0].legible
    assert spines[1].title == "Sapiens" and spines[1].age_cues == []


def test_legible_flag_without_title_is_not_legible():
    payload = {"books": [{"box_2d": [0, 0, 10, 10], "orientation": "flat", "title": "", "author": "", "publisher": "",
                          "all_text": "", "legible": True, "age_cues": []}]}
    assert parse_spines(payload, 100, 100)[0].legible is False


def test_items_reject_unknown_categories_and_mark_art():
    payload = {"items": [
        {"box_2d": [0, 0, 10, 10], "category": "art", "description": "portrait", "material": "oil", "brand_model": "", "is_artwork": False},
        {"box_2d": [0, 0, 10, 10], "category": "spaceship", "description": "?", "material": "", "brand_model": "", "is_artwork": False},
    ]}
    items = parse_items(payload, 100, 100)
    assert len(items) == 1 and items[0].is_artwork
