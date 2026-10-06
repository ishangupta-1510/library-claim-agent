from library_claim.stages.identify import Candidate, SpineReading, choose


def cand(title, authors, publisher="", isbn="", source="Google Books", subtitle=""):
    return Candidate(title, authors, publisher, isbn, "2001", source, "https://example.org/" + title, subtitle)


ORWELL = cand("Nineteen Eighty-Four", ["George Orwell"], "Penguin Books", "9780141036144")
ANIMAL = cand("Animal Farm", ["George Orwell"], "Penguin Books", "9780141036137")


def test_title_and_surname_identify_the_work():
    result = choose(SpineReading(title="Nineteen Eighty-Four", author="ORWELL"), [ANIMAL, ORWELL])
    assert result.status == "identified"
    assert result.title == "Nineteen Eighty-Four"
    assert result.confidence >= 0.85
    assert result.isbn == ""  # no publisher on the spine: work, not edition


def test_publisher_on_spine_pins_the_edition():
    result = choose(SpineReading(title="Nineteen Eighty-Four", author="Orwell", publisher="PENGUIN"), [ORWELL])
    assert result.isbn == "9780141036144"
    assert "Penguin" in result.edition


def test_wrong_author_is_not_accepted():
    other = cand("Nineteen Eighty-Four", ["Someone Else"])
    assert choose(SpineReading(title="Nineteen Eighty-Four", author="Orwell"), [other]).status == "unidentified"


def test_partial_title_is_not_guessed():
    # A half-read spine ("NINETEEN") must not become 1984 just because it is the closest record.
    assert choose(SpineReading(title="Nineteen", author="Orwell"), [ORWELL]).status == "unidentified"


def test_title_only_needs_near_exact_and_is_capped():
    result = choose(SpineReading(title="Animal Farm"), [ANIMAL])
    assert result.status == "identified"
    assert result.confidence <= 0.7  # always reviewed


def test_one_word_title_without_author_stays_unidentified():
    dune = cand("Dune", ["Frank Herbert"])
    assert choose(SpineReading(title="Dune"), [dune]).status == "unidentified"


def test_blank_spine_never_identified():
    assert choose(SpineReading(), [ORWELL]).status == "unidentified"


def test_leading_article_and_subtitle_tolerated():
    hobbit = cand("The Hobbit", ["J. R. R. Tolkien"], subtitle="Or There and Back Again")
    result = choose(SpineReading(title="Hobbit", author="Tolkien"), [hobbit])
    assert result.status == "identified"


from library_claim.stages.identify import Edition, Identification, pin_edition


def _work():
    return Identification("identified", 1.0, "Nineteen Eighty-Four", "George Orwell", source="Open Library")


def test_single_publisher_edition_pins_isbn():
    editions = [Edition("Penguin Books", "9780141036144", "2008", "eng", "u1"), Edition("Gyldendal", "9788702291872", "2019", "dan", "u2")]
    result = pin_edition(_work(), "PENGUIN", editions)
    assert result.isbn == "9780141036144"


def test_many_printings_leave_isbn_empty():
    editions = [Edition("Penguin Books", "9780141036144", "2008", "eng", "u1"), Edition("Penguin", "9780451524935", "1961", "eng", "u2")]
    result = pin_edition(_work(), "Penguin", editions)
    assert result.isbn == ""
    assert "exact printing not visible" in result.edition


def test_foreign_language_editions_ignored():
    editions = [Edition("Penguin", "9788700000000", "2019", "dan", "u1")]
    result = pin_edition(_work(), "Penguin", editions)
    assert result.isbn == "" and result.edition == ""


def test_a_partial_reading_needs_the_author():
    from library_claim.stages.identify import Candidate, SpineReading, score

    cand = Candidate("Homo Deus", ["Yuval Noah Harari"], "Vintage", "", "2017", "Google Books", "u")
    assert not score(SpineReading("Homo Deus", "", "", "", partial=True), cand)[0]
    accepted, _, reasons = score(SpineReading("Homo Deus", "Yuval Noah Harari", "", "", partial=True), cand)
    assert accepted and "read from a spine partly outside the frame" in reasons
    assert not score(SpineReading("DEUS", "Yuval Noah Harari", "", "", partial=True), cand)[0]


def test_accented_and_native_script_author_names_match():
    from library_claim.stages.identify import score

    ikigai = Candidate("Ikigai", ["Héctor García", "Francesc Miralles"], "", "", "2016", "Open Library", "u")
    assert score(SpineReading("IKIGAI", "Hector Garcia", "", ""), ikigai)[0]
    wood = Candidate("Norwegian Wood", ["村上春樹"], "", "", "1987", "Open Library", "u",
                     author_aliases=["MURAKAMI HARUKI", "村上春树", "Haruki Murakami"])
    result = choose(SpineReading("NORWEGIAN WOOD", "Haruki Murakami", "", ""), [wood])
    assert result.status == "identified" and result.author == "Haruki Murakami"


def test_subtitle_folded_into_the_catalog_title_still_matches():
    cand = Candidate("Man's Search for Meaning : An Introduction to Logotherapy", ["Viktor Emil Frankl"], "", "", "1959",
                     "Open Library", "u")
    assert choose(SpineReading("MAN'S SEARCH FOR MEANING", "Viktor E. Frankl", "", ""), [cand]).status == "identified"


def test_folded_subtitle_is_not_shown_as_the_title():
    cand = Candidate("Man's Search for Meaning : An Introduction to Logotherapy", ["Viktor Emil Frankl"], "", "", "1959",
                     "Open Library", "u")
    assert choose(SpineReading("MAN'S SEARCH FOR MEANING", "Frankl", "", ""), [cand]).title == "Man's Search for Meaning"


async def test_google_books_is_dropped_for_the_run_once_unusable(monkeypatch):
    import httpx

    from library_claim import net
    from library_claim.stages.identify import Catalogs, fetch_candidates

    async def no_sleep(_):
        return None

    monkeypatch.setattr(net.asyncio, "sleep", no_sleep)
    calls = []

    def handler(request):
        calls.append((request.url.host, "x-goog-api-key" in request.headers))
        assert "key" not in request.url.params  # the key never travels in the URL
        if request.url.host == "www.googleapis.com":
            return httpx.Response(401 if "x-goog-api-key" in request.headers else 429, json={})
        return httpx.Response(200, json={"docs": [{"title": "Ikigai", "author_name": ["Héctor García"], "key": "/works/X"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        catalogs = Catalogs(http, "gemini-only-key")
        first = await fetch_candidates(SpineReading("IKIGAI", "Hector Garcia", "", ""), catalogs)
        google_calls = sum(host == "www.googleapis.com" for host, _ in calls)
        second = await fetch_candidates(SpineReading("SAPIENS", "Harari", "", ""), catalogs)
    assert first and second  # Open Library still answers
    assert sum(host == "www.googleapis.com" for host, _ in calls) == google_calls  # no Google calls after it failed
    assert len(catalogs.events) == 2 and not catalogs.google_available
