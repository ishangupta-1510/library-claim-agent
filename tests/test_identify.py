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
