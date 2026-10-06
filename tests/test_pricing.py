"""Pricing logic on recorded-shape responses; network calls are faked."""

import pytest

from library_claim.config import locale_for
from library_claim.stages.pricing import (
    FxRate,
    Listing,
    appraisal_check,
    parse_ebay,
    parse_google_shopping,
    pick,
    price_book,
    relevant,
)

IN, US = locale_for("IN"), locale_for("US")


class FakeClient:
    def __init__(self, shopping=(), ebay=(), rate=96.3):
        self._shopping, self._ebay, self._rate = list(shopping), list(ebay), rate
        self.queries = []

    async def shopping(self, query, locale):
        self.queries.append(("shopping", query, locale.country))
        return [l for l in self._shopping if l.currency == locale.currency]

    async def ebay_used(self, query, locale):
        self.queries.append(("ebay", query, locale.country))
        return self._ebay

    async def fx(self, base, quote):
        return FxRate(self._rate, "2026-10-05")


def L(title, amount, currency="INR", condition="new", merchant="Amazon.in"):
    return Listing(title, amount, currency, merchant, f"https://shop/{merchant}/{amount}", condition)


async def _price(client, **kw):
    args = dict(title="Nineteen Eighty-Four", author="George Orwell", isbn="", edition_year="", notes=[], spine_text="",
                visual_flags=[], locale=IN, fallback=US, threshold=10000, client=client)
    args.update(kw)
    return await price_book(**args)


async def test_median_of_local_new_listings_with_source():
    client = FakeClient([L("Nineteen Eighty-Four (Penguin)", 199), L("Nineteen Eighty-Four Paperback", 250, merchant="Flipkart"), L("Nineteen Eighty-Four", 1200, merchant="Rare Co")])
    result = await _price(client)
    assert result.replacement.amount == 250  # median, not the 1200 outlier
    assert result.replacement.url and result.replacement.retrieved_at
    assert not result.replacement.converted
    assert "median of 3" in result.replacement.basis


async def test_used_falls_back_to_converted_us_listing():
    client = FakeClient([L("Nineteen Eighty-Four", 250)], ebay=[L("Nineteen Eighty-Four Orwell", 2.0, "USD", "used", "eBay")])
    result = await _price(client)
    used = result.used
    assert used.converted and used.original_currency == "USD" and used.currency == "INR"
    assert used.amount == pytest.approx(2.0 * 96.3)
    assert "2026-10-05" in used.fx_source


async def test_nothing_found_means_no_amount():
    result = await _price(FakeClient())
    assert result.replacement.amount is None and result.used.amount is None
    assert "no new listing found in local market" in result.notes


async def test_modern_copy_of_an_old_work_is_priced():
    # 1984 was first published in 1949; a modern paperback is not antiquarian.
    result = await _price(FakeClient([L("Nineteen Eighty-Four", 250)]))
    assert not result.appraisal.needed and result.replacement.amount == 250


async def test_old_edition_goes_to_appraisal_without_lookup():
    client = FakeClient([L("Nineteen Eighty-Four", 250)])
    result = await _price(client, edition_year="1949")
    assert result.appraisal.needed
    assert result.replacement.amount is None
    assert client.queries == []  # never priced


async def test_visual_age_cues_go_to_appraisal():
    result = await _price(FakeClient([L("Nineteen Eighty-Four", 250)]), visual_flags=["cloth binding with gilt lettering"])
    assert result.appraisal.needed


async def test_signed_copy_from_user_voice_note_is_not_priced():
    result = await _price(FakeClient([L("Sapiens", 499)]), title="Sapiens", notes=["user: this copy is signed"])
    assert result.appraisal.needed and result.replacement.amount is None


async def test_above_threshold_is_routed_to_appraisal():
    result = await _price(FakeClient([L("Atlas of Everything", 15000)]), title="Atlas of Everything")
    assert result.appraisal.needed and result.replacement.amount is None


def test_study_guides_are_not_the_book():
    assert not relevant("Nineteen Eighty-Four Study Guide", "Nineteen Eighty-Four", "Orwell")
    assert relevant("Nineteen Eighty-Four (Penguin Modern Classics)", "Nineteen Eighty-Four", "Orwell")


def test_parsers_skip_listings_without_numeric_price():
    shopping = parse_google_shopping({"shopping_results": [
        {"title": "A", "extracted_price": 299, "source": "Amazon.in", "product_link": "u"},
        {"title": "B", "price": "see site", "source": "X"},
        {"title": "C", "extracted_price": 150, "second_hand_condition": "used", "source": "Y", "link": "v"},
    ]}, "INR")
    assert [(l.title, l.condition) for l in shopping] == [("A", "new"), ("C", "used")]
    ebay = parse_ebay({"organic_results": [{"title": "D", "price": {"extracted": 5.5}, "link": "w", "condition": "Pre-Owned"}]}, "USD")
    assert ebay[0].condition == "used" and ebay[0].amount == 5.5


def test_pick_ignores_other_condition():
    assert pick([L("x", 100, condition="used")], "new") is None


def test_rarity_words_and_year():
    assert appraisal_check("Dune", "1990", [], "", []).needed is False
    assert appraisal_check("Ulysses", "1922", [], "", []).needed is True
    assert appraisal_check("Dune", "", [], "SIGNED BY AUTHOR", []).needed is True


async def test_cached_search_keeps_its_original_date(tmp_path):
    import httpx

    from library_claim.stages.pricing import PriceClient

    calls = []

    def handler(request):
        calls.append(request.url.params["q"])
        return httpx.Response(200, json={"shopping_results": [
            {"title": "Sapiens", "extracted_price": 499, "source": "Amazon.in", "product_link": "https://a/1"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        first = PriceClient(http, "key", cache_dir=tmp_path)
        a = await first.shopping("Sapiens Harari book", IN)
        second = PriceClient(http, "key", cache_dir=tmp_path)
        b = await second.shopping("Sapiens Harari book", IN)
    assert calls == ["Sapiens Harari book"]  # the second run used the cache
    assert a[0].retrieved_at == b[0].retrieved_at != ""
    assert first.live_searches == 1 and second.live_searches == 0


async def test_a_search_that_keeps_timing_out_leaves_the_line_unpriced(tmp_path, monkeypatch):
    import httpx

    from library_claim import net
    from library_claim.stages.pricing import PriceClient

    monkeypatch.setattr(net.asyncio, "sleep", _no_sleep)
    attempts = []

    def handler(request):
        attempts.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = PriceClient(http, "key", cache_dir=tmp_path)
        listings = await client.shopping("Sapiens Harari book", IN)
    assert listings == [] and client.failed_searches == 1
    assert len(attempts) == net.RETRIES + 1  # retried before giving up


async def _no_sleep(_):
    return None


def test_short_titles_need_the_whole_phrase():
    assert relevant("Deep Work: Rules for Focused Success (Paperback)", "Deep Work", "Cal Newport")
    assert not relevant("JDM Deep rim WORK Schwert SC4 245/40R20", "Deep Work", "Cal Newport")
    assert not relevant("Rare Luxus Ruhla Cal. UMF Working Men's Wristwatch, Deep", "Deep Work", "Cal Newport")


def test_collectible_lots_and_other_formats_are_not_like_kind():
    title, author = "Long Walk to Freedom", "Nelson Mandela"
    assert relevant("Long Walk To Freedom: The Autobiography of Nelson Mandela", title, author)
    for listing in ("Long Walk To Freedom, Mandela, Nelson, Signed First Edition, 1994",
                    "Long Walk to Freedom, the South African first edition, inscribed and...",
                    "Long Walk To Freedom Vol 1 - Audiobook", "Long Walk to Freedom (Kindle Edition)",
                    "Lot of 3 books: Long Walk to Freedom, Invictus"):
        assert not relevant(listing, title, author), listing


def test_exclusion_words_in_the_books_own_title_do_not_reject_it():
    assert relevant("The Cambridge Companion to Orwell (Paperback)", "The Cambridge Companion to Orwell", "")


async def test_used_listings_above_replacement_are_collectible_copies():
    client = FakeClient([
        L("Nineteen Eighty-Four", 300), L("Nineteen Eighty-Four", 350, merchant="Flipkart"),
        L("Nineteen Eighty-Four", 180, condition="used", merchant="Bookchor"),
        L("Nineteen Eighty-Four", 404395, condition="used", merchant="Biblio.com"),
        L("Nineteen Eighty-Four", 95000, condition="used", merchant="Peter Harrington"),
    ])
    result = await _price(client)
    assert result.used.amount == 180 and "median of 1" in result.used.basis
    assert result.used.amount <= result.replacement.amount


async def test_converted_used_value_is_bounded_by_replacement():
    client = FakeClient([L("Nineteen Eighty-Four", 250)],
                        ebay=[L("Nineteen Eighty-Four Orwell", 2430.0, "USD", "used", "eBay")])
    result = await _price(client)
    assert result.used.amount is None
    assert any("at or below the replacement" in n for n in result.notes)


async def test_no_used_value_without_a_replacement_to_bound_it():
    client = FakeClient([L("Nineteen Eighty-Four", 120, condition="used")],
                        ebay=[L("Nineteen Eighty-Four Orwell", 4.0, "USD", "used", "eBay")])
    result = await _price(client)
    assert result.replacement.amount is None and result.used.amount is None


async def test_listings_without_a_link_are_not_evidence():
    linkless = Listing("The Selfish Gene", 14, "INR", "", "", "new")
    result = await _price(FakeClient([linkless, linkless]), title="The Selfish Gene", author="Richard Dawkins")
    assert result.replacement.amount is None


def test_same_title_by_someone_else_or_in_translation_is_another_book():
    title, author = "The Midnight Library", "Matt Haig"
    assert relevant("The Midnight Library (Special Hardcover Edition with Sprayed edges)", title, author)
    assert relevant("The Midnight Library: A Novel: A GMA Book Club Pick", title, author)
    assert relevant("Midnight Library by Matt Haig, Paperback", title, author)
    for listing in ("The Midnight Library by Kazuno Kohara", "Tales from the Midnight Library: I Can See You",
                    "The Midnight Library (Malayalam)", "La biblioteca de la medianoche / The Midnight Library",
                    "The Midnight Library & Reasons to Stay Alive By Matt Haig 2 Books Collection Set",
                    "The Midnight Library For The Coolest Stories!",
                    "The Midnight Library by Matt Haig Leather Bound Hardcover Book",
                    "The Midnight Library by Matt Haig Book Poster, Contemporary Fiction"):
        assert not relevant(listing, title, author), listing
    assert relevant("Rich Dad, Poor Dad: What the Rich Teach Their Kids", "Rich Dad, Poor Dad", "Robert Kiyosaki")
