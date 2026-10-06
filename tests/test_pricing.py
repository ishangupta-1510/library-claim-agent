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
    client = FakeClient([L("Nineteen Eighty-Four", 250)], ebay=[L("Nineteen Eighty-Four Orwell", 4.0, "USD", "used", "eBay")])
    result = await _price(client)
    used = result.used
    assert used.converted and used.original_currency == "USD" and used.currency == "INR"
    assert used.amount == pytest.approx(4.0 * 96.3)
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
    assert relevant("1984 / Nineteen Eighty-Four (Penguin Modern Classics)", "Nineteen Eighty-Four", "Orwell")


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
