from library_claim.config import locale_for
from library_claim.stages.item_pricing import price_item
from library_claim.stages.pricing import Listing

IN = locale_for("IN")


class Shop:
    def __init__(self, by_query):
        self.by_query = by_query
        self.queries = []

    async def shopping(self, query, locale):
        self.queries.append(query)
        return self.by_query.get(query, [])


def L(amount, merchant="Amazon.in"):
    return Listing("x", amount, "INR", merchant, f"https://shop/{amount}", "new")


async def _item(shop, **kw):
    args = dict(category="lighting", description="floor lamp", material="metal", brand_model="", is_artwork=False,
                user_says_print=False, locale=IN, client=shop)
    args.update(kw)
    return await price_item(**args)


async def test_unknown_brand_gives_sourced_interquartile_range():
    shop = Shop({"metal floor lamp": [L(1500), L(2000), L(2500), L(3000), L(3500), L(90000)]})
    result = await _item(shop)
    assert result.status == "range"
    assert 1500 <= result.price.low < result.price.high <= 3500  # the 90000 outlier is trimmed
    assert result.price.url and result.price.retrieved_at and "interquartile" in result.price.basis


async def test_legible_model_is_priced_at_median():
    shop = Shop({"Philips HD7431": [L(3200), L(3400), L(3600)]})
    result = await _item(shop, category="appliance", description="coffee machine", brand_model="Philips HD7431")
    assert result.status == "priced" and result.price.low == result.price.high == 3400


async def test_art_needs_appraisal_without_asking_the_market():
    shop = Shop({})
    result = await _item(shop, category="art", description="portrait in gilt frame", is_artwork=True)
    assert result.status == "needs_appraisal" and shop.queries == []


async def test_art_confirmed_as_print_is_priced_as_a_print():
    shop = Shop({"paper framed portrait print": [L(800), L(1200)]})
    result = await _item(shop, category="art", description="framed portrait", material="paper", is_artwork=True, user_says_print=True)
    assert result.status == "range" and result.price.low == 800


async def test_no_listings_leaves_price_empty():
    result = await _item(Shop({}))
    assert result.price.low is None and result.notes
