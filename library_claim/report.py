"""Human-readable report, rendered only from the claim packet (no other inputs).

Every line links to its saved frame and its price source, so an adjuster can
trace any figure. The report adds no numbers of its own.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, select_autoescape

from .schemas import ClaimPacket

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contents claim {{ p.sweep.id }}</title>
<style>
 body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1b1b1f;max-width:1200px}
 h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
 table{border-collapse:collapse;width:100%;font-size:12.5px} th,td{border-bottom:1px solid #eee;padding:5px 6px;text-align:left;vertical-align:top}
 th{background:#f6f6f8;position:sticky;top:0} .num{text-align:right;white-space:nowrap} .muted{color:#6b6b76}
 .tag{display:inline-block;padding:1px 6px;border-radius:9px;font-size:11px;background:#eee}
 .identified{background:#e3f4e8} .unidentified{background:#f1f1f3} .needs_appraisal,.range{background:#fdf0d5} .priced{background:#e3f4e8}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
 .card{border:1px solid #e5e5ea;border-radius:8px;padding:10px} .card b{display:block;font-size:18px}
 a{color:#2456b3}
</style></head><body>
<h1>Contents claim: home library</h1>
<div class="muted">Sweep {{ p.sweep.id }} · captured {{ p.sweep.captured_at }} · {{ p.sweep.duration_s }} s · {{ p.sweep.device or "device not recorded" }} · {{ p.sweep.country }} / {{ p.sweep.currency }}</div>

<h2>Totals <span class="muted">(computed in code from the lines below)</span></h2>
<div class="cards">
 <div class="card"><span class="muted">Books counted</span><b>{{ t.book_count }}</b>{{ t.books_identified }} identified · {{ t.books_unidentified }} unidentified · {{ t.books_needs_appraisal }} for appraisal{% if t.books_excluded_by_policyholder %} · {{ t.books_excluded_by_policyholder }} excluded{% endif %}</div>
 <div class="card"><span class="muted">Books, replacement</span><b>{{ money(t.books_replacement_cost) }}</b>used value {{ money(t.books_used_value) }}{% if t.books_pending_review_cost %}<br>pending review {{ money(t.books_pending_review_cost) }}{% endif %}</div>
 <div class="card"><span class="muted">Other contents</span><b>{{ money(t.items_replacement_cost_low) }} – {{ money(t.items_replacement_cost_high) }}</b>replacement range</div>
 <div class="card"><span class="muted">Shelf run</span><b>{{ t.shelf_run_m }} m</b>summed measured spine thickness</div>
 <div class="card"><span class="muted">Lines excluded from totals</span><b>{{ t.excluded_from_totals }}</b>no sourced price, or appraisal needed</div>
</div>

<h2>Room</h2>
{% if r.floor_area_m2 is not none %}
<table><tr><th>Length</th><th>Width</th><th>Height</th><th>Floor area</th><th>Wall area (gross)</th><th>Shelved wall area</th><th>Shape</th></tr>
<tr><td>{{ r.length_m }} m</td><td>{{ r.width_m }} m</td><td>{{ r.height_m if r.height_m is not none else "—" }}{% if r.height_m is not none %} m{% endif %}</td>
<td>{{ r.floor_area_m2 }} m² ({{ r.floor_area_ft2 }} ft²)</td>
<td>{% if r.wall_area_m2 is not none %}{{ r.wall_area_m2 }} m² ({{ r.wall_area_ft2 }} ft²){% else %}—{% endif %}</td>
<td>{% if r.shelved_wall_area_m2 is not none %}{{ r.shelved_wall_area_m2 }} m² ({{ r.shelved_wall_area_ft2 }} ft²){% else %}—{% endif %}</td><td>{{ r.shape }}</td></tr></table>
{% else %}<p>Room not measured in this sweep.</p>{% endif %}
<p class="muted">Scale: {{ r.scale_method }} · confidence {{ r.confidence }}</p>

<h2>Books ({{ p.books|length }})</h2>
<table><tr><th>ID</th><th>Shelf · pos</th><th>Status</th><th>Title / author</th><th>Edition · ISBN</th><th class="num">H × T (cm)</th><th class="num">Replacement</th><th class="num">Used</th><th>Evidence</th></tr>
{% for b in p.books %}<tr>
<td>{{ b.id }}</td><td>{{ b.shelf }} · {{ b.position }}</td><td><span class="tag {{ b.status }}">{{ b.status }}</span>{% if b.excluded %} <span class="tag">excluded</span>{% endif %}</td>
<td>{% if b.title %}<b>{{ b.title }}</b><br>{{ b.author }}<br><span class="muted">conf {{ b.id_confidence }} · <a href="{{ b.id_url|safe_url }}">{{ b.id_source }}</a></span>{% else %}<span class="muted">spine: "{{ b.spine_text or "unreadable" }}"</span>{% endif %}</td>
<td>{{ b.edition }}{% if b.isbn %}<br>{{ b.isbn }}{% endif %}</td>
<td class="num">{{ dim(b.spine_height_cm) }} × {{ dim(b.spine_thickness_cm) }}</td>
<td class="num">{{ price(b.replacement_cost) }}</td><td class="num">{{ price(b.used_value) }}</td>
<td><a href="{{ b.frame_ref }}">frame</a></td></tr>{% endfor %}
</table>

<h2>Other contents ({{ p.items|length }})</h2>
<table><tr><th>ID</th><th>Category</th><th>Description</th><th>Brand / model</th><th>Size (cm)</th><th>Status</th><th class="num">Replacement</th><th>Evidence</th></tr>
{% for i in p.items %}<tr><td>{{ i.id }}</td><td>{{ i.category }}</td><td>{{ i.description }}{% if i.material %}<br><span class="muted">{{ i.material }}</span>{% endif %}</td>
<td>{{ i.brand_model or "—" }}</td><td>{% if i.dimensions_cm.w %}{{ i.dimensions_cm.w }} × {{ i.dimensions_cm.h }}{% else %}—{% endif %}</td>
<td><span class="tag {{ i.status }}">{{ i.status }}</span></td>
<td class="num">{% if i.replacement_cost.low is not none %}{{ money(i.replacement_cost.low) }}{% if i.replacement_cost.high != i.replacement_cost.low %} – {{ money(i.replacement_cost.high) }}{% endif %}<br><a href="{{ i.replacement_cost.url|safe_url }}">{{ i.replacement_cost.source }}</a> <span class="muted">{{ i.replacement_cost.retrieved_at[:10] }}</span>{% else %}—{% endif %}</td>
<td><a href="{{ i.frame_ref }}">frame</a></td></tr>{% endfor %}
</table>

<h2>Review queue ({{ p.review_queue|length }})</h2>
<table><tr><th>Line</th><th>Reason</th></tr>{% for q in p.review_queue %}<tr><td>{{ q.ref_id }}</td><td>{{ q.reason }}</td></tr>{% endfor %}</table>

{% if p.locale_comparison and p.locale_comparison.rows %}
<h2>Same books priced in {{ p.locale_comparison.country }} ({{ p.locale_comparison.currency }})</h2>
<table><tr><th>Book</th><th class="num">{{ p.sweep.country }} replacement</th><th class="num">{{ p.locale_comparison.country }} replacement</th></tr>
{% for row in p.locale_comparison.rows %}<tr><td>{{ row.book_id }} · {{ row.title }}</td><td class="num">{{ raw_price(row.home) }}</td><td class="num">{{ raw_price(row.other) }}</td></tr>{% endfor %}</table>
{% endif %}

<h2>Pipeline</h2>
<table><tr><th>Stage</th><th class="num">Seconds</th></tr>{% for k, v in p.stages.get("latency_s", {}).items() %}<tr><td>{{ k }}</td><td class="num">{{ v }}</td></tr>{% endfor %}
<tr><td><b>Time to packet after sweep</b></td><td class="num"><b>{{ p.stages.get("time_to_packet_s") }}</b></td></tr></table>
<p class="muted">Estimated cost: {{ p.stages.get("cost_usd_estimate") }}</p>
</body></html>
"""


def write_report(packet: ClaimPacket, sweep_dir: Path) -> Path:
    currency = packet.sweep.currency

    def money(value) -> str:
        return "—" if value is None else f"{currency} {value:,.0f}"

    def dim(value) -> str:
        return "—" if value is None else f"{value:.1f}"

    def safe_url(url: str) -> str:
        """Only web links and the sweep's own relative files; anything else (javascript:, data:) becomes inert."""
        url = url or ""
        return url if url.startswith(("https://", "http://", "frames/")) else "#"

    def price(p) -> str:
        if p.amount is None:
            return "—"
        label = f'{money(p.amount)}<br><a href="{p.url}">{p.source}</a> <span class="muted">{p.retrieved_at[:10]}'
        if p.converted:
            label += f" · converted from {p.original_currency} {p.original_amount}"
        if p.condition_assumed:
            label += f" · {p.condition_assumed}"
        return label + "</span>"

    def raw_price(p: dict) -> str:
        return "—" if p.get("amount") is None else f'{p["currency"]} {p["amount"]:,.0f} <a href="{p["url"]}">source</a>'

    env = Environment(autoescape=select_autoescape(default=True))
    env.globals.update(money=money, dim=dim)
    env.filters["safe_url"] = safe_url
    # price() builds trusted markup from escaped-at-source fields; mark it safe explicitly.
    from markupsafe import Markup, escape

    def safe_price(p):
        if p.amount is None:
            return "—"
        return Markup(price(p.model_copy(update={"source": str(escape(p.source)), "url": str(escape(safe_url(p.url)))})))

    def safe_raw(p):
        return Markup(raw_price({**p, "url": str(escape(safe_url(p.get("url", ""))))}))

    html = env.from_string(TEMPLATE).render(p=packet, t=packet.totals, r=packet.room, price=safe_price, raw_price=safe_raw)
    path = sweep_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
