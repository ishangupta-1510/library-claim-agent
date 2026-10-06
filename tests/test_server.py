"""Server routes that need no live Gemini session."""

from fastapi.testclient import TestClient

from library_claim.server import app


def test_demo_camera_lists_the_synthetic_sweep_with_its_marker_size():
    client = TestClient(app)
    manifest = client.get("/api/demo").json()
    assert manifest["frames"] and manifest["marker_cm"] == 10.0
    first = client.get(manifest["frames"][0])
    assert first.status_code == 200 and first.headers["content-type"] == "image/jpeg"


def test_app_shell_is_revalidated_so_updates_reach_the_browser():
    client = TestClient(app)
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"


def test_a_country_is_set_only_from_words_that_name_it():
    from library_claim.config import confirms_country

    assert confirms_country("IN", "I'm in India") and confirms_country("IN", "rupees please")
    assert confirms_country("US", "the US") and confirms_country("GB", "UK")
    assert not confirms_country("US", "Sh")
    assert not confirms_country("US", "use it") and not confirms_country("IN", "")


def test_the_pronoun_us_does_not_confirm_the_usa():
    from library_claim.config import confirms_country

    assert not confirms_country("US", "can you help us") and confirms_country("US", "We're in the U.S.")


def test_mock_flow_needs_no_keys_and_confirms_the_country(monkeypatch):
    import library_claim.server as server
    from dataclasses import replace

    monkeypatch.setattr(server, "cfg", replace(server.cfg, google_api_key="", serpapi_key=""))
    client = TestClient(app)
    with client.websocket_connect("/ws/sweep?mock=1") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["mock"]
        greeting = ws.receive_json()
        assert greeting["speaker"] == "agent" and "country" in greeting["text"]
        ws.receive_json()  # turn_complete
        ws.send_json({"type": "text", "text": "I'm in India"})
        events = [ws.receive_json() for _ in range(3)]
        assert {"type": "locale", "country": "IN", "currency": "INR"} in events
