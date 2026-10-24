from fastapi.testclient import TestClient

from app.api import location
from app.main import app
from app.models.requests import AskRequest


def test_chat_ui_and_assets_are_served_by_fastapi():
    client = TestClient(app)
    page = client.get("/")
    script = client.get("/app.js")
    styles = client.get("/styles.css")

    assert page.status_code == 200
    assert "Vedic astrology assistant" in page.text
    assert "advanced-settings" in page.text
    assert 'id="response-model"' in page.text
    for chunk_count in range(1, 5):
        assert f"<option>{chunk_count}</option>" in page.text
    assert "<option selected>5</option>" in page.text
    assert 'value="gpt-5-mini"' in page.text
    assert 'value="astro-qwen-lora"' in page.text
    assert 'value="astro-qwen-npu"' in page.text
    assert script.status_code == 200 and "async function sendQuestion" in script.text
    assert '"astro-qwen-npu"' in script.text
    assert 'row.className = "message-row message-assistant"' in script.text
    assert "setInterval(() => {" in script.text
    assert "Still processing your question" in script.text
    assert 'fetch("/ask/stream"' in script.text
    assert '"text/event-stream"' in script.text
    assert 'fetch("/ask"' in script.text
    assert 'eventType === "progress"' in script.text
    assert 'eventType === "delta"' in script.text
    assert 'eventType === "replace"' in script.text
    assert "await handleFrame(frame)" in script.text
    assert "requestAnimationFrame(() => setTimeout(resolve, 0))" in script.text
    assert 'response_model: "gpt-5-mini"' in script.text
    assert '"#response-model"' in script.text
    assert "localStorage.setItem(STORAGE_KEY, JSON.stringify(state))" in script.text
    assert "response_model: responseModel" in script.text
    assert "message.response_model || state.settings.response_model" in script.text
    assert styles.status_code == 200 and ".profile-dialog" in styles.text
    assert ".progress-spinner" in styles.text


def test_chat_request_shape_remains_compatible_with_ask_api():
    request = AskRequest.model_validate({
        "birth_date": "2004-05-17",
        "birth_time": "14:35",
        "location": {
            "mode": "geocoded",
            "latitude": 25.4358,
            "longitude": 81.8463,
            "display_name": "Prayagraj, India",
        },
        "settings": {"ayanamsa": "lahiri", "house_system": "whole_sign", "node_type": "mean"},
        "question": "What themes are associated with my career?",
        "strategy": "hybrid",
        "top_k": 5,
    })

    assert request.location.latitude == 25.4358
    assert request.question == "What themes are associated with my career?"
    assert request.settings.house_system == "whole_sign"
    assert request.top_k == 5
    assert request.response_model == "gpt-5-mini"


def test_ask_request_accepts_each_available_source_passage_count():
    for chunk_count in (1, 2, 3, 4, 5, 7, 10):
        request = AskRequest.model_validate({
            "chart_json": {},
            "question": "What does my chart suggest?",
            "top_k": chunk_count,
        })
        assert request.top_k == chunk_count


def test_same_origin_geocode_route_returns_chart_engine_candidates(monkeypatch):
    location.clear_geocode_cache()
    monkeypatch.setattr(location, "_next_upstream_request_at", 0.0)
    monkeypatch.setattr(location.requests, "post", lambda *args, **kwargs: type("Response", (), {
        "status_code": 200,
        "json": staticmethod(lambda: {"results": [{
            "display_name": "Delhi, India", "latitude": 28.6139, "longitude": 77.2090,
        }]})
    })())
    response = TestClient(app).post("/location/geocode", json={"query": "Delhi", "limit": 3})

    assert response.status_code == 200
    assert response.json()["results"][0]["display_name"] == "Delhi, India"
    assert response.json()["attribution"] == "Geocoding data © OpenStreetMap contributors"
