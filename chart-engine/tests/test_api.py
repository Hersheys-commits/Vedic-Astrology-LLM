from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chart_endpoint():
    r = client.post(
        "/chart",
        json={
            "birth_date": "2004-05-17",
            "birth_time": "14:35:00",
            "location": {"mode": "map_click", "latitude": 25.4358, "longitude": 81.8463},
            "as_of_date": "2026-01-01",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["validation"]["status"] == "passed"
    assert len(body["planets"]) == 9


def test_error_contract():
    r = client.post(
        "/chart",
        json={
            "birth_date": "2004-13-45",
            "birth_time": "14:35:00",
            "location": {"latitude": 25.4, "longitude": 81.8},
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_DATE"
