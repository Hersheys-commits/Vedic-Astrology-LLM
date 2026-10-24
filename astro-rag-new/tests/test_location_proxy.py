from __future__ import annotations

import requests
import pytest

from app.api import location
from app.models.requests import GeocodeRequest
from app.utils.errors import RAGError


@pytest.fixture(autouse=True)
def clear_cache():
    location.clear_geocode_cache()
    yield
    location.clear_geocode_cache()


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"results": []}

    def json(self):
        return self._payload


def test_geocode_forwards_search_and_caches_normalized_repeats(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse(200, {"results": [{
            "display_name": "Prayagraj, Uttar Pradesh, India",
            "latitude": 25.4358,
            "longitude": 81.8463,
            "type": "city",
        }]})

    monkeypatch.setattr(location.requests, "post", fake_post)
    first = location.geocode(GeocodeRequest(query="Prayagraj", limit=5))
    second = location.geocode(GeocodeRequest(query="  PRAYAGRAJ  ", limit=5))

    assert first.results[0].latitude == 25.4358
    assert second.results[0].display_name == "Prayagraj, Uttar Pradesh, India"
    assert first.attribution == "Geocoding data © OpenStreetMap contributors"
    assert len(calls) == 1
    assert calls[0][1] == {"query": "Prayagraj", "limit": 5}


def test_geocode_cache_key_includes_result_limit(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json["limit"])
        return FakeResponse(200, {"results": [{
            "display_name": "Delhi, India", "latitude": 28.6, "longitude": 77.2,
        }]})

    monkeypatch.setattr(location.requests, "post", fake_post)
    location.geocode(GeocodeRequest(query="Delhi", limit=3))
    location.geocode(GeocodeRequest(query="Delhi", limit=5))
    assert calls == [3, 5]


def test_geocode_reports_unavailable_upstream(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(location.requests, "post", fake_post)
    with pytest.raises(RAGError) as exc:
        location.geocode(GeocodeRequest(query="Delhi"))
    assert exc.value.code == "LOCATION_LOOKUP_UNAVAILABLE"
    assert exc.value.status_code == 502


def test_geocode_surfaces_empty_search_as_not_found(monkeypatch):
    monkeypatch.setattr(location.requests, "post", lambda *args, **kwargs: FakeResponse(200, {"results": []}))
    with pytest.raises(RAGError) as exc:
        location.geocode(GeocodeRequest(query="A place that does not exist"))
    assert exc.value.code == "LOCATION_NOT_FOUND"
    assert exc.value.status_code == 404


def test_geocode_rejects_malformed_upstream_response(monkeypatch):
    monkeypatch.setattr(location.requests, "post", lambda *args, **kwargs: FakeResponse(200, {"results": [{"lat": 12}]}))
    with pytest.raises(RAGError) as exc:
        location.geocode(GeocodeRequest(query="Somewhere"))
    assert exc.value.code == "LOCATION_LOOKUP_BAD_RESPONSE"
    assert exc.value.status_code == 502
