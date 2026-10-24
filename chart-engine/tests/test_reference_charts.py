"""Replay every frozen fixture in tests/fixtures/ and fail on drift."""

import json
from pathlib import Path

import pytest

from app.engine.calculator import generate_chart

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))
TOLERANCE_DEG = 1 / 3600  # 1 arc-second


@pytest.mark.skipif(not FIXTURES, reason="no reference fixtures generated yet")
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_reference_chart(path):
    fixture = json.loads(path.read_text())
    inp, expected = fixture["input"], fixture["expected"]
    chart = generate_chart(
        birth_date=inp["birth_date"], birth_time=inp["birth_time"],
        latitude=inp["latitude"], longitude=inp["longitude"],
        timezone_id=inp.get("timezone"), as_of_date="2026-01-01",
    )
    assert chart["time"]["utc_datetime"] == expected["utc_datetime"]
    assert chart["angles"]["ascendant"]["sign"] == expected["ascendant_sign"]
    assert abs(chart["angles"]["ascendant"]["longitude"] - expected["ascendant_longitude"]) < TOLERANCE_DEG

    computed = {p["body"]: p for p in chart["planets"]}
    for body, want in expected["planets"].items():
        got = computed[body]
        assert abs(got["position"]["sidereal_longitude"] - want["longitude"]) < TOLERANCE_DEG, body
        assert got["position"]["zodiac_sign"] == want["sign"], body
        assert got["placement"]["house"] == want["house"], body
        assert got["nakshatra"]["name"] == want["nakshatra"], body
        assert got["nakshatra"]["pada"] == want["pada"], body
