from app.astronomy.julian import utc_to_julian_day
from app.config.settings import ChartSettings
from app.engine.planets import calculate_planets
from datetime import datetime, timezone

JD = utc_to_julian_day(datetime(2004, 5, 17, 9, 5, tzinfo=timezone.utc))


def test_nine_bodies_by_default():
    planets = calculate_planets(JD, ChartSettings())
    assert [p["body"] for p in planets] == [
        "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"
    ]


def test_nodes_are_exactly_opposed():
    planets = {p["body"]: p for p in calculate_planets(JD, ChartSettings())}
    rahu = planets["Rahu"]["position"]["sidereal_longitude"]
    ketu = planets["Ketu"]["position"]["sidereal_longitude"]
    assert abs(((ketu - rahu) % 360.0) - 180.0) < 1e-6


def test_sidereal_differs_from_tropical_by_ayanamsa():
    from app.astronomy.swisseph import ayanamsa_value

    sidereal = calculate_planets(JD, ChartSettings(zodiac="sidereal"))
    tropical = calculate_planets(JD, ChartSettings(zodiac="tropical"))
    sun_sid = sidereal[0]["position"]["sidereal_longitude"]
    sun_trop = tropical[0]["position"]["sidereal_longitude"]
    assert abs(((sun_trop - sun_sid) % 360.0) - ayanamsa_value(JD)) < 0.01


def test_longitudes_normalised():
    for p in calculate_planets(JD, ChartSettings()):
        assert 0.0 <= p["position"]["sidereal_longitude"] < 360.0
        assert 0.0 <= p["position"]["degree_in_sign"] < 30.0


def test_determinism():
    a = calculate_planets(JD, ChartSettings())
    b = calculate_planets(JD, ChartSettings())
    assert a == b
