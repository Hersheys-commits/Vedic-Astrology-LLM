from datetime import datetime, timezone

from app.astronomy.julian import julian_day_to_utc_string, utc_to_julian_day


def test_j2000_epoch():
    jd = utc_to_julian_day(datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    assert abs(jd - 2451545.0) < 1e-4


def test_roundtrip():
    dt = datetime(2004, 5, 17, 9, 5, 0, tzinfo=timezone.utc)
    jd = utc_to_julian_day(dt)
    assert julian_day_to_utc_string(jd).startswith("2004-05-17T09:05")


def test_determinism():
    dt = datetime(1995, 8, 12, 3, 14, 15, tzinfo=timezone.utc)
    assert utc_to_julian_day(dt) == utc_to_julian_day(dt)
