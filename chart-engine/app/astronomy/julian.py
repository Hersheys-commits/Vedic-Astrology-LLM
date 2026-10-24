"""UTC -> Julian Day (UT). Deterministic, no side effects."""

from __future__ import annotations

from datetime import datetime

import swisseph as swe

from app.utils.errors import ChartError


def utc_to_julian_day(utc_dt: datetime) -> float:
    """Convert an aware UTC datetime to Julian Day in Universal Time."""
    try:
        seconds = utc_dt.second + utc_dt.microsecond / 1_000_000
        jd_et, jd_ut = swe.utc_to_jd(
            utc_dt.year, utc_dt.month, utc_dt.day,
            utc_dt.hour, utc_dt.minute, seconds,
            swe.GREG_CAL,
        )
        return jd_ut
    except Exception as exc:  # pragma: no cover - defensive
        raise ChartError("CALCULATION_FAILED", f"Julian day conversion failed: {exc}")


def julian_day_to_utc_string(jd_ut: float) -> str:
    """Inverse conversion, used by tests and the dasha timeline."""
    y, m, d, hour = swe.revjul(jd_ut, swe.GREG_CAL)
    h = int(hour)
    minute_float = (hour - h) * 60
    mi = int(minute_float)
    s = int(round((minute_float - mi) * 60))
    if s == 60:
        s, mi = 0, mi + 1
    if mi == 60:
        mi, h = 0, h + 1
    return f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}Z"
