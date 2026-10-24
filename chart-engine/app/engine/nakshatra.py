"""Nakshatra and pada from a sidereal longitude."""

from __future__ import annotations

from app.config.nakshatras import NAKSHATRA_SPAN, NAKSHATRAS, PADA_SPAN
from app.utils.angles import norm360


def nakshatra_of(longitude: float) -> dict:
    """Sidereal longitude -> nakshatra name, index (1-27), pada (1-4), lord."""
    lon = norm360(longitude)
    idx0 = int(lon // NAKSHATRA_SPAN) % 27          # 0-based
    degree_in_nakshatra = lon - idx0 * NAKSHATRA_SPAN
    pada = int(degree_in_nakshatra // PADA_SPAN) + 1
    pada = min(pada, 4)
    data = NAKSHATRAS[idx0]
    return {
        "name": data["name"],
        "index": data["index"],
        "lord": data["lord"],
        "pada": pada,
        "degree_in_nakshatra": round(degree_in_nakshatra, 6),
        "fraction_traversed": round(degree_in_nakshatra / NAKSHATRA_SPAN, 9),
    }
