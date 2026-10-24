"""Ascendant (Lagna) and Midheaven."""

from __future__ import annotations

from app.astronomy.swisseph import calc_houses
from app.config.signs import SIGN_LORDS, sign_name
from app.engine.nakshatra import nakshatra_of
from app.utils.angles import degree_in_sign, norm360, sign_index_of
from app.utils.formatting import format_dms


def _angle_object(longitude: float, with_nakshatra: bool = True) -> dict:
    lon = norm360(longitude)
    idx = sign_index_of(lon)
    deg = degree_in_sign(lon)
    obj = {
        "longitude": round(lon, 6),
        "sign": sign_name(idx),
        "sign_index": idx,
        "sign_lord": SIGN_LORDS[sign_name(idx)],
        "degree_in_sign": round(deg, 6),
        "formatted": format_dms(deg),
    }
    if with_nakshatra:
        obj["nakshatra"] = nakshatra_of(lon)
    return obj


def calculate_angles(jd_ut: float, latitude: float, longitude: float, settings) -> dict:
    """Returns ascendant, midheaven and the raw cusp array."""
    sidereal = settings.zodiac == "sidereal"
    houses = calc_houses(jd_ut, latitude, longitude, settings.house_system, sidereal)
    return {
        "ascendant": _angle_object(houses["ascendant"]),
        "midheaven": _angle_object(houses["midheaven"], with_nakshatra=False),
        "_cusps": [round(norm360(c), 6) for c in houses["cusps"]],
    }
