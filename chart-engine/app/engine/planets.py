"""Planetary positions with Vedic semantics attached."""

from __future__ import annotations

from app.astronomy.swisseph import calc_body, node_body_id, set_sidereal_mode
from app.config.planets import BODIES, OPTIONAL_BODIES
from app.config.signs import SIGN_ELEMENT, SIGN_LORDS, SIGN_MODALITY, sign_name
from app.engine.nakshatra import nakshatra_of
from app.utils.angles import degree_in_sign, norm360, sign_index_of
from app.utils.formatting import format_dms


def _position_block(longitude: float) -> dict:
    lon = norm360(longitude)
    idx = sign_index_of(lon)
    sign = sign_name(idx)
    deg = degree_in_sign(lon)
    return {
        "sidereal_longitude": round(lon, 6),
        "zodiac_sign": sign,
        "sign_index": idx,
        "sign_lord": SIGN_LORDS[sign],
        "sign_element": SIGN_ELEMENT[sign],
        "sign_modality": SIGN_MODALITY[sign],
        "degree_in_sign": round(deg, 6),
        "formatted_degree": format_dms(deg),
    }


def calculate_planets(jd_ut: float, settings) -> list[dict]:
    """Compute every body in the configured body list.

    Ketu is derived as Rahu + 180 degrees (it is a mathematical point, not an
    object the ephemeris tracks).
    """
    sidereal = settings.zodiac == "sidereal"
    if sidereal:
        set_sidereal_mode(settings.ayanamsa)

    bodies = list(BODIES)
    if settings.include_outer_planets:
        bodies = bodies + OPTIONAL_BODIES

    node_id = node_body_id(settings.node_type)
    rahu_raw = calc_body(jd_ut, node_id, sidereal)

    planets: list[dict] = []
    for body in bodies:
        name = body["name"]
        if name == "Rahu":
            raw = dict(rahu_raw)
        elif name == "Ketu":
            raw = dict(rahu_raw)
            raw["longitude"] = norm360(rahu_raw["longitude"] + 180.0)
            raw["latitude"] = -rahu_raw["latitude"]
        else:
            raw = calc_body(jd_ut, body["swe_id"], sidereal)

        speed = raw["speed_longitude"]
        # The lunar nodes are always retrograde in mean motion; Swiss Ephemeris
        # reports a small negative speed for the mean node.
        retrograde = speed < 0

        planets.append(
            {
                "body": name,
                "kind": body["kind"],
                "position": _position_block(raw["longitude"]),
                "nakshatra": nakshatra_of(raw["longitude"]),
                "motion": {
                    "retrograde": bool(retrograde),
                    "daily_longitude_speed": round(speed, 6),
                    "stationary": abs(speed) < 0.0005,
                },
                "raw": {
                    "ecliptic_longitude": round(norm360(raw["longitude"]), 8),
                    "ecliptic_latitude": round(raw["latitude"], 8),
                    "distance_au": round(raw["distance_au"], 8),
                    "speed_longitude": round(raw["speed_longitude"], 8),
                },
            }
        )
    return planets
