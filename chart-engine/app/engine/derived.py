"""Derived deterministic facts: dignity, combustion, chart-level counts.

Strictly factual. No interpretation, no prediction.
"""

from __future__ import annotations

from app.config.dignities import DEBILITATION, EXALTATION, MOOLATRIKONA, OWN_SIGNS
from app.config.planets import COMBUSTION_ORB
from app.utils.angles import angular_separation


def dignity_of(body: str, sign: str, degree_in_sign: float) -> dict:
    """exalted | debilitated | moolatrikona | own_sign | neutral."""
    status = "neutral"
    if body in EXALTATION and EXALTATION[body]["sign"] == sign:
        status = "exalted"
    elif body in DEBILITATION and DEBILITATION[body]["sign"] == sign:
        status = "debilitated"
    elif body in MOOLATRIKONA:
        mt_sign, start, end = MOOLATRIKONA[body]
        if mt_sign == sign and start <= degree_in_sign < end:
            status = "moolatrikona"
    if status == "neutral" and sign in OWN_SIGNS.get(body, []):
        status = "own_sign"

    detail = {"status": status}
    if body in EXALTATION:
        detail["exaltation_sign"] = EXALTATION[body]["sign"]
        detail["debilitation_sign"] = DEBILITATION[body]["sign"]
    return detail


def annotate_dignity(planets: list[dict]) -> None:
    for planet in planets:
        planet["dignity"] = dignity_of(
            planet["body"],
            planet["position"]["zodiac_sign"],
            planet["position"]["degree_in_sign"],
        )


def annotate_combustion(planets: list[dict]) -> None:
    """Distance from the Sun against configured orbs."""
    sun = next((p for p in planets if p["body"] == "Sun"), None)
    if sun is None:
        return
    sun_lon = sun["position"]["sidereal_longitude"]
    for planet in planets:
        body = planet["body"]
        if body not in COMBUSTION_ORB:
            planet["combustion"] = {"applicable": False}
            continue
        separation = angular_separation(sun_lon, planet["position"]["sidereal_longitude"])
        planet["combustion"] = {
            "applicable": True,
            "distance_from_sun": round(separation, 4),
            "orb_used": COMBUSTION_ORB[body],
            "combust": separation < COMBUSTION_ORB[body],
        }


def chart_summary(planets: list[dict], houses: list[dict], ascendant: dict) -> dict:
    by_body = {p["body"]: p for p in planets}
    counts = {str(h["number"]): len(h["occupants"]) for h in houses}
    moon = by_body.get("Moon")
    sun = by_body.get("Sun")
    return {
        "ascendant_sign": ascendant["sign"],
        "ascendant_lord": ascendant["sign_lord"],
        "ascendant_nakshatra": ascendant["nakshatra"]["name"],
        "moon_sign": moon["position"]["zodiac_sign"] if moon else None,
        "moon_nakshatra": moon["nakshatra"]["name"] if moon else None,
        "moon_pada": moon["nakshatra"]["pada"] if moon else None,
        "sun_sign": sun["position"]["zodiac_sign"] if sun else None,
        "retrograde_planets": [p["body"] for p in planets if p["motion"]["retrograde"]],
        "combust_planets": [
            p["body"] for p in planets if p.get("combustion", {}).get("combust")
        ],
        "planet_count_by_house": counts,
        "empty_houses": [h["number"] for h in houses if not h["occupants"]],
    }
