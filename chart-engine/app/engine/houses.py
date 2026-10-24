"""Whole-sign house construction and planet-to-house placement."""

from __future__ import annotations

from app.config.signs import SIGN_LORDS, sign_name
from app.utils.angles import house_distance
from app.utils.errors import ChartError

HOUSE_KARAKA_HINT = None  # interpretation belongs to the RAG layer, not here


def build_houses(ascendant_sign_index: int, settings) -> list[dict]:
    """House 1 = ascendant sign, then one sign per house (whole-sign model)."""
    if settings.house_system != "whole_sign":
        # Equal/Placidus cusps are still returned in angles["_cusps"]; the
        # semantic house model below is only defined for whole sign so far.
        raise ChartError(
            "CONFIGURATION_ERROR",
            f"House model {settings.house_system!r} is not implemented in the semantic layer yet. "
            "Use 'whole_sign'.",
        )
    houses = []
    for i in range(12):
        idx = (ascendant_sign_index + i) % 12
        sign = sign_name(idx)
        houses.append(
            {
                "number": i + 1,
                "sign": sign,
                "sign_index": idx,
                "lord": SIGN_LORDS[sign],
                "method": "whole_sign",
                "occupants": [],
                "aspects_received": [],
            }
        )
    return houses


def place_planets_in_houses(planets: list[dict], ascendant_sign_index: int, houses: list[dict]) -> None:
    """Mutates planets and houses so both sides of the relation are explicit."""
    by_number = {h["number"]: h for h in houses}
    for planet in planets:
        sign_index = planet["position"]["sign_index"]
        number = house_distance(ascendant_sign_index, sign_index)
        house = by_number[number]
        planet["placement"] = {
            "house": number,
            "house_sign": house["sign"],
            "house_lord": house["lord"],
            "method": "whole_sign",
        }
        house["occupants"].append(planet["body"])


def annotate_house_lord_placement(houses: list[dict], planets: list[dict]) -> None:
    """For each house, record where its lord actually sits."""
    position_of = {p["body"]: p for p in planets}
    for house in houses:
        lord = house["lord"]
        lord_planet = position_of.get(lord)
        if lord_planet is None:
            house["lord_placement"] = None
            continue
        house["lord_placement"] = {
            "planet": lord,
            "house": lord_planet["placement"]["house"],
            "sign": lord_planet["position"]["zodiac_sign"],
        }
