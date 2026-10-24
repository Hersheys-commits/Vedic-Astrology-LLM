"""
Thin accessors over the chart-engine's JSON shape (see the sample response
in the chart-engine project). Kept in one place so that if the chart-engine's
schema grows/changes, only this file needs updating.
"""
from __future__ import annotations

from typing import Any, Optional


def get_planet(chart: dict[str, Any], planet: str) -> Optional[dict[str, Any]]:
    for p in chart.get("planets", []):
        if p.get("body", "").lower() == planet.lower():
            return p
    return None


def planet_house_sign_nakshatra(chart: dict[str, Any], planet: str) -> Optional[dict[str, Any]]:
    p = get_planet(chart, planet)
    if not p:
        return None
    return {
        "planet": p.get("body"),
        "house": p.get("placement", {}).get("house"),
        "sign": p.get("position", {}).get("zodiac_sign"),
        "nakshatra": p.get("nakshatra", {}).get("name"),
        "pada": p.get("nakshatra", {}).get("pada"),
        "retrograde": p.get("motion", {}).get("retrograde", False),
    }


def get_house(chart: dict[str, Any], house_number: int) -> Optional[dict[str, Any]]:
    for h in chart.get("houses", []):
        if h.get("number") == house_number:
            return h
    return None


def house_summary(chart: dict[str, Any], house_number: int) -> Optional[dict[str, Any]]:
    h = get_house(chart, house_number)
    if not h:
        return None
    return {
        "house": house_number,
        "sign": h.get("sign"),
        "lord": h.get("lord"),
        "occupants": h.get("occupants", []),
        "lord_placement_house": h.get("lord_placement", {}).get("house"),
        "lord_placement_sign": h.get("lord_placement", {}).get("sign"),
    }


def current_dasha(chart: dict[str, Any]) -> Optional[dict[str, Any]]:
    cur = chart.get("dashas", {}).get("current_period")
    if not cur:
        return None
    return {"mahadasha": cur.get("mahadasha"), "antardasha": cur.get("antardasha")}


def chart_summary(chart: dict[str, Any]) -> dict[str, Any]:
    return chart.get("chart_summary", {})


def natural_language_facts(chart: dict[str, Any]) -> list[str]:
    return chart.get("natural_language_facts", [])


def aspects_from(chart: dict[str, Any], planet: str) -> list[dict[str, Any]]:
    out = []
    for a in chart.get("aspects", []):
        if a.get("relationship") == "aspect" and a.get("from", "").lower() == planet.lower():
            out.append(a)
    return out
    

def house_facts(chart: dict[str, Any], house: int) -> list[dict[str, Any]]:
    """
    "User asks about house N" -> house_lookup (context only), planet_house for each
    occupant, and planet_house for the house LORD at its placed house (tagged with a
    `role`). The lord fact is what makes empty houses still retrievable.
    """
    hs = house_summary(chart, house)
    if not hs:
        return []
    facts: list[dict[str, Any]] = [
        {
            "type": "house_lookup",
            "house": house,
            "sign": hs["sign"],
            "lord": hs.get("lord"),
            "occupants": list(hs.get("occupants") or []),
        }
    ]
    for occupant in hs["occupants"]:
        occupant_data = planet_house_sign_nakshatra(chart, occupant)
        occupant_fact: dict[str, Any] = {
            "type": "planet_house",
            "planet": occupant,
            "house": house,
        }
        sign = (occupant_data or {}).get("sign") or hs.get("sign")
        if sign:
            occupant_fact["sign"] = sign
        facts.append(occupant_fact)
    if hs.get("lord") and hs.get("lord_placement_house"):
        lord_data = planet_house_sign_nakshatra(chart, hs["lord"])
        lord_house_summary = house_summary(chart, hs["lord_placement_house"])
        lord_sign = (
            (lord_data or {}).get("sign")
            or hs.get("lord_placement_sign")
            or (lord_house_summary or {}).get("sign")
        )
        lord_fact: dict[str, Any] = {
            "type": "planet_house",
            "planet": hs["lord"],
            "house": hs["lord_placement_house"],
            "role": f"lord of house {house}",
        }
        if lord_sign:
            lord_fact["sign"] = lord_sign
        facts.append(lord_fact)
    return facts