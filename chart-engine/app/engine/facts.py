"""Retrieval-ready fact keys, fact ids and mechanical natural-language facts.

Everything here is generated from structured values by deterministic templates
so the RAG layer can index chart facts without re-deriving relationships.
"""

from __future__ import annotations

from app.utils.formatting import slug


def build_fact_keys(planets: list[dict], houses: list[dict], ascendant: dict, dashas: dict) -> list[str]:
    keys: list[str] = []

    asc_sign = slug(ascendant["sign"])
    keys.append(f"ascendant|sign:{asc_sign}")
    keys.append(f"ascendant|nakshatra:{slug(ascendant['nakshatra']['name'])}")
    keys.append(f"ascendant|lord:{slug(ascendant['sign_lord'])}")

    for p in planets:
        body = slug(p["body"])
        sign = slug(p["position"]["zodiac_sign"])
        house = p["placement"]["house"]
        nak = slug(p["nakshatra"]["name"])
        pada = p["nakshatra"]["pada"]
        keys += [
            f"planet:{body}",
            f"planet:{body}|sign:{sign}",
            f"planet:{body}|house:{house}",
            f"planet:{body}|sign:{sign}|house:{house}",
            f"planet:{body}|nakshatra:{nak}",
            f"planet:{body}|nakshatra:{nak}|pada:{pada}",
        ]
        if p["motion"]["retrograde"]:
            keys.append(f"planet:{body}|state:retrograde")
        status = p.get("dignity", {}).get("status")
        if status and status != "neutral":
            keys.append(f"planet:{body}|dignity:{status}")
        if p.get("combustion", {}).get("combust"):
            keys.append(f"planet:{body}|state:combust")

    for h in houses:
        keys.append(f"house:{h['number']}|sign:{slug(h['sign'])}")
        keys.append(f"house:{h['number']}|lord:{slug(h['lord'])}")
        for occupant in h["occupants"]:
            keys.append(f"house:{h['number']}|occupant:{slug(occupant)}")
        if h.get("lord_placement"):
            keys.append(f"house:{h['number']}|lord_in_house:{h['lord_placement']['house']}")

    keys.append(f"dasha:mahadasha|lord:{slug(dashas['birth_mahadasha']['lord'])}")

    # Deterministic order, no duplicates.
    return sorted(set(keys))


def build_fact_ids(planets: list[dict], houses: list[dict], aspects: list[dict], dashas: dict) -> list[str]:
    ids: list[str] = []
    for p in planets:
        body = slug(p["body"])
        ids.append(f"planet.{body}.sign.{slug(p['position']['zodiac_sign'])}")
        ids.append(f"planet.{body}.house.{p['placement']['house']}")
        ids.append(f"planet.{body}.nakshatra.{slug(p['nakshatra']['name'])}")
        ids.append(f"planet.{body}.nakshatra.{slug(p['nakshatra']['name'])}.pada.{p['nakshatra']['pada']}")
    for h in houses:
        ids.append(f"house.{h['number']}.sign.{slug(h['sign'])}")
        ids.append(f"house.{h['number']}.lord.{slug(h['lord'])}")
    for a in aspects:
        if a["relationship"] == "aspect":
            ids.append(f"aspect.{slug(a['from'])}.{slug(a['to'])}.{slug(a['aspect_type'])}")
        else:
            ids.append(f"conjunction.{slug(a['from'])}.{slug(a['to'])}")
    ids.append(f"dasha.mahadasha.{slug(dashas['birth_mahadasha']['lord'])}")
    return sorted(set(ids))


def build_natural_language_facts(
    planets: list[dict], houses: list[dict], ascendant: dict, aspects: list[dict], dashas: dict
) -> list[str]:
    """Plain factual sentences generated from templates. No interpretation."""
    lines = [
        f"The ascendant (lagna) is {ascendant['sign']} at {ascendant['formatted']}.",
        f"The ascendant lord is {ascendant['sign_lord']}.",
        f"The ascendant falls in {ascendant['nakshatra']['name']} nakshatra, pada {ascendant['nakshatra']['pada']}.",
    ]
    for p in planets:
        b = p["body"]
        pos, place, nak = p["position"], p["placement"], p["nakshatra"]
        lines.append(f"{b} is in {pos['zodiac_sign']} at {pos['formatted_degree']}.")
        lines.append(f"{b} occupies house {place['house']}, whose lord is {place['house_lord']}.")
        lines.append(f"{b} is in {nak['name']} nakshatra, pada {nak['pada']}, ruled by {nak['lord']}.")
        if p["motion"]["retrograde"]:
            lines.append(f"{b} is retrograde.")
        status = p.get("dignity", {}).get("status")
        if status and status != "neutral":
            lines.append(f"{b} is {status.replace('_', ' ')} in {pos['zodiac_sign']}.")
        if p.get("combustion", {}).get("combust"):
            lines.append(f"{b} is combust, {p['combustion']['distance_from_sun']} degrees from the Sun.")
    for h in houses:
        occupants = ", ".join(h["occupants"]) if h["occupants"] else "no planets"
        lines.append(f"House {h['number']} is {h['sign']}, ruled by {h['lord']}, and contains {occupants}.")
        if h.get("lord_placement"):
            lines.append(
                f"The lord of house {h['number']} ({h['lord']}) is placed in house {h['lord_placement']['house']}."
            )
    for a in aspects:
        if a["relationship"] == "aspect":
            lines.append(f"{a['from']} casts its {a['aspect_type']} aspect on {a['to']}.")
        else:
            lines.append(f"{a['from']} is conjunct {a['to']} in the same sign.")
    bm = dashas["birth_mahadasha"]
    lines.append(
        f"At birth the running Vimshottari mahadasha is {bm['lord']} with "
        f"{bm['balance_years']:.2f} years remaining, ending {bm['ends']}."
    )
    return lines
