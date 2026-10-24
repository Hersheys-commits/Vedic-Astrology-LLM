"""
Turns (user question, chart JSON) into a short list of concrete chart facts
to retrieve doctrine for -- e.g. [{"type": "planet_house", "planet": "Mars",
"house": 10}]. This is the deterministic "query / intent identification"
step called for in the architecture doc (don't hand the LLM raw JSON and a
free-text question and ask it to decide what's relevant; do it with rules
first, so retrieval stays debuggable and reproducible).

Precedence, roughly most-specific first:
  1. planet + explicit house      -> planet_house
  2. planet + explicit sign       -> planet_sign
  3. planet + explicit nakshatra  -> nakshatra
  4. planet + dasha wording       -> dasha
  5. planet only (no house/sign)  -> pull the planet's ACTUAL house+sign from
                                      the chart (personalized default)
  6. house only                   -> that house's sign, and planet_house for
                                      each occupant (personalized)
  7. nakshatra only                -> nakshatra
  8. dasha wording only            -> current running mahadasha
  9. topic words ("career", ...)   -> map to houses, then behave like (6)
 10. nothing matched               -> natal snapshot fallback: ascendant
                                      sign, moon sign, current mahadasha
"""
from __future__ import annotations

import re
from typing import Any

from app.facts import chart_reader as cr
from app.facts import vocab

MAX_TOPIC_HOUSES = 4
_PERSONAL_PRONOUN_RE = re.compile(r"\b(my|i|me|mine)\b", re.IGNORECASE)


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _hashable_val(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_val(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_val(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_hashable_val(item) for item in value))
    return value


def dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[Any, ...], int] = {}
    out = []
    for f in facts:
        fact_type = f.get("type")
        if fact_type == "house_lookup":
            key = ("house_lookup", f.get("house"), f.get("sign"))
        elif fact_type == "planet_house":
            key = (
                "planet_house",
                f.get("planet"),
                f.get("house"),
                f.get("role"),
            )
        else:
            key = tuple(sorted((k, _hashable_val(v)) for k, v in f.items()))
        existing_index = seen.get(key)
        if existing_index is None:
            seen[key] = len(out)
            out.append(f)
        else:
            existing = out[existing_index]
            for field, value in f.items():
                if value is not None and (
                    field not in existing or existing[field] in (None, [], "")
                ):
                    existing[field] = value
    return out


def _facts_for_house(chart: dict[str, Any], house: int) -> list[dict[str, Any]]:
    return cr.house_facts(chart, house)


_GENERIC_TOPIC_WORDS = {
    "succeed",
    "success",
    "future",
    "changing",
    "support",
    "move",
    "bold",
}


def _ordered_topic_houses(
    question_lower: str,
    topics: list[str],
    max_houses: int = MAX_TOPIC_HOUSES,
) -> list[int]:
    """Prioritize domain nouns over generic wording, then cap house expansion."""
    ordered_topics = sorted(
        topics,
        key=lambda topic: (
            topic in _GENERIC_TOPIC_WORDS,
            question_lower.find(topic) if question_lower.find(topic) != -1 else 9999,
            -len(topic),
        ),
    )
    houses: list[int] = []
    for topic in ordered_topics:
        for house in vocab.TOPIC_HOUSES.get(topic, []):
            if house not in houses:
                houses.append(house)
                if len(houses) >= max_houses:
                    return houses
    return houses


def natal_snapshot_facts(chart: dict[str, Any]) -> list[dict[str, Any]]:
    """Generic fallback: ascendant (context), Moon/Sun placements, running mahadasha."""
    facts: list[dict[str, Any]] = []
    summary = cr.chart_summary(chart)
    if summary.get("ascendant_sign"):
        facts.append({"type": "sign_lookup", "sign": summary["ascendant_sign"]})
    for planet in ("Moon", "Sun"):
        data = cr.planet_house_sign_nakshatra(chart, planet)
        if data:
            if data.get("sign"):
                facts.append({"type": "planet_sign", "planet": planet, "sign": data["sign"]})
            if data.get("house") is not None:
                planet_house = {
                    "type": "planet_house",
                    "planet": planet,
                    "house": data["house"],
                }
                if data.get("sign"):
                    planet_house["sign"] = data["sign"]
                facts.append(planet_house)
    cur = cr.current_dasha(chart)
    if cur and cur.get("mahadasha"):
        facts.append({"type": "dasha", "dasha_lord": cur["mahadasha"]})
    return facts


def parse_question(question: str, chart: dict[str, Any]) -> list[dict[str, Any]]:
    q = question.lower()

    is_personal = bool(_PERSONAL_PRONOUN_RE.search(q)) and bool(
        chart.get("planets") or chart.get("houses")
    )
    planets = vocab.find_planets(q)
    houses = vocab.find_house_numbers(q)
    signs = vocab.find_signs(q)
    nakshatras = vocab.find_nakshatras(q)
    is_dasha_q = vocab.mentions_any(q, vocab.DASHA_WORDS)
    is_aspect_q = vocab.mentions_any(q, vocab.ASPECT_WORDS)
    is_timing_q = vocab.mentions_any(q, vocab.TIMING_WORDS)
    topics = vocab.find_topics(q)

    facts: list[dict[str, Any]] = []

    if planets and houses:
        for p in planets:
            data = cr.planet_house_sign_nakshatra(chart, p) if is_personal else None
            for h in houses:
                if is_personal:
                    if data and data.get("house") is not None:
                        actual_fact: dict[str, Any] = {
                            "type": "planet_house",
                            "planet": p,
                            "house": data["house"],
                        }
                        if data.get("sign"):
                            actual_fact["sign"] = data["sign"]
                        if data["house"] != h:
                            actual_fact["queried_house_mismatch"] = h
                        facts.append(actual_fact)
                        if data.get("sign"):
                            facts.append(
                                {"type": "planet_sign", "planet": p, "sign": data["sign"]}
                            )
                    facts.extend(_facts_for_house(chart, h))
                else:
                    facts.append({"type": "planet_house", "planet": p, "house": h})

    elif planets and signs:
        for p in planets:
            data = cr.planet_house_sign_nakshatra(chart, p) if is_personal else None
            for s in signs:
                if is_personal and data:
                    if data.get("house") is not None:
                        planet_house = {
                            "type": "planet_house",
                            "planet": p,
                            "house": data["house"],
                        }
                        if data.get("sign"):
                            planet_house["sign"] = data["sign"]
                        facts.append(planet_house)
                    if data.get("sign"):
                        facts.append(
                            {"type": "planet_sign", "planet": p, "sign": data["sign"]}
                        )
                else:
                    facts.append({"type": "planet_sign", "planet": p, "sign": s})

    elif planets and nakshatras:
        for p in planets:
            for n in nakshatras:
                facts.append({"type": "nakshatra", "nakshatra": n, "planet": p})

    elif planets and is_dasha_q:
        for p in planets:
            facts.append({"type": "dasha", "dasha_lord": p})

    elif planets and is_aspect_q:
        for p in planets:
            aspects = cr.aspects_from(chart, p)
            aspect_types = sorted({a["aspect_type"] for a in aspects}) or ["7th"]
            for at in aspect_types:
                facts.append({"type": "aspect", "planet": p, "aspect_type": at})

    elif planets:
        # Personalized default: "what does Mars mean for me" -> use the
        # planet's actual placement in THIS chart.
        for p in planets:
            data = cr.planet_house_sign_nakshatra(chart, p)
            if data:
                if data.get("house") is not None:
                    planet_house = {
                        "type": "planet_house",
                        "planet": p,
                        "house": data["house"],
                    }
                    if data.get("sign"):
                        planet_house["sign"] = data["sign"]
                    facts.append(planet_house)
                if data.get("sign"):
                    facts.append({"type": "planet_sign", "planet": p, "sign": data["sign"]})

    elif houses:
        for h in houses:
            facts.extend(_facts_for_house(chart, h))

    elif nakshatras:
        for n in nakshatras:
            facts.append({"type": "nakshatra", "nakshatra": n})

    elif is_dasha_q:
        cur = cr.current_dasha(chart)
        if cur and cur.get("mahadasha"):
            facts.append({"type": "dasha", "dasha_lord": cur["mahadasha"]})

    # Life-area facts supplement explicit planet, house, and timing intents.
    topic_cap = 2 if planets or houses else MAX_TOPIC_HOUSES
    for house in _ordered_topic_houses(q, topics, max_houses=topic_cap):
        facts.extend(_facts_for_house(chart, house))

    # Timing questions need a temporal chart context even when the user does
    # not explicitly mention the word "dasha". Keep the existing life-area
    # facts and add the running Mahadasha as an independent retrieval target.
    if is_timing_q:
        cur = cr.current_dasha(chart)
        if cur and cur.get("mahadasha"):
            facts.append({"type": "dasha", "dasha_lord": cur["mahadasha"]})

    if not facts:
        facts = natal_snapshot_facts(chart)

    return dedupe_facts(facts)


def fact_to_metadata_filter(fact: dict[str, Any]) -> dict[str, Any]:
    """Maps a parsed fact to the exact metadata filter used for corpus lookup."""
    t = fact["type"]
    if t == "planet_house":
        role = str(fact.get("role") or "")
        m = __import__("re").search(r"lord of house (\d+)", role.lower())
        if m:
            # Prefer an explicit house-lord placement relation over simple
            # co-occurrence of a lord name and a destination house.
            return {"lord_house_pairs": f"{int(m.group(1))}|{fact['house']}"}
        return {"planet_house_pairs": f"{fact['planet']}|{fact['house']}"}
    if t == "planet_sign":
        return {"planet_sign_pairs": f"{fact['planet']}|{fact['sign']}"}
    if t == "nakshatra":
        if fact.get("planet"):
            return {"planet_nakshatra_pairs": f"{fact['planet']}|{fact['nakshatra']}"}
        return {"nakshatra": fact["nakshatra"]}
    if t == "dasha":
        lord = str(fact["dasha_lord"])
        canonical_lord = vocab.PLANET_ALIASES.get(lord.casefold(), lord)
        return {"dasha_lord": canonical_lord}
    if t == "aspect":
        return {"planet_aspect_pairs": f"{fact['planet']}|{fact['aspect_type']}"}
    if t == "house_lookup":
        return {"house": fact["house"]}
    if t == "sign_lookup":
        return {"sign": fact["sign"]}
    return {}


def normalize_fact(fact: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """
    Returns (metadata_filter, semantic_query_text) for any fact dict --
    whether it's the internal `{"type": "planet_house", ...}` shape produced
    by `parse_question`, or a raw metadata filter like `{"planet": "Mars",
    "house": 10}` passed directly to POST /retrieve for debugging. Having
    both shapes work means the retrieval-testing endpoint is forgiving about
    exactly how a fact is phrased, which matters a lot for a debugging tool.
    """
    if "type" in fact:
        return fact_to_metadata_filter(fact), fact_to_query_text(fact)

    filt = {k: v for k, v in fact.items() if v is not None}
    query_text = " ".join(f"{v}" for v in filt.values())
    return filt, query_text


def fact_to_query_text(fact: dict[str, Any]) -> str:
    """Natural-language rendering of a fact, used as the semantic-search query."""
    t = fact["type"]
    if t == "planet_house":
        role = f", {fact['role']}" if fact.get("role") else ""
        return (
            f"{fact['planet']} placed in the "
            f"{_ordinal(int(fact['house']))} house{role}"
        )
    if t == "planet_sign":
        return f"{fact['planet']} placed in {fact['sign']}"
    if t == "nakshatra":
        if fact.get("planet"):
            return f"{fact['planet']} in {fact['nakshatra']} nakshatra"
        return f"{fact['nakshatra']} nakshatra"
    if t == "dasha":
        lord = str(fact["dasha_lord"])
        canonical_lord = vocab.PLANET_ALIASES.get(lord.casefold(), lord)
        return f"{canonical_lord} Dasha Mahadasha dasa period"
    if t == "aspect":
        return f"{fact['planet']} {fact['aspect_type']} house aspect"
    if t == "house_lookup":
        return f"the {_ordinal(int(fact['house']))} house in {fact.get('sign', '')}"
    if t == "sign_lookup":
        return f"{fact['sign']} sign"
    return ""
