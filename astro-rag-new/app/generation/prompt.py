from __future__ import annotations

import re
from typing import Any

from app.facts import chart_reader as cr
from app.facts.parser import fact_to_metadata_filter


_PLANETARY_DIGNITIES = {
    "sun": {"exalted": {"aries"}, "debilitated": {"libra"}, "own": {"leo"}},
    "moon": {"exalted": {"taurus"}, "debilitated": {"scorpio"}, "own": {"cancer"}},
    "mars": {
        "exalted": {"capricorn"},
        "debilitated": {"cancer"},
        "own": {"aries", "scorpio"},
    },
    "mercury": {
        "exalted": {"virgo"},
        "debilitated": {"pisces"},
        "own": {"gemini", "virgo"},
    },
    "jupiter": {
        "exalted": {"cancer"},
        "debilitated": {"capricorn"},
        "own": {"sagittarius", "pisces"},
    },
    "venus": {
        "exalted": {"pisces"},
        "debilitated": {"virgo"},
        "own": {"taurus", "libra"},
    },
    "saturn": {
        "exalted": {"libra"},
        "debilitated": {"aries"},
        "own": {"capricorn", "aquarius"},
    },
}


def _dignity_note(planet: str, sign: str | None) -> str:
    if not sign:
        return ""
    dignities = _PLANETARY_DIGNITIES.get(planet.casefold())
    if not dignities:
        return ""
    normalized_sign = sign.casefold()
    matches = [
        dignity
        for dignity in ("exalted", "debilitated", "own")
        if normalized_sign in dignities[dignity]
    ]
    if matches:
        return f" (dignity={' and '.join(matches)})"
    return ""


CORE_GROUNDING_POLICY = """\
CHART: Supplied chart facts are authoritative. Occupancy is not lordship; use
the explicit house-lord mapping. Check every rule condition against the chart.
EVIDENCE: Synthesize only when both chart facts and retrieved passages support
the claim. Cite only supplied passage labels and sources; invent no doctrine.
DIVISIONAL DATA: Use only divisions explicitly supplied; name missing data.
OUTPUT: Give a qualified user-facing answer, not internal reasoning or guarantees.
"""


SYSTEM_PROMPT = CORE_GROUNDING_POLICY + """\
You are an assistant that explains Vedic astrology (Jyotisha) chart readings.

You will be given three things:
1. A short factual snapshot of the person's birth chart (computed deterministically -- treat every line as ground truth).
2. A set of retrieved reference passages, each numbered and tagged with its source.
3. The person's question.

Rules:
1. Treat every line in CHART SNAPSHOT and RELEVANT CHART FACTS FOR THIS QUESTION as ground truth. Do not invent, alter, or assume any chart placement, house, sign, nakshatra, dasha, conjunction, aspect, lordship, relationship, or other chart fact that is not explicitly provided.

2. Use the RETRIEVED REFERENCE PASSAGES as the primary doctrinal evidence for the reading. When you rely on a passage, cite its passage number and source, e.g. "Passage 2 — Brihat Jataka." Use the metadata shown with the passage when useful (chapter, sloka, PDF page).

3. Synthesize only when the chart fact is explicitly supplied and the retrieved passage supports the theme being inferred. Do not use pretrained astrology knowledge to add a missing rule, yoga, significator, dignity, aspect, divisional chart, or prediction. Translate supported technical descriptions into understandable practical tendencies without presenting them as direct quotations.

4. Distinguish three evidence levels:
   - DIRECT TEXT: explicitly stated in a retrieved passage.
   - TRADITIONAL SYNTHESIS: a reasonable interpretation obtained by combining the supplied chart facts with themes present in the retrieved passages.
   - UNSUPPORTED DOCTRINE: an astrological rule, combination, significator, formula, or conclusion that requires information not present in the chart snapshot or retrieved passages.
   Use DIRECT TEXT and TRADITIONAL SYNTHESIS. Do not introduce UNSUPPORTED DOCTRINE merely to make the answer more detailed.

5. When a retrieved passage connects a house, planet, relationship, or other factor with themes such as popularity, comfort, learning, relatives, possessions, domestic life, personality, or status, you may translate those themes into practical tendencies when the chart facts support that connection. Present such statements as traditional tendencies, not guaranteed facts.

6. Synthesize multiple relevant chart facts and passages into one coherent reading. Answer the user's actual question first. Do not turn the response into a passage-by-passage literature review or a dump of every retrieved passage.

7. Do not overstate an inference. Prefer language such as "traditionally associated with," "often points toward," "may indicate," "can suggest," "the chart suggests," or "this combination may manifest as." Avoid deterministic wording such as "you definitely are" or "you will definitely."

8. If two retrieved passages genuinely disagree, explicitly point out the disagreement and attribute it to the relevant passages instead of silently blending them into one claim.

9. If the passages do not directly answer part of the question, first check whether a reasonable interpretation can still be made from the supplied chart facts and the themes actually present in the passages. If yes, provide the cautious interpretation. Only say that the material is insufficient when the requested conclusion would require unsupported rules or missing chart information.

10. Keep the answer focused and readable. For general questions, give the direct reading first, followed by a small number of supporting points. For personality questions specifically, use this structure:
   - Start with one short overall personality summary using roughly 3–5 cautious descriptors.
   - Follow with 3–5 concise supporting points connecting the relevant chart factors to the retrieved passages.
   - End with a brief limitation sentence naming important personality areas that cannot be inferred from the supplied material, when applicable.

11. Do not begin with a limitation such as "the passages do not directly say..." when a useful traditional synthesis can reasonably be made. Put limitations after the reading unless the evidence truly prevents a meaningful interpretation.

12. Astrological placements describe traditional tendencies and associations, not certainties. Do not present them as scientific facts or guaranteed outcomes.
13. Preserve the scope and conditions of each source rule. Do not transfer a rule about a house lord to a planet in that house unless the cited passage makes that connection. Do not introduce generic planet/house meanings unless they are stated in retrieved evidence.
14. For a question about an exact formula or calculation rule, answer with the directly relevant formula and its stated conditions. Do not blend it with a neighboring rule that uses similar terms or numbers.
15. If a source contains severe historical claims about death or harm, describe them only as claims made by that source, with their exact stated condition. Never turn them into guidance or a prediction about the user.
16. When retrieved evidence is incomplete for the question, state the narrow supported answer and identify what the passages do not establish. Do not fill gaps with general astrological knowledge.
17. Cite each source-based claim with the specific passage number that supports it. Avoid vague combined references such as "Passage 2/4/5"; when claims rely on different passages, identify the relevant passage beside each claim.
18. Passage numbers are the 1-based labels shown in the provided evidence list. Do not use a chapter or sloka number as a passage number; write "Sloka 14" when referring to a source verse unless the evidence list labels it Passage 14.
19. For a narrow question about one calculation, give that passage's stated calculation directly. Do not characterize nearby rules as competing methods or claim a textual inconsistency unless the question asks you to compare them and the passages establish a real contradiction.
20. Verify each condition in a retrieved passage against the supplied chart snapshot and relevant chart facts before applying it. If the chart explicitly contradicts a required condition, say that condition does not apply. If the supplied chart data does not establish the condition either way, do not assume it is present; state that it cannot be verified from the available chart data.
21. If none of the retrieved passages is relevant to the question, say that the retrieved classical passages do not provide a rule to answer it. Never invent meanings for signs or slokas, expose internal notes such as "Intent:" or "Filter context:", or reproduce raw "[Passage ...]" blocks.
22. Before assigning any planet's house, sign, or lordship, cross-check that exact assignment against the supplied chart facts. Never silently swap placements or use a passage's example as this chart's placement.
23. Do not assert mutually exclusive yogas or both the presence and absence of the same defining condition. Apply a named yoga only when every defining condition is explicitly supported by the chart facts and a relevant passage; if the evidence is incomplete or conflicting, state the limitation instead.
24. Do not invent Sanskrit names, yoga classifications, dignities, or planet nicknames. Use technical labels only when a retrieved passage names them and the passage's conditions are met.
25. A planet's occupancy is not its lordship. Determine house lords only from the supplied Houses map; never transfer the lordship in an example passage to this chart.
26. Verify dignity only from the planet's supplied sign and the traditional dignity table. If its sign is not listed as exalted, debilitated, or own, call it an ordinary sign and do not claim a special dignity.
27. Use D-1 chart facts unless D-9, D-10, or another divisional chart is explicitly supplied. If the user asks about an unavailable divisional chart, state that its data was not supplied; never cite an unavailable divisional chart as evidence.
28. For narrow factual questions, answer in 2–5 sentences. For ordinary interpretive questions, use about 80–140 words; for broad life-area questions, use about 120–180 words. Do not add detail just to reach a target length.
29. Do not append a generic astrologer disclaimer to every answer. Mention limitations only when they materially affect the answer.
"""


def _format_chart_facts(
    chart_facts: list[dict[str, Any]],
    chart: dict[str, Any] | None = None,
) -> str:
    lines = []
    for fact in chart_facts:
        t = fact["type"]
        if t == "planet_house":
            planet = fact["planet"]
            house = fact["house"]
            role = ""
            role_match = None
            if fact.get("role"):
                role_match = re.search(
                    r"lord of house (\d+)", str(fact["role"]), re.IGNORECASE
                )
            if role_match:
                role = f"; role=lord_of_H{role_match.group(1)}"
            elif fact.get("role"):
                role = f"; role={fact['role']}"
            sign = fact.get("sign")
            if not sign and chart:
                planet_data = cr.planet_house_sign_nakshatra(chart, planet)
                sign = (planet_data or {}).get("sign")
                if not sign and fact.get("role"):
                    house_number = fact["role"].rsplit(" ", 1)[-1]
                    try:
                        house_data = cr.house_summary(chart, int(house_number))
                    except ValueError:
                        house_data = None
                    sign = (house_data or {}).get("lord_placement_sign")
                if not sign:
                    house_data = cr.house_summary(chart, house)
                    sign = (house_data or {}).get("sign")
            sign_note = f"; sign={sign}" if sign else ""
            ruled_houses = [
                str(house.get("number"))
                for house in (chart or {}).get("houses", [])
                if house.get("lord") == planet and house.get("number") is not None
            ]
            ruled_note = (
                f"; houses_ruled={','.join(f'H{number}' for number in ruled_houses)}"
                if ruled_houses
                else ""
            )
            mismatch_house = fact.get("queried_house_mismatch")
            mismatch_note = (
                f" [Note: {planet} is in house {house}, NOT in house {mismatch_house}]"
                if mismatch_house
                else ""
            )
            lines.append(
                f"- {planet}: house=H{house}{sign_note}{role}{ruled_note}"
                f"{mismatch_note}"
            )
        elif t == "planet_sign":
            dignity_note = _dignity_note(fact["planet"], fact["sign"])
            lines.append(
                f"- {fact['planet']}: sign={fact['sign']}{dignity_note}"
            )
        elif t == "nakshatra":
            if fact.get("planet"):
                lines.append(f"- {fact['planet']}: {fact['nakshatra']} nakshatra")
            else:
                lines.append(f"- Nakshatra: {fact['nakshatra']}")
        elif t == "dasha":
            lines.append(f"- Dasha lord: {fact['dasha_lord']} (current running mahadasha)")
        elif t == "aspect":
            lines.append(f"- {fact['planet']}: {fact['aspect_type']} aspect")
        elif t == "house_lookup":
            house = fact["house"]
            sign = fact.get("sign", "unknown sign")
            lord = fact.get("lord")
            occupants = fact.get("occupants")
            if chart and (lord is None or occupants is None):
                house_data = cr.house_summary(chart, house)
                if house_data:
                    lord = lord or house_data.get("lord")
                    if occupants is None:
                        occupants = house_data.get("occupants", [])
            lord_note = f"; lord={lord}" if lord else ""
            occupant_note = (
                f"; occupants={','.join(occupants) if occupants else 'none'}"
                if occupants is not None
                else ""
            )
            lines.append(
                f"- H{house}: sign={sign}{lord_note}{occupant_note}"
            )
        elif t == "sign_lookup":
            lines.append(f"- Sign: {fact['sign']}")
    return "\n".join(lines) if lines else "(no specific chart facts matched this question)"


def _chart_headline(chart: dict[str, Any]) -> str:
    if not chart:
        return "No chart supplied."
    summary = cr.chart_summary(chart)
    cur = cr.current_dasha(chart)
    parts = []
    if summary.get("ascendant_sign"):
        parts.append(f"Ascendant: {summary['ascendant_sign']}")
    if summary.get("moon_sign"):
        parts.append(f"Moon sign: {summary['moon_sign']}")
    if summary.get("moon_nakshatra"):
        parts.append(f"Moon nakshatra: {summary['moon_nakshatra']} (pada {summary.get('moon_pada', '?')})")
    if summary.get("sun_sign"):
        parts.append(f"Sun sign: {summary['sun_sign']}")
    if cur and cur.get("mahadasha"):
        antar = f" / {cur['antardasha']} antardasha" if cur.get("antardasha") else ""
        parts.append(f"Current dasha: {cur['mahadasha']} mahadasha{antar}")
    chart_settings = chart.get("settings") or {}
    house_system = chart.get("house_system") or chart_settings.get("house_system")
    if house_system:
        parts.append(f"House system: {house_system}")
    headline = " | ".join(parts) if parts else "(no chart summary available)"

    lordships_by_planet: dict[str, list[int]] = {}
    house_parts = []
    for house_number in range(1, 13):
        house = cr.house_summary(chart, house_number)
        if house is None:
            house_parts.append(
                f"H{house_number} sign=not supplied lord=not supplied "
                "occupants=not supplied"
            )
            continue
        sign = house.get("sign") or "not supplied"
        lord = house.get("lord") or "not supplied"
        if lord not in {"unknown", "not supplied"}:
            lordships_by_planet.setdefault(lord, []).append(house_number)
        occupants = ",".join(house.get("occupants") or [])
        occupant_text = occupants or "none"
        house_parts.append(
            f"H{house_number} sign={sign} lord={lord} occupants={occupant_text}"
        )
    if chart.get("houses"):
        headline += "\nHouses: " + " ".join(house_parts)

    planet_parts = []
    for planet in chart.get("planets", []):
        body = planet.get("body")
        sign = (planet.get("position") or {}).get("zodiac_sign")
        house = (planet.get("placement") or {}).get("house")
        if not body:
            continue
        ruled = lordships_by_planet.get(str(body), [])
        ruled_text = (
            f" lord_of={','.join(f'H{number}' for number in ruled)}"
            if ruled
            else ""
        )
        sign_text = sign or "not supplied"
        house_text = f"H{house}" if house is not None else "not supplied"
        dignity = _dignity_note(str(body), str(sign) if sign else None)
        dignity_text = (
            f" dignity={dignity.removeprefix(' (dignity=').removesuffix(')')}"
            if dignity
            else ""
        )
        planet_parts.append(
            f"{body} sign={sign_text} house={house_text}{ruled_text}{dignity_text}"
        )
    if planet_parts:
        headline += "\nPlanets: " + " | ".join(planet_parts)
    return headline


def _verify_passage_conditions(
    passage: dict[str, Any],
    chart: dict[str, Any] | None,
) -> str:
    if not chart or not (chart.get("planets") or chart.get("houses")):
        return ""
    metadata = passage.get("metadata") or {}
    unmet: list[str] = []
    bindings: list[str] = []

    for pair in metadata.get("planet_house_pairs") or []:
        if "|" not in str(pair):
            continue
        planet, required_house = str(pair).split("|", 1)
        placement = cr.planet_house_sign_nakshatra(chart, planet.strip())
        if placement and placement.get("house") is not None:
            actual_house = placement["house"]
            if str(actual_house) != required_house.strip():
                sign = placement.get("sign") or "unknown sign"
                unmet.append(
                    f"{planet} is in H{actual_house} {sign} "
                    f"(NOT H{required_house.strip()})"
                )

    for pair in metadata.get("lord_house_pairs") or []:
        if "|" not in str(pair):
            continue
        source_house, required_house = str(pair).split("|", 1)
        try:
            source_number = int(source_house.strip())
            required_number = int(required_house.strip())
        except ValueError:
            continue
        house = cr.house_summary(chart, source_number)
        actual_house = (house or {}).get("lord_placement_house")
        lord = (house or {}).get("lord")
        if not lord or actual_house is None:
            continue
        if actual_house != required_number:
            unmet.append(
                f"H{source_number} lord ({lord}) is in H{actual_house} "
                f"(NOT H{required_number}; this lord-placement condition is not met)"
            )
            continue

        destination = cr.house_summary(chart, required_number)
        destination_sign = (destination or {}).get("sign") or "unknown sign"
        disambiguation = ""
        destination_lord = (destination or {}).get("lord")
        if destination_lord and destination_lord != lord:
            destination_lord_data = cr.planet_house_sign_nakshatra(
                chart, destination_lord
            )
            if destination_lord_data and destination_lord_data.get("house") is not None:
                destination_lord_sign = destination_lord_data.get("sign") or "unknown sign"
                disambiguation = (
                    f"; distinguish this from H{required_number} lord "
                    f"{destination_lord}, placed in H{destination_lord_data['house']} "
                    f"{destination_lord_sign}"
                )
        bindings.append(
            f"H{source_number} lord {lord} is placed in H{required_number} "
            f"{destination_sign}{disambiguation}"
        )

    if not unmet and not bindings:
        return ""
    notes = []
    if bindings:
        notes.append("[Chart Binding: " + "; ".join(bindings) + ".]")
    if unmet:
        notes.append(
            "[Chart Verification Note: In this chart, "
            + "; ".join(unmet)
            + ". Do not claim these unmet conditions apply.]"
        )
    return "\n" + "\n".join(notes)


def _format_passages(
    passages: list[dict[str, Any]],
    chart: dict[str, Any] | None = None,
) -> str:
    if not passages:
        return "(no matching reference passages were retrieved)"
    lines = []
    for i, p in enumerate(passages, start=1):
        meta = p.get("metadata", {})
        source = meta.get("source", {})
        source_label = source.get("book") if isinstance(source, dict) else source
        source_label = source_label or "uploaded_vedic_astrology_corpus"

        chapter_no = meta.get("chapter_no")
        chapter_title = meta.get("chapter_title")
        sloka_start = meta.get("sloka_start")
        sloka_end = meta.get("sloka_end")
        pdf_page_start = source.get("pdf_page_start") if isinstance(source, dict) else None
        pdf_page_end = source.get("pdf_page_end") if isinstance(source, dict) else None

        provenance = [source_label]
        if chapter_no is not None:
            chapter_text = f"Chapter {chapter_no}"
            if chapter_title:
                chapter_text += f" — {chapter_title}"
            provenance.append(chapter_text)
        if sloka_start is not None:
            sloka_text = f"Sloka {sloka_start}"
            if sloka_end is not None and sloka_end != sloka_start:
                sloka_text += f"–{sloka_end}"
            provenance.append(sloka_text)
        if pdf_page_start is not None:
            page_text = f"PDF p. {pdf_page_start}"
            if pdf_page_end is not None and pdf_page_end != pdf_page_start:
                page_text += f"–{pdf_page_end}"
            provenance.append(page_text)

        source_detail = " | ".join(provenance)
        verification_note = _verify_passage_conditions(p, chart)
        lines.append(
            f"[Passage {i} — source: {source_detail}, match: {p.get('match_type')}]\n"
            f"{p['text']}{verification_note}"
        )
    return "\n\n".join(lines)


def _metadata_values(metadata: dict[str, Any], key: str) -> list[str]:
    aliases = {
        "nakshatra": "nakshatras",
        "dasha_lord": "dasha_lords",
    }
    value = metadata.get(key, metadata.get(aliases.get(key, "")))
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip().casefold() for item in values if item is not None]


def _has_structural_chart_match(
    chart_facts: list[dict[str, Any]],
    passages: list[dict[str, Any]],
) -> bool:
    """Require a retrieved passage to carry a chart fact's exact indexed relation."""
    structural_types = {"planet_house", "planet_sign", "nakshatra", "aspect", "dasha"}
    for passage in passages:
        fact = passage.get("matched_fact")
        metadata = passage.get("metadata")
        if (
            not isinstance(fact, dict)
            or fact.get("type") not in structural_types
            or fact not in chart_facts
            or not isinstance(metadata, dict)
        ):
            continue
        filters = fact_to_metadata_filter(fact)
        if not filters:
            continue
        if all(
            str(expected).strip().casefold()
            in _metadata_values(metadata, key)
            for key, expected in filters.items()
        ):
            return True
    return False


def _retrieval_is_sufficient(
    chart_facts: list[dict[str, Any]],
    passages: list[dict[str, Any]],
) -> bool:
    if _has_structural_chart_match(chart_facts, passages):
        return True
    for passage in passages:
        try:
            score = float(passage.get("score"))
        except (TypeError, ValueError):
            continue
        if score >= 0.45:
            return True
    return False


def build_prompt(
    chart: dict[str, Any],
    chart_facts: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    question: str,
) -> tuple[str, str]:
    abstention_guard = ""
    if not _retrieval_is_sufficient(chart_facts, passages):
        abstention_guard = (
            "\n\nRETRIEVAL EVIDENCE WARNING\n"
            "No passage has a strong relevance score or an exact indexed chart "
            "relation. Treat retrieved text only as context, do not use pretrained "
            "astrology knowledge to fill gaps, and state when the supplied "
            "references do not establish the requested rule. "
            "(If no retrieved passage directly answers this question, abstain; "
            "do not infer relevance from shared keywords or chapter titles.)"
        )
    user_prompt = f"""\
CHART SNAPSHOT
{_chart_headline(chart)}

RELEVANT CHART FACTS FOR THIS QUESTION
{_format_chart_facts(chart_facts, chart)}

RETRIEVED REFERENCE PASSAGES
{_format_passages(passages, chart)}

QUESTION
{question}{abstention_guard}
"""
    return SYSTEM_PROMPT, user_prompt
