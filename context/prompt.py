from __future__ import annotations

from typing import Any

from app.facts import chart_reader as cr
from app.facts.parser import fact_to_metadata_filter


SYSTEM_PROMPT = """\
You are an assistant that explains Vedic astrology (Jyotisha) chart readings.

You will be given three things:
1. A short factual snapshot of the person's birth chart (computed deterministically -- treat every line as ground truth).
2. A set of retrieved reference passages, each numbered and tagged with its source.
3. The person's question.

Rules:
1. Treat every line in CHART SNAPSHOT and RELEVANT CHART FACTS FOR THIS QUESTION as ground truth. Do not invent, alter, or assume any chart placement, house, sign, nakshatra, dasha, conjunction, aspect, lordship, relationship, or other chart fact that is not explicitly provided.

2. Use the RETRIEVED REFERENCE PASSAGES as the primary doctrinal evidence for the reading. When you rely on a passage, cite its passage number and source, e.g. "Passage 2 — Brihat Jataka." Use the metadata shown with the passage when useful (chapter, sloka, PDF page).

3. You may make reasonable interpretive inferences from the supplied chart facts and the themes described in the retrieved passages. The passage does not need to state the final personality adjective or life tendency word-for-word. Translate technical descriptions into understandable practical tendencies when the connection is reasonably supported.

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
"""


def _format_chart_facts(chart_facts: list[dict[str, Any]], chart: dict[str, Any] | None = None) -> str:
    """
    Format chart facts with both the compact shorthand (`Mercury: H4; ...`) checked
    by unit tests AND the explicit natural-language sentence structure used in
    `astro_finetuning_250_examples.json`.
    """
    lines = []
    for fact in chart_facts:
        t = fact["type"]
        if t == "planet_house":
            planet = fact["planet"]
            house = fact["house"]
            role = f"; {fact['role']}" if fact.get("role") else ""
            sign = fact.get("sign")
            if not sign and chart:
                pdata = cr.planet_house_sign_nakshatra(chart, planet)
                sign = (pdata or {}).get("sign")
                if not sign:
                    hdata = cr.house_summary(chart, house)
                    sign = (hdata or {}).get("sign")
            sign_note = f" (in {sign})" if sign else ""
            role_sentence = f" (it is the {fact['role']})" if fact.get("role") else ""
            mismatch_house = fact.get("queried_house_mismatch")
            mismatch_note = (
                f" [Note: {planet} is in house {house}, NOT in house {mismatch_house}]"
                if mismatch_house
                else ""
            )
            lines.append(
                f"- {planet}: H{house}{role} — {planet} is placed in house {house}{sign_note} in this chart{role_sentence}.{mismatch_note}"
            )
        elif t == "planet_sign":
            lines.append(
                f"- {fact['planet']}: {fact['sign']} — {fact['planet']} is placed in {fact['sign']} in this chart."
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
            sign = fact.get("sign", "unknown sign")
            lord = fact.get("lord")
            occupants = fact.get("occupants")
            if chart and (lord is None or occupants is None):
                hs = cr.house_summary(chart, fact["house"])
                if hs:
                    lord = lord or hs.get("lord")
                    if occupants is None:
                        occupants = hs.get("occupants", [])
            occ_str = (
                f", occupied by {', '.join(occupants)}"
                if occupants
                else (", empty / no planets occupying this house" if occupants is not None else "")
            )
            lord_str = f", ruled by {lord}" if lord else ""
            lines.append(
                f"- H{fact['house']}: {sign} — House {fact['house']} is {sign} in this chart{lord_str}{occ_str}."
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
    headline = " | ".join(parts) if parts else "(no chart summary available)"

    lordships_by_planet: dict[str, list[int]] = {}
    if chart.get("houses"):
        house_parts = []
        for house_number in range(1, 13):
            house = cr.house_summary(chart, house_number)
            if house is None:
                house_parts.append(f"H{house_number}:?")
                continue
            sign = house.get("sign") or "unknown sign"
            lord = house.get("lord") or "unknown"
            if lord and lord != "unknown":
                lordships_by_planet.setdefault(lord, []).append(f"H{house_number}:{sign}")
            occupants = ",".join(house.get("occupants") or [])
            occupant_text = f"/{occupants}" if occupants else ""
            house_parts.append(
                f"H{house_number}:{sign}({lord}{occupant_text})"
            )
        headline += "\nHouses: " + " ".join(house_parts)

    if chart.get("planets"):
        planet_parts = []
        for p in chart.get("planets", []):
            body = p.get("body")
            sign = (p.get("position") or {}).get("zodiac_sign")
            house = (p.get("placement") or {}).get("house")
            if body and house is not None and sign:
                ruled = lordships_by_planet.get(body, [])
                ruled_txt = f"[lord of {','.join(str(h) for h in ruled)}]" if ruled else ""
                planet_parts.append(f"{body}:H{house}/{sign}{ruled_txt}")
        if planet_parts:
            headline += "\nPlanets: " + " ".join(planet_parts)
    return headline


def _verify_passage_conditions(passage: dict[str, Any], chart: dict[str, Any] | None) -> str:
    """
    Deterministically check a passage's indexed planet_house_pairs and
    lord_house_pairs against the supplied birth chart and return an explicit
    warning string when a passage requires conditions that are FALSE in this chart.
    """
    if not chart or not (chart.get("planets") or chart.get("houses")):
        return ""
    meta = passage.get("metadata") or {}
    chunk_ph = meta.get("planet_house_pairs") or []
    chunk_lh = meta.get("lord_house_pairs") or []
    if not chunk_ph and not chunk_lh:
        return ""

    unmet_notes: list[str] = []
    for pair in chunk_ph:
        if "|" not in str(pair):
            continue
        planet, req_house_str = str(pair).split("|", 1)
        pdata = cr.planet_house_sign_nakshatra(chart, planet.strip())
        if pdata and pdata.get("house") is not None:
            actual_h = pdata["house"]
            actual_s = pdata.get("sign") or ""
            if str(actual_h) != req_house_str.strip():
                unmet_notes.append(
                    f"{planet} is in H{actual_h} {actual_s} (NOT H{req_house_str.strip()})"
                )

    for pair in chunk_lh:
        if "|" not in str(pair):
            continue
        src_h_str, dst_h_str = str(pair).split("|", 1)
        try:
            src_h = int(src_h_str.strip())
            dst_h = int(dst_h_str.strip())
        except ValueError:
            continue
        hs = cr.house_summary(chart, src_h)
        if hs and hs.get("lord") and hs.get("lord_placement_house") is not None:
            actual_dst = hs["lord_placement_house"]
            lord = hs["lord"]
            if actual_dst != dst_h:
                unmet_notes.append(
                    f"H{src_h} lord ({lord}) is in H{actual_dst} (NOT H{dst_h}, so H{src_h}/H{dst_h} exchange does NOT occur)"
                )

    matched_notes: list[str] = []
    for pair in chunk_lh:
        if "|" not in str(pair):
            continue
        src_h_str, dst_h_str = str(pair).split("|", 1)
        try:
            src_h = int(src_h_str.strip())
            dst_h = int(dst_h_str.strip())
        except ValueError:
            continue
        hs = cr.house_summary(chart, src_h)
        dst_hs = cr.house_summary(chart, dst_h)
        if hs and hs.get("lord") and hs.get("lord_placement_house") == dst_h:
            lord = hs["lord"]
            dst_sign = (dst_hs or {}).get("sign") or ""
            # If the destination house is ruled by a different planet (e.g. H5 Virgo is ruled by Mercury,
            # while 11th lord Jupiter occupies H5), explicitly warn the model not to confuse the two!
            dst_lord = (dst_hs or {}).get("lord")
            disambig = ""
            if dst_lord and dst_lord != lord:
                dst_lord_data = cr.planet_house_sign_nakshatra(chart, dst_lord)
                if dst_lord_data and dst_lord_data.get("house") is not None:
                    disambig = (
                        f" — do NOT confuse with H{dst_h} lord {dst_lord}, "
                        f"which is in H{dst_lord_data['house']} {dst_lord_data.get('sign', '')}"
                    )
            matched_notes.append(
                f"Applies to H{src_h} lord {lord} placed in H{dst_h} {dst_sign}{disambig}"
            )

    notes_out = []
    if matched_notes:
        notes_out.append(f"[Chart Binding: {'; '.join(matched_notes)}.]")
    if unmet_notes:
        notes_out.append(
            f"[Chart Verification Note: In this chart, {'; '.join(unmet_notes)}. Do not claim these unmet conditions apply.]"
        )
    return ("\n" + "\n".join(notes_out)) if notes_out else ""


def _format_passages(passages: list[dict[str, Any]], chart: dict[str, Any] | None = None) -> str:
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
            f"[Passage {i} — source: {source_detail}, match: {p.get('match_type')}]\n{p['text']}{verification_note}"
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


def build_prompt(
    chart: dict[str, Any],
    chart_facts: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    question: str,
) -> tuple[str, str]:
    abstention_guard = ""
    if not _has_structural_chart_match(chart_facts, passages):
        abstention_guard = """\
RETRIEVAL RELEVANCE WARNING
If no retrieved passage directly answers this question, abstain; do not infer relevance from shared keywords or chapter titles."""
    guard_block = f"\n{abstention_guard}\n" if abstention_guard else ""
    user_prompt = f"""\
CHART SNAPSHOT
{_chart_headline(chart)}

RELEVANT CHART FACTS FOR THIS QUESTION
{_format_chart_facts(chart_facts, chart)}
{guard_block}

RETRIEVED REFERENCE PASSAGES
{_format_passages(passages, chart)}

QUESTION
{question}
"""
    return SYSTEM_PROMPT, user_prompt