"""Lazy local inference for the Astro Qwen LoRA adapter."""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from threading import Lock
from typing import Any

from app.config.settings import settings
from app.generation.prompt import _PLANETARY_DIGNITIES
from app.utils.errors import RAGError

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
ADAPTER_MODEL = "XeroxFlynn/astro-rag-lora-v2"


def _load_adapter_tokenizer(
    tokenizer_loader: Any,
    adapter_dir: Path,
    token: str | None,
    base_model: str = BASE_MODEL,
) -> Any:
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
    )
    tokenizer_source = (
        adapter_dir
        if any((adapter_dir / name).is_file() for name in tokenizer_files)
        else base_model
    )
    tokenizer = tokenizer_loader.from_pretrained(
        str(tokenizer_source),
        token=token,
        trust_remote_code=True,
    )
    chat_template_path = adapter_dir / "chat_template.jinja"
    if chat_template_path.is_file():
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    if not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise RuntimeError(
            "A string chat template was not found in the adapter or base "
            f"tokenizer for {ADAPTER_MODEL}."
        )
    return tokenizer


def _clean_model_output(text: str) -> str:
    text = re.sub(
        r"<tool_call(?:\s[^>]*)?>.*?</tool_call\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"</?tool_call(?:\s[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<thought_process(?:\s[^>]*)?>.*?(?:</thought_process>|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"<think(?:\s[^>]*)?>.*?(?:</think>|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"^\s*(?:Answer:|ANSWER)\s*", "", text)
    text = re.sub(r"^\s*\*\*Response:\*\*", "**Answer:**", text)
    text = re.sub(
        r"\bBrihat\s+Paras(?:h)?ara\s+(?:Bhashya|Hora\s+Sastry|"
        r"Hora\s+Samhasta)\b|\bBrihat\s+Parashar\s+Samhita\b|"
        r"\bBrihat\s+Parashar\s+Samh\s+Sastra\b|"
        r"\bBPBH\s*Parasara\b|\bBrihadaranyananda\b",
        "Brihat Parasara Hora Sastra",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bPhala\s+Deepika\b", "Phaladeepika", text, flags=re.IGNORECASE)
    text = re.sub(r"\bBrihat\s+Jatakam\b", "Brihat Jataka", text, flags=re.IGNORECASE)
    text = re.sub(r"\bJataka\s+Parijatha\b", "Jataka Parijata", text, flags=re.IGNORECASE)
    # The NPU model occasionally emits malformed expansions of familiar source
    # titles. Normalize only known variants; do not rewrite arbitrary citations.
    text = re.sub(
        r"\bBrihat\s+(?:Parasara|Parashar|Parāśara|Parasharā)\s+"
        r"(?:Bhashya|BhashSastra|BhashShastra|Bhash\s+Sastra|Samhita|Samh\s+Sastra)\b",
        "Brihat Parasara Hora Sastra",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bBṛhat\s+Parāśarāhāra\b|\bBrihat\s+Parasara\s+Hora\s+Sastra(?:\s+Hora\s+Sastra)+\b",
        "Brihat Parasara Hora Sastra",
        text,
        flags=re.IGNORECASE,
    )
    # Normalize frequent abbreviations without expanding unrelated words.
    text = re.sub(r"\b(?:BPBS|BPBH|BPSh|BPP)\b", "BPHS", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSlipper(?=\s+\d)", "Sloka", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPassge(?=\s+\d)", "Passage", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\(\*\*BP(?:BH?)?\*\*\)?\*{0,2}",
        "Brihat Parasara Hora Sastra",
        text,
    )
    text = re.sub(
        r"\b(Mercury|Venus|Earth|Mars|Jupiter|Saturn)\s*/\s*"
        r"(?:Pluto|Uranus|Neptune)\b",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:Dwipaka|Dvivida|Drishti)\s+Yoga\b",
        "classical combination",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[Chart (?:Binding|Verification Note):[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*:?\s*\*\*Answer:\*\*\s*\*\*Reasoning:\*\*",
        "**Answer:**",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*:\s*(\*\*Answer:\*\*)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = _strip_reasoning_leakage(text)
    text = re.split(r"(?:^|\n)\[Passage\s+\d+\s*[—-]", text, maxsplit=1)[0]
    text = re.split(r"\nNote:\s+Per the prompt", text, maxsplit=1)[0]
    text = re.split(
        r"\*\*\s*End\s+(?:of\s+)?(?:Abstain\s+)?Response\s*\*\*"
        r"|\n\*\*Final Answer:\*\*|\n\*\*Source Check:\*\*",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.sub(
        r"\s*(?:\()?\*?(?:According to|Per)\s+Rule\s+\d+[^\n]*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return _dedupe_repeated_sections(text.strip())


_BLOCKED_SECTION_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?"
    r"(?:Reasoning|Chart Factors Examined|Internal Verification)"
    r"(?:\*\*)?[ \t]*:"
)
_BLOCKED_SECTION_TITLES = (
    "reasoning",
    "chart factors examined",
    "internal verification",
)


def _stream_visible_text(text: str, *, final: bool = False) -> tuple[str, bool]:
    """Hide blocked section headings, including a heading split across chunks."""
    match = _BLOCKED_SECTION_RE.search(text)
    if match:
        return text[:match.start()].rstrip(), True

    line_start = max(text.rfind("\n"), text.rfind("\r")) + 1
    line = text[line_start:]
    candidate = line.lstrip()
    whitespace_length = len(line) - len(candidate)
    if candidate.startswith("#"):
        heading_prefix = re.match(r"#{1,6}[ \t]+", candidate)
        if heading_prefix:
            whitespace_length += heading_prefix.end()
            candidate = candidate[heading_prefix.end():]
    if candidate.startswith("**"):
        whitespace_length += 2
        candidate = candidate[2:]

    folded = candidate.casefold()
    possible_heading = False
    for title in _BLOCKED_SECTION_TITLES:
        if title.startswith(folded):
            possible_heading = bool(folded)
            break
        if folded.startswith(title):
            remainder = folded[len(title):]
            possible_heading = remainder in {"", "*", "**", " ", " *", " **"}
            break

    if possible_heading:
        if final:
            return text[:line_start].rstrip(), False
        return text[:line_start].rstrip(), False
    return text, False


def _repair_passage_citations(
    text: str,
    passages: list[dict[str, Any]],
) -> str:
    def source_name(index: int) -> str | None:
        if index < 1 or index > len(passages):
            return None
        metadata = passages[index - 1].get("metadata") or {}
        source = metadata.get("source")
        if isinstance(source, dict):
            label = source.get("book") or source.get("title") or source.get("book_id")
        else:
            label = source or metadata.get("book_id")
        return str(label).strip() if label else None

    citation = re.compile(
        r"(?i:\bPassage)\s+(?P<number>\d+)"
        r"(?:\s*[—–-]\s*(?P<source>"
        r"[A-Z][A-Za-z0-9&'’\-]*(?:\s+[A-Z][A-Za-z0-9&'’\-]*){0,5}))?"
    )

    def repair(match: re.Match[str]) -> str:
        number = int(match.group("number"))
        expected_source = source_name(number)
        if expected_source is None:
            return ""
        claimed_source = match.group("source")
        if claimed_source and claimed_source.strip().casefold() != expected_source.casefold():
            return f"Passage {number} — {expected_source}"
        return match.group(0)

    repaired = citation.sub(repair, text)
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    repaired = re.sub(r"\.\s*\.", ".", repaired)
    return repaired


def _strip_reasoning_leakage(text: str) -> str:
    """Remove generated analysis sections after the visible answer."""
    return re.split(
        r"(?:^|\n)[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?"
        r"(?:Reasoning|Chart Factors Examined|Internal Verification):"
        r"(?:\*\*)?[ \t]*",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].rstrip()


def _has_divisional_chart_data(chart: dict[str, Any]) -> bool:
    division_keys = {
        "d2", "d3", "d4", "d7", "d9", "d10", "d12", "d16", "d20",
        "d24", "d27", "d30", "d40", "d45", "d60",
        "divisionalchart", "divisionalcharts", "vargachart", "vargacharts",
        "navamsa", "dashamsa",
    }

    def contains_division(value: Any) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                if normalized_key in division_keys and nested:
                    return True
                if contains_division(nested):
                    return True
        elif isinstance(value, list):
            return any(contains_division(item) for item in value)
        return False

    return contains_division(chart)


def _remove_unsupported_divisional_claims(
    text: str,
    chart: dict[str, Any],
) -> str:
    if _has_divisional_chart_data(chart):
        return text

    division = re.compile(
        r"\b(?:D\s*[- ]?\s*(?:2|3|4|7|9|10|12|16|20|24|27|30|40|45|60)|"
        r"Navamsa|Navamsha|Dashamsa|Dasamsa)\b",
        re.IGNORECASE,
    )
    sentences = re.compile(r"[^.!?\n]*(?:[.!?]|$)")
    output = []
    for match in sentences.finditer(text):
        sentence = match.group(0)
        if not sentence:
            continue
        division_match = division.search(sentence)
        if not division_match:
            output.append(sentence)
            continue
        subject_and_claim = re.search(
            r"\b(?:your|my|this)\s+(?:D\s*[- ]?\s*\d+|Navamsa|Navamsha|"
            r"Dashamsa|Dasamsa)\b|\b(?:chart|horoscope)\s+"
            r"(?:shows|indicates|places|confirms)\b",
            sentence,
            re.IGNORECASE,
        )
        already_disclaimed = re.search(
            r"\b(?:not supplied|not provided|unavailable|cannot assess|"
            r"cannot be assessed|cannot verify)\b",
            sentence,
            re.IGNORECASE,
        )
        if subject_and_claim and not already_disclaimed:
            name = division_match.group(0)
            output.append(
                f"The supplied chart data does not include {name}, "
                "so this claim cannot be verified."
            )
        else:
            output.append(sentence)
    return "".join(output)


def verify_and_repair_chart_claims(
    text: str,
    chart: dict[str, Any] | None = None,
    passages: list[dict[str, Any]] | None = None,
) -> str:
    """Repair generated planet placements, lordships, and unsupported yoga labels."""
    if not text:
        return text

    passage_text = " ".join(
        str(passage.get("text") or "") for passage in (passages or [])
    ).casefold()
    standard_yogas = {
        "raja yoga",
        "raj yoga",
        "dhana yoga",
        "viparita raja yoga",
        "parivartana yoga",
        "gajakesari yoga",
        "budhaditya yoga",
    }

    def preserve_supported_yoga(match: re.Match[str]) -> str:
        yoga_name = match.group(0)
        if yoga_name.casefold() in standard_yogas or yoga_name.casefold() in passage_text:
            return yoga_name
        return "classical combination"

    text = re.sub(
        r"\b[A-Z][A-Za-z-]*(?:\s+[A-Z][A-Za-z-]*){0,2}\s+Yoga\b",
        preserve_supported_yoga,
        text,
    )
    if not chart:
        return text

    ordinal_names = {
        1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th",
        7: "7th", 8: "8th", 9: "9th", 10: "10th", 11: "11th", 12: "12th",
    }
    planets = [
        str(planet.get("body"))
        for planet in chart.get("planets", [])
        if planet.get("body")
    ]
    planet_signs: dict[str, str | None] = {}
    planet_info: dict[str, tuple[int, str | None]] = {}
    for planet in chart.get("planets", []):
        body = planet.get("body")
        house = (planet.get("placement") or {}).get("house")
        if body:
            sign = (planet.get("position") or {}).get("zodiac_sign")
            planet_signs[str(body)] = str(sign) if sign else None
        if body and house is not None:
            try:
                planet_info[str(body)] = (
                    int(house),
                    str(sign) if sign else None,
                )
            except (TypeError, ValueError):
                continue

    house_lordships: dict[str, list[tuple[int, str]]] = {}
    for house in chart.get("houses", []):
        number = house.get("number")
        lord = house.get("lord")
        sign = house.get("sign")
        if number is None or not lord:
            continue
        try:
            house_lordships.setdefault(str(lord), []).append(
                (int(number), str(sign or "unknown sign"))
            )
        except (TypeError, ValueError):
            continue

    # Correct explicit house-lord labels from the deterministic Houses map.
    # A planet merely occupying a house is not that house's lord. This repairs
    # common inversions like "12th lord Rahu" when the map names Mars instead.
    house_lord_by_number: dict[int, str] = {}
    for house in chart.get("houses", []):
        try:
            number = int(house.get("number"))
        except (TypeError, ValueError):
            continue
        lord = house.get("lord")
        if lord:
            house_lord_by_number[number] = str(lord)

    def _sentence_for_position(source: str, position: int) -> str:
        left = max(
            source.rfind(".", 0, position),
            source.rfind("?", 0, position),
            source.rfind("!", 0, position),
            source.rfind("\n", 0, position),
        ) + 1
        right_candidates = [
            value for value in (
                source.find(".", position), source.find("?", position),
                source.find("!", position), source.find("\n", position),
            ) if value != -1
        ]
        right = min(right_candidates) if right_candidates else len(source)
        return source[left:right].strip()

    def _is_chart_claim(source: str, position: int) -> bool:
        sentence = _sentence_for_position(source, position)
        lowered = sentence.casefold()
        # Never rewrite a conditional textbook rule as though it were a fact
        # about this user (a common failure mode in a simple regex validator).
        if re.match(r"^(?:if|when|suppose|for example|in general|traditionally)\b", lowered):
            return False
        if re.search(
            r"\b(?:your chart|this chart|in your chart|in this chart|chart shows|"
            r"chart indicates|chart's|your horoscope|natal chart|birth chart|"
            r"in my chart|my horoscope|my natal chart|your kundli|my kundli|"
            r"for you|you have)\b",
            lowered,
        ):
            return True
        if re.search(
            r"\b(?:your|my)\s+\d{1,2}(?:st|nd|rd|th)\s+"
            r"(?:house\s+)?lord\b",
            lowered,
        ):
            return True
        return False

    def _repair_dignity_claims(source: str) -> str:
        for planet, sign in planet_signs.items():
            if not sign:
                continue
            dignities = _PLANETARY_DIGNITIES.get(planet.casefold())
            if not dignities:
                continue
            actual_statuses = {
                status
                for status in ("exalted", "debilitated", "own")
                if str(sign).casefold() in dignities[status]
            }
            if actual_statuses == {"exalted", "own"}:
                actual_phrase = "exalted and in its own sign"
            elif "exalted" in actual_statuses:
                actual_phrase = "exalted"
            elif "debilitated" in actual_statuses:
                actual_phrase = "debilitated"
            elif "own" in actual_statuses:
                actual_phrase = "in its own sign"
            else:
                actual_phrase = "in an ordinary sign"

            other_planets = "|".join(
                re.escape(name)
                for name in planets
                if name.casefold() != planet.casefold()
            )
            bridge = (
                rf"(?:(?!\b(?:{other_planets})\b)[^.!?\n]){{0,100}}?"
                if other_planets
                else r"[^.!?\n]{0,100}?"
            )
            pattern = re.compile(
                rf"\b(?P<planet>{re.escape(planet)})\b"
                rf"(?P<bridge>{bridge})"
                rf"(?P<claim>\bexalted\b|\bdebilitated\b|"
                rf"\b(?:in\s+)?(?:its\s+)?own\s+sign\b|"
                rf"\bordinary\s+sign\b)",
                re.IGNORECASE,
            )

            def repair(match: re.Match[str]) -> str:
                if not _is_chart_claim(source, match.start()):
                    return match.group(0)
                bridge_text = match.group("bridge")
                if re.search(r"\b(?:not|never|isn't|aren't)\b", bridge_text, re.IGNORECASE):
                    return match.group(0)
                claimed = match.group("claim").casefold()
                if "exalted" in claimed:
                    claimed_status = "exalted"
                elif "debilitated" in claimed:
                    claimed_status = "debilitated"
                elif "own sign" in claimed:
                    claimed_status = "own"
                else:
                    claimed_status = "ordinary"
                if claimed_status in actual_statuses:
                    return match.group(0)
                if claimed.startswith("in ") and bridge_text.casefold().endswith("in "):
                    bridge_text = bridge_text[:-3]
                return f"{match.group('planet')}{bridge_text}{actual_phrase}"

            source = pattern.sub(repair, source)
        return source

    planet_alternation = "|".join(
        re.escape(name) for name in sorted(planets, key=len, reverse=True)
    ) or r"(?!)"
    ordinals = r"(?:1st|2nd|3rd|[4-9]th|10th|11th|12th)"
    # Phrase form: "the 12th lord Rahu" / "your 5th-house lord Jupiter".
    lord_subject = re.compile(
        rf"(?P<prefix>\b(?:(?:your|this chart(?:'s|’s)?|the)\s+)?)"
        rf"(?P<num>\d{{1,2}})(?:st|nd|rd|th)\s+(?:house\s+)?lord\s+"
        rf"(?P<mark>\*{{0,2}})(?P<planet>{planet_alternation})\b(?P=mark)",
        re.IGNORECASE,
    )

    def repair_lord_subject(match: re.Match[str]) -> str:
        try:
            house_num = int(match.group("num"))
        except ValueError:
            return match.group(0)
        actual_lord = house_lord_by_number.get(house_num)
        if not actual_lord or match.group("planet").casefold() == actual_lord.casefold():
            return match.group(0)
        # Avoid rewriting a textbook's hypothetical condition or direct source
        # quotation; only repair applied chart readings.
        if not _is_chart_claim(text, match.start()):
            return match.group(0)
        replacement = ordinal_names.get(house_num, f"{house_num}th")
        mark = match.group("mark") or ""
        return f"{match.group('prefix')}{replacement} lord {mark}{actual_lord}{mark}"

    text = lord_subject.sub(repair_lord_subject, text)

    # Inverted form: "Rahu is the 12th lord". Keep the original surface phrase,
    # correcting only the named planet.
    inverted_lord = re.compile(
        rf"(?P<mark>\*{{0,2}})(?P<planet>{planet_alternation})\b(?P=mark)"
        rf"(?P<bridge>[^.!?\n]{{0,32}}?\b(?:is\s+)?the\s+)"
        rf"(?P<num>\d{{1,2}})(?:st|nd|rd|th)\s+(?:house\s+)?lord\b",
        re.IGNORECASE,
    )

    def repair_inverted_lord(match: re.Match[str]) -> str:
        try:
            house_num = int(match.group("num"))
        except ValueError:
            return match.group(0)
        actual_lord = house_lord_by_number.get(house_num)
        if not actual_lord or match.group("planet").casefold() == actual_lord.casefold():
            return match.group(0)
        if not _is_chart_claim(text, match.start()):
            return match.group(0)
        mark = match.group("mark") or ""
        replacement_ordinal = ordinal_names.get(house_num, f"{house_num}th")
        return f"{mark}{actual_lord}{mark}{match.group('bridge')}{replacement_ordinal} lord"

    text = inverted_lord.sub(repair_inverted_lord, text)

    # Correct single-house lordship statements too (the previous validator only
    # handled "Planet rules both X and Y houses").
    for house_number, actual_lord in house_lord_by_number.items():
        ordinal = ordinal_names.get(house_number, f"{house_number}th")
        wrong_lord_pattern = re.compile(
            rf"(?P<planet>\b{planet_alternation}\b)(?P<bridge>[^.!?\n]{{0,28}}?\brules\s+(?:the\s+)?"
            rf"{house_number}(?:st|nd|rd|th)\s+house)",
            re.IGNORECASE,
        )

        def repair_single_lord(match: re.Match[str]) -> str:
            if match.group("planet").casefold() == actual_lord.casefold():
                return match.group(0)
            if not _is_chart_claim(text, match.start()):
                return match.group(0)
            return f"{actual_lord}{match.group('bridge')}"

        text = wrong_lord_pattern.sub(repair_single_lord, text)

    for planet, (actual_house, actual_sign) in planet_info.items():
        others = [name for name in planets if name.casefold() != planet.casefold()]
        other_planets = "|".join(re.escape(name) for name in others)
        bridge_pattern = (
            rf"(?:(?!\b(?:{other_planets})\b)[^.\n]){{0,60}}?"
            if other_planets
            else r"[^.\n]{0,60}?"
        )
        placement_pattern = re.compile(
            rf"(?P<lead>\b{re.escape(planet)}\b"
            rf"{bridge_pattern}"
            rf"\b(?:is\s+)?(?:placed\s+in|positioned\s+in|sits\s+in|occupies|in)"
            rf"\s+(?:the\s+)?)(?P<number>\d{{1,2}})"
            rf"(?:st|nd|rd|th)\s+house(?P<details>\s*\([^)]*\))?",
            flags=re.IGNORECASE,
        )

        def repair_placement(match: re.Match[str]) -> str:
            bridge = match.group("lead").casefold()
            if re.search(r"\b(?:not|never|without)\b", bridge):
                return match.group(0)
            if not _is_chart_claim(text, match.start()):
                return match.group(0)
            try:
                claimed_house = int(match.group("number"))
            except ValueError:
                return match.group(0)
            claimed_details = (match.group("details") or "").casefold()
            sign_correct = not actual_sign or str(actual_sign).casefold() in claimed_details
            house_correct = f"h{actual_house}" in claimed_details
            if claimed_house == actual_house and sign_correct and house_correct:
                return match.group(0)
            ordinal = ordinal_names.get(actual_house, f"{actual_house}th")
            details = (
                f" ({actual_sign}/H{actual_house})"
                if actual_sign
                else f" (H{actual_house})"
            )
            return f"{match.group('lead')}{ordinal} house{details}"

        text = placement_pattern.sub(repair_placement, text)

        # Also validate the compact house notation used by the prompt and model
        # (e.g. "Mars is in H12 (Aries)").
        h_pattern = re.compile(
            rf"(?P<lead>\b{re.escape(planet)}\b"
            rf"{bridge_pattern}"
            rf"\b(?:is\s+)?(?:placed\s+in|positioned\s+in|sits\s+in|occupies|in)"
            rf"\s+)(?P<tick>`?)(?:H|house\s*)(?P<number>\d{{1,2}})(?P=tick)"
            rf"(?P<details>\s*\([^)]*\))?",
            flags=re.IGNORECASE,
        )

        def repair_h_number(match: re.Match[str]) -> str:
            if re.search(r"\b(?:not|never|without)\b", match.group("lead"), re.IGNORECASE):
                return match.group(0)
            if not _is_chart_claim(text, match.start()):
                return match.group(0)
            try:
                claimed_house = int(match.group("number"))
            except ValueError:
                return match.group(0)
            details_text = match.group("details") or ""
            if (
                claimed_house == actual_house
                and (not actual_sign or actual_sign.casefold() in details_text.casefold())
                and f"h{actual_house}" in details_text.casefold()
            ):
                return match.group(0)
            ordinal = ordinal_names.get(actual_house, f"{actual_house}th")
            details = f" ({actual_sign}/H{actual_house})" if actual_sign else f" (H{actual_house})"
            return f"{match.group('lead')}{ordinal} house{details}"

        text = h_pattern.sub(repair_h_number, text)

        ruled_houses = sorted(house_lordships.get(planet, []))
        known_house_numbers = set()
        for house in chart.get("houses", []):
            try:
                if house.get("number") is not None:
                    known_house_numbers.add(int(house["number"]))
            except (TypeError, ValueError):
                continue
        if len(ruled_houses) >= 1 and known_house_numbers >= set(range(1, 13)):
            ruled_pattern = re.compile(
                rf"\b{re.escape(planet)}\s+rules\s+(?:both\s+)?(?:the\s+)?"
                rf"\d{{1,2}}(?:st|nd|rd|th)(?:\s*\([^)]*\))?\s+and\s+"
                rf"(?:the\s+)?\d{{1,2}}(?:st|nd|rd|th)(?:\s*\([^)]*\))?"
                rf"\s+houses?(?:\s*\([^)]*\))?",
                flags=re.IGNORECASE,
            )
            house_names = [
                (
                    f"{ordinal_names.get(number, f'{number}th')} ({sign})"
                    if sign != "unknown sign"
                    else ordinal_names.get(number, f"{number}th")
                )
                for number, sign in ruled_houses
            ]
            if len(house_names) == 1:
                correct_claim = f"{planet} rules the {house_names[0]} house"
            else:
                correct_claim = (
                    f"{planet} rules the "
                    + ", ".join(house_names[:-1])
                    + f" and {house_names[-1]} houses"
                )
            text = ruled_pattern.sub(correct_claim, text)

    # Remove a self-contradictory "rather than Nth house" tail when the same
    # sentence has just been corrected to place that planet in that exact house.
    for planet, (actual_house, _actual_sign) in planet_info.items():
        ordinal = ordinal_names.get(actual_house, f"{actual_house}th")
        contrast = re.compile(
            rf"(?P<valid>\b{re.escape(planet)}\b[^.!?\n]{{0,110}}?"
            rf"\b(?:occupies|sits\s+in|is\s+in|placed\s+in|positioned\s+in)\s+"
            rf"(?:the\s+)?(?:{ordinal}\s+house|H{actual_house})(?:\s*\([^)]*\))?)"
            rf"\s+(?:rather\s+than|instead\s+of)\s+(?:the\s+)?{ordinal}\s+house"
            rf"(?:\s*\([^)]*\))?",
            re.IGNORECASE,
        )
        text = contrast.sub(lambda m: m.group("valid") if _is_chart_claim(text, m.start()) else m.group(0), text)

    # If an answer explicitly maps a sign to a house, validate that mapping
    # against the chart's whole-sign house map (e.g. Aries is H12, not H4).
    sign_to_house = {
        str(house.get("sign", "")).casefold(): int(house["number"])
        for house in chart.get("houses", [])
        if house.get("sign") and str(house.get("number", "")).isdigit()
    }
    sign_names = "|".join(re.escape(house.get("sign", "")) for house in chart.get("houses", []) if house.get("sign"))
    if sign_names:
        sign_house_pattern = re.compile(
            rf"(?P<sign>\b(?:{sign_names})\b)(?P<close>\)?)(?P<separator>\s*,?\s+which\s+is\s+(?:the\s+)?)"
            rf"(?P<number>\d{{1,2}})(?:st|nd|rd|th)\s+house(?P<details>\s*\([^)]*\))?",
            re.IGNORECASE,
        )

        def repair_sign_house(match: re.Match[str]) -> str:
            actual_house = sign_to_house.get(match.group("sign").casefold())
            if actual_house is None or not _is_chart_claim(text, match.start()):
                return match.group(0)
            claimed = int(match.group("number"))
            details = (match.group("details") or "").casefold()
            if claimed == actual_house and f"h{actual_house}" in details:
                return match.group(0)
            # If this "which is ..." clause follows a planet location that was
            # just corrected to a different sign, it referred to the model's
            # incorrect location premise. Drop the now-misattached clause.
            sentence_start = max(
                text.rfind(".", 0, match.start()), text.rfind("?", 0, match.start()),
                text.rfind("!", 0, match.start()), text.rfind("\n", 0, match.start()),
            ) + 1
            preceding = text[sentence_start:match.start()]
            location_pattern = re.compile(
                r"(?:occupies|sits\s+in|placed\s+in|positioned\s+in|\bin)\s+"
                r"(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)\s+house|H\d{1,2})"
                r"\s*\((?P<sign>[^()/]+)(?:/H\d{1,2})?\)",
                re.IGNORECASE,
            )
            location_matches = list(location_pattern.finditer(preceding))
            if location_matches and location_matches[-1].group("sign").strip().casefold() != match.group("sign").casefold():
                return ""
            ordinal = ordinal_names.get(actual_house, f"{actual_house}th")
            actual_sign = match.group("sign")
            return f"{match.group('sign')}{match.group('close')}{match.group('separator')}{ordinal} house ({actual_sign}/H{actual_house})"

        text = sign_house_pattern.sub(repair_sign_house, text)

    # Avoid duplicated phrasing introduced when a wrong sign/house clause is
    # corrected (e.g. "Mars occupies 4th house (Leo/H4), which is the 4th house").
    for house_number, ordinal in ordinal_names.items():
        repeated_house = re.compile(
            rf"(?P<before>[^.!?\n]{{0,110}}\b{ordinal}\s+house"
            rf"(?:\s*\([^)]*H{house_number}[^)]*\))?)"
            rf"\s*,?\s+which\s+is\s+(?:the\s+)?{ordinal}\s+house"
            rf"(?:\s*\([^)]*\))?",
            re.IGNORECASE,
        )
        text = repeated_house.sub(
            lambda match: match.group("before")
            if _is_chart_claim(text, match.start()) else match.group(0),
            text,
        )

    text = _repair_dignity_claims(text)
    text = _remove_unsupported_divisional_claims(text, chart)
    text = _repair_passage_citations(text, passages or [])
    return text


def _dedupe_repeated_sections(text: str) -> str:
    seen_sections: set[str] = set()
    section_header = re.compile(
        r"(?:^|\n)[ \t]*\*\*(Answer|Reasoning):\*\*",
        flags=re.IGNORECASE,
    )
    for match in section_header.finditer(text):
        section = match.group(1).casefold()
        if section in seen_sections:
            return text[:match.start()].rstrip()
        seen_sections.add(section)
    return text


class QwenLocalClient:
    """Load Qwen3 + the Astro LoRA adapter once, on the first selected request."""

    mode = "transformers"
    model = ADAPTER_MODEL

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = Lock()
        self._generation_lock = Lock()
        self.last_diagnostics: dict[str, Any] = {
            "status": "not_called",
            "incomplete_reason": None,
            "usage": None,
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                import bitsandbytes  # noqa: F401 - required by the base's NF4 weights
                from huggingface_hub import get_token, snapshot_download
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_dependencies_missing",
                    "Astro Qwen needs PyTorch, Transformers, PEFT, and Accelerate. "
                    "Install the optional dependencies listed in "
                    "requirements-local-model.txt, then restart Astro RAG.",
                    503,
                ) from exc

            try:
                hub_token = get_token()
                adapter_candidate = Path(ADAPTER_MODEL).expanduser()
                if adapter_candidate.is_dir():
                    adapter_path = adapter_candidate.resolve()
                elif not hub_token:
                    raise RAGError(
                        "qwen_hf_auth_required",
                        "The Astro Qwen adapter is in a private Hugging Face repo. "
                        "Run `hf auth login` as the same Windows user that runs Astro RAG, "
                        f"using an account with read access to {ADAPTER_MODEL}.",
                        503,
                    )
                else:
                    adapter_path = Path(
                        snapshot_download(ADAPTER_MODEL, token=hub_token)
                    )
                tokenizer = _load_adapter_tokenizer(
                    AutoTokenizer,
                    adapter_path,
                    hub_token,
                )
                # `auto` may create a mixed CPU/disk map when RAM is tight.
                # bitsandbytes 4-bit rejects mixed placement; use its CPU backend
                # explicitly on CPU-only machines and retain auto placement on GPU.
                device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    torch_dtype="auto",
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(
                    base,
                    adapter_path,
                    low_cpu_mem_usage=True,
                )
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
            except RAGError:
                self.last_diagnostics["status"] = "error"
                raise
            except ImportError as exc:
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_dependencies_missing",
                    "A local Qwen dependency is missing. Install the optional "
                    "dependencies listed in requirements-local-model.txt, "
                    "then restart Astro RAG.",
                    503,
                ) from exc
            except Exception as exc:
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_model_load_failed",
                    "Could not load Astro Qwen locally. Check network access to "
                    "Hugging Face, available system memory, and the model cache. "
                    f"Details: {exc}",
                    503,
                ) from exc

    def unload(self) -> None:
        with self._load_lock:
            if self._model is not None or self._tokenizer is not None:
                import gc

                self._model = None
                self._tokenizer = None
                gc.collect()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        self._ensure_loaded()
        try:
            import torch

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_device = self._model.get_input_embeddings().weight.device
            inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            eos_ids = []
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                eos_ids.append(eos_token_id)
            convert_tokens_to_ids = getattr(
                self._tokenizer, "convert_tokens_to_ids", None
            )
            if callable(convert_tokens_to_ids):
                im_end_id = convert_tokens_to_ids("<|im_end|>")
                unk_token_id = getattr(self._tokenizer, "unk_token_id", None)
                if (
                    isinstance(im_end_id, int)
                    and im_end_id != unk_token_id
                    and im_end_id not in eos_ids
                ):
                    eos_ids.append(im_end_id)
            generation_options: dict[str, Any] = {
                "max_new_tokens": min(
                    max_tokens or settings.generation_max_tokens, 700
                ),
                "do_sample": False,
                "repetition_penalty": 1.15,
            }
            if eos_ids:
                generation_options["eos_token_id"] = eos_ids
            if isinstance(eos_token_id, int):
                generation_options["pad_token_id"] = eos_token_id
            with self._generation_lock, torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    **generation_options,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            answer = _clean_model_output(
                self._tokenizer.decode(
                    output[0][prompt_length:], skip_special_tokens=True
                )
            )
        except RAGError:
            raise
        except Exception as exc:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": None,
            }
            raise RAGError(
                "qwen_generation_failed",
                f"Astro Qwen local generation failed: {exc}",
                502,
            ) from exc

        if not answer:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": "no_visible_output",
                "usage": None,
            }
            raise RAGError(
                "qwen_empty_output",
                "Astro Qwen returned no visible answer. Try a shorter question or check available memory.",
                502,
            )

        self.last_diagnostics = {
            "status": "completed",
            "incomplete_reason": None,
            "usage": None,
        }
        return answer

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Yield cleaned answer text while local Qwen generates in a worker."""
        import threading

        self._ensure_loaded()
        try:
            import torch
            from transformers import TextIteratorStreamer

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_device = self._model.get_input_embeddings().weight.device
            inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            eos_ids = []
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                eos_ids.append(eos_token_id)
            convert_tokens_to_ids = getattr(
                self._tokenizer, "convert_tokens_to_ids", None
            )
            if callable(convert_tokens_to_ids):
                im_end_id = convert_tokens_to_ids("<|im_end|>")
                unk_token_id = getattr(self._tokenizer, "unk_token_id", None)
                if (
                    isinstance(im_end_id, int)
                    and im_end_id != unk_token_id
                    and im_end_id not in eos_ids
                ):
                    eos_ids.append(im_end_id)
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_options: dict[str, Any] = {
                "max_new_tokens": min(
                    max_tokens or settings.generation_max_tokens, 700
                ),
                "do_sample": False,
                "repetition_penalty": 1.15,
                "streamer": streamer,
            }
            if eos_ids:
                generation_options["eos_token_id"] = eos_ids
            if isinstance(eos_token_id, int):
                generation_options["pad_token_id"] = eos_token_id
        except RAGError:
            raise
        except Exception as exc:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": None,
            }
            raise RAGError(
                "qwen_generation_failed",
                f"Astro Qwen local streaming setup failed: {exc}",
                502,
            ) from exc

        errors: list[Exception] = []

        def worker() -> None:
            try:
                with self._generation_lock, torch.inference_mode():
                    self._model.generate(**inputs, **generation_options)
            except Exception as exc:
                errors.append(exc)
                try:
                    streamer.end()
                except Exception as stream_error:
                    errors.append(stream_error)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        raw_chunks: list[str] = []
        emitted_visible = ""
        for chunk in streamer:
            raw_chunks.append(chunk)
            current_raw = "".join(raw_chunks)
            if (
                re.search(r"<think\b", current_raw, flags=re.IGNORECASE)
                and not re.search(r"</think\s*>", current_raw, flags=re.IGNORECASE)
            ) or (
                re.search(r"<tool_call\b", current_raw, flags=re.IGNORECASE)
                and not re.search(
                    r"</tool_call\s*>", current_raw, flags=re.IGNORECASE
                )
            ):
                continue
            safe_raw, _ = _stream_visible_text(current_raw)
            cleaned = _clean_model_output(safe_raw)
            if len(cleaned) > len(emitted_visible):
                delta = cleaned[len(emitted_visible):]
                emitted_visible = cleaned
                yield delta

        thread.join()
        if errors:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": None,
            }
            raise RAGError(
                "qwen_generation_failed",
                f"Astro Qwen local generation failed: {errors[0]}",
                502,
            ) from errors[0]

        safe_raw, _ = _stream_visible_text(
            "".join(raw_chunks),
            final=True,
        )
        answer = _clean_model_output(safe_raw)
        if len(answer) > len(emitted_visible):
            yield answer[len(emitted_visible):]
        if not answer:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": "no_visible_output",
                "usage": None,
            }
            raise RAGError(
                "qwen_empty_output",
                "Astro Qwen returned no visible answer.",
                502,
            )
        self.last_diagnostics = {
            "status": "completed",
            "incomplete_reason": None,
            "usage": None,
        }


_client: QwenLocalClient | None = None
_client_lock = Lock()


def get_qwen_client() -> QwenLocalClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = QwenLocalClient()
    return _client
