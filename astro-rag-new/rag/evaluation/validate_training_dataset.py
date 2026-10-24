"""Validate user-facing training targets and their corpus references."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "rag" / "processed" / "corpus.jsonl"

REQUIRED_FIELDS = {
    "id",
    "domain",
    "chart_facts",
    "retrieved_context",
    "question",
    "final_answer",
}
INTERNAL_TARGET_RE = re.compile(
    r"\*\*\s*(?:Reasoning|Chart Factors Examined|Internal Verification)\s*:|"
    r"<(?:think|thought_process)\b|"
    r"\b(?:chain of thought|per prompt rule|per architecture\.md|per prd\.md)\b",
    re.IGNORECASE,
)
PLANET_RE = re.compile(
    r"\b(?:Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\b",
    re.IGNORECASE,
)
HOUSE_RE = re.compile(r"\b(?:\d{1,2}(?:st|nd|rd|th)\s+house|H\d{1,2})\b", re.IGNORECASE)
PLACEMENT_RE = re.compile(
    r"\b(?P<planet>Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\b"
    r"[^.!?\n]{0,50}?\b(?:placed\s+in|occupies|is\s+in|sits\s+in)\s+"
    r"(?:the\s+)?(?P<house>\d{1,2})(?:st|nd|rd|th)\s+house\b",
    re.IGNORECASE,
)
CONDITIONAL_RE = re.compile(r"^\s*(?:if|when|suppose|for example)\b", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _source_key(value: str) -> str:
    normalized = _normalize_text(value)
    aliases = {
        "bphs": "brihat parasara hora sastra",
        "brihat parashara hora shastra": "brihat parasara hora sastra",
        "brihat parasara": "brihat parasara hora sastra",
        "jataka parijatha": "jataka parijata",
        "phaladeepika": "phaladeepika",
    }
    for alias, canonical in aliases.items():
        if alias in normalized:
            return canonical
    return normalized


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", _normalize_text(left)))
    right_tokens = set(re.findall(r"[a-z0-9]+", _normalize_text(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid corpus JSON: {exc}") from exc
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"{path}:{line_number}: corpus chunk_id is missing")
            chunks[chunk_id] = chunk
    return chunks


def _fact_placements(chart_facts: list[Any]) -> tuple[dict[str, int], dict[int, str]]:
    planet_houses: dict[str, int] = {}
    house_lords: dict[int, str] = {}
    for fact in chart_facts:
        if not isinstance(fact, str):
            continue
        planet_match = re.search(
            r"\b(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\s*:"
            r"\s*(?:Placed\s+)?in\s+(\d{1,2})(?:st|nd|rd|th)\s+house\b",
            fact,
            re.IGNORECASE,
        )
        if planet_match:
            planet_houses[planet_match.group(1).casefold()] = int(planet_match.group(2))
        lord_match = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)\s+house\b[^.;]*?\b(?:ruled by|lord is)\s+"
            r"(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\b",
            fact,
            re.IGNORECASE,
        )
        if lord_match:
            house_lords[int(lord_match.group(1))] = lord_match.group(2)
    return planet_houses, house_lords


def _check_explicit_chart_claims(
    answer: str,
    chart_facts: list[Any],
) -> list[str]:
    planet_houses, _ = _fact_placements(chart_facts)
    issues = []
    for match in PLACEMENT_RE.finditer(answer):
        sentence_start = max(
            answer.rfind(".", 0, match.start()),
            answer.rfind("?", 0, match.start()),
            answer.rfind("!", 0, match.start()),
            answer.rfind("\n", 0, match.start()),
        ) + 1
        sentence = answer[sentence_start:match.start()]
        if CONDITIONAL_RE.search(sentence):
            continue
        actual = planet_houses.get(match.group("planet").casefold())
        claimed = int(match.group("house"))
        if actual is not None and actual != claimed:
            issues.append(
                f"{match.group('planet')} is stated in house {claimed}, "
                f"but chart_facts place it in house {actual}"
            )
    return issues


def validate_dataset(
    examples: Any,
    corpus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(examples, list):
        return {
            "example_count": 0,
            "errors": ["dataset root must be a JSON array"],
            "warnings": [],
            "class_distribution": {},
            "split_validation": "not available in this dataset schema",
        }

    ids: Counter[str] = Counter()
    questions: dict[str, list[str]] = defaultdict(list)
    prompts: dict[str, list[str]] = defaultdict(list)
    targets: dict[str, list[str]] = defaultdict(list)
    class_distribution: Counter[str] = Counter()
    placement_issues: list[dict[str, Any]] = []
    passage_issues: list[dict[str, Any]] = []
    reasoning_leaks: list[str] = []
    unrelated_mentions: list[dict[str, Any]] = []

    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            errors.append(f"example {index}: expected an object")
            continue
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            errors.append(f"example {index}: missing or invalid id")
            example_id = f"<example {index}>"
        else:
            ids[example_id] += 1

        missing = sorted(REQUIRED_FIELDS - set(example))
        if missing:
            errors.append(f"{example_id}: missing required fields {missing}")

        question = example.get("question")
        answer = example.get("final_answer")
        chart_facts = example.get("chart_facts")
        context = example.get("retrieved_context")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{example_id}: missing user question")
            question = ""
        if not isinstance(answer, str) or not answer.strip():
            errors.append(f"{example_id}: missing user-facing final_answer")
            answer = ""
        if not isinstance(chart_facts, list):
            errors.append(f"{example_id}: chart_facts must be an array")
            chart_facts = []
        if not isinstance(context, list):
            errors.append(f"{example_id}: retrieved_context must be an array")
            context = []

        category = str(example.get("domain") or "uncategorized")
        class_distribution[category] += 1
        normalized_question = _normalize_text(question)
        questions[normalized_question].append(example_id)
        prompt_key = json.dumps(
            [normalized_question, chart_facts],
            sort_keys=True,
            ensure_ascii=False,
        )
        prompts[prompt_key].append(example_id)
        targets[_normalize_text(answer)].append(example_id)

        if answer and INTERNAL_TARGET_RE.search(answer):
            reasoning_leaks.append(example_id)
        if answer:
            mismatches = _check_explicit_chart_claims(answer, chart_facts)
            if mismatches:
                placement_issues.append(
                    {"id": example_id, "issues": mismatches}
                )

        question_planets = {
            match.group(0).casefold() for match in PLANET_RE.finditer(question)
        }
        fact_planets = {
            match.group(0).casefold()
            for fact in chart_facts
            if isinstance(fact, str)
            for match in PLANET_RE.finditer(fact)
        }
        answer_planets = {
            match.group(0).casefold() for match in PLANET_RE.finditer(answer)
        }
        extra_planets = sorted(answer_planets - question_planets - fact_planets)
        if extra_planets:
            unrelated_mentions.append(
                {"id": example_id, "possible_unrelated_planets": extra_planets}
            )

        for passage in context:
            if not isinstance(passage, dict):
                passage_issues.append(
                    {"id": example_id, "issue": "retrieved_context entry is not an object"}
                )
                continue
            chunk_id = passage.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                passage_issues.append(
                    {"id": example_id, "issue": "retrieved passage is missing chunk_id"}
                )
                continue
            source_chunk = corpus.get(chunk_id)
            if source_chunk is None:
                passage_issues.append(
                    {"id": example_id, "chunk_id": chunk_id, "issue": "chunk_id not found"}
                )
                continue
            target_source = passage.get("source")
            metadata = source_chunk.get("metadata") or {}
            source_meta = metadata.get("source") or {}
            actual_source = (
                source_meta.get("book")
                if isinstance(source_meta, dict)
                else source_meta
            ) or metadata.get("book_id")
            if (
                target_source
                and actual_source
                and _source_key(str(actual_source)) not in _source_key(str(target_source))
                and _source_key(str(target_source)) not in _source_key(str(actual_source))
            ):
                passage_issues.append(
                    {
                        "id": example_id,
                        "chunk_id": chunk_id,
                        "issue": "source label does not match corpus metadata",
                    }
                )
            context_text = passage.get("text")
            corpus_text = source_chunk.get("text")
            if (
                isinstance(context_text, str)
                and isinstance(corpus_text, str)
                and context_text.strip()
                and _token_overlap(context_text, corpus_text) < 0.2
            ):
                warnings.append(
                    f"{example_id}: retrieved passage text for {chunk_id} has "
                    "low token overlap with the current corpus"
                )

        messages = example.get("messages")
        if messages is not None:
            if not isinstance(messages, list) or not all(
                isinstance(message, dict)
                and message.get("role") in {"system", "user", "assistant"}
                and isinstance(message.get("content"), str)
                for message in messages
            ):
                errors.append(f"{example_id}: malformed training messages")

    for example_id, count in ids.items():
        if count > 1:
            errors.append(f"duplicated id {example_id} ({count} occurrences)")
    duplicate_questions = [
        {"question": key, "ids": value}
        for key, value in questions.items()
        if key and len(value) > 1
    ]
    duplicate_prompts = [
        {"ids": value}
        for value in prompts.values()
        if len(value) > 1
    ]
    duplicate_targets = [
        {"ids": value}
        for key, value in targets.items()
        if key and len(value) > 1
    ]

    if reasoning_leaks:
        errors.append(
            f"{len(reasoning_leaks)} final_answer field(s) contain internal reasoning "
            "or meta-prompt text"
        )
    if passage_issues:
        errors.append(f"{len(passage_issues)} retrieved_context reference issue(s)")
    if placement_issues:
        warnings.append(
            f"{len(placement_issues)} possible chart-fact/answer placement mismatch(es)"
        )
    if duplicate_questions:
        warnings.append(f"{len(duplicate_questions)} duplicate question(s)")
    if duplicate_prompts:
        warnings.append(f"{len(duplicate_prompts)} duplicate question/chart-fact prompt(s)")
    if duplicate_targets:
        warnings.append(f"{len(duplicate_targets)} duplicate final-answer target(s)")

    return {
        "example_count": len(examples),
        "unique_id_count": len(ids),
        "missing_id_count": sum(
            1 for example in examples
            if not isinstance(example, dict)
            or not isinstance(example.get("id"), str)
            or not example.get("id", "").strip()
        ),
        "duplicate_ids": sorted(
            example_id for example_id, count in ids.items() if count > 1
        ),
        "duplicate_questions": duplicate_questions,
        "duplicate_question_chart_fact_prompts": duplicate_prompts,
        "duplicate_targets": duplicate_targets,
        "invalid_passage_references": passage_issues,
        "possible_chart_fact_mismatches": placement_issues,
        "internal_reasoning_leaks": reasoning_leaks,
        "possible_unrelated_planet_mentions": unrelated_mentions,
        "class_distribution": dict(sorted(class_distribution.items())),
        "split_validation": "not available in this dataset schema",
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Training JSON array to validate.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Corpus JSONL used to validate passage IDs (default: {DEFAULT_CORPUS}).",
    )
    args = parser.parse_args()

    try:
        examples = json.loads(args.dataset.read_text(encoding="utf-8"))
        corpus = _load_corpus(args.corpus)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"errors": [str(exc)]}, ensure_ascii=False, indent=2))
        return 2

    report = validate_dataset(examples, corpus)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
