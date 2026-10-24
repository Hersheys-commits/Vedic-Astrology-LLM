#!/usr/bin/env python3
"""Generate a corpus-specific retrieval smoke set from the actual OCR corpus.

This is intentionally a *smoke* set, not a scholarly relevance benchmark.
Each expected chunk is a real chunk from the supplied PDFs whose metadata
matches the query fact. It is regenerated whenever the corpus is rebuilt.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.corpus_store import reload_store  # noqa: E402

OUT_PATH = PROJECT_ROOT / "rag" / "evaluation" / "retrieval_gold.jsonl"
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th", 11: "11th", 12: "12th"}
PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
NAKS = ["Ashwini", "Ardra", "Rohini", "Pushya", "Ashlesha", "Magha", "Chitra", "Swati", "Mula", "Revati", "Vishakha", "Anuradha"]
ASPECTS = ["3rd", "4th", "5th", "7th", "8th", "9th", "10th"]


def ranked_candidate_ids(store, filt: dict, limit: int = 3) -> list[str]:
    ids = list(store.metadata_candidate_ids(filt))
    # Prefer primary rule chunks over overview/commentary children for smoke
    # cases; the latter remain valid retrieval targets in production.
    def key(cid: str):
        c = store.chunk(cid) or {}
        m = c.get("metadata", {})
        kind = m.get("chunk_type")
        priority = {"rule": 0, "numbered_rule": 1, "commentary": 2, "section_overview": 3, "prose": 4}.get(kind, 9)
        return (priority, len(c.get("text", "")), cid)
    return [cid for cid in sorted(ids, key=key)[:limit]]


def main() -> None:
    store = reload_store()
    tests: list[dict] = []

    def add(category: str, query: str, fact: dict, expected: list[str]):
        if not expected or len(tests) >= 80:
            return
        tests.append({
            "id": f"smoke_{len(tests) + 1:03d}",
            "category": category,
            "query": query,
            "chart_facts": fact,
            "expected_chunks": expected,
        })

    # Use combinations that actually exist in this corpus rather than a fixed
    # matrix that may contain no matching classical passage.
    for planet in PLANETS:
        for house in range(1, 13):
            fact = {"type": "planet_house", "planet": planet, "house": house}
            expected = ranked_candidate_ids(store, {"planet": planet, "house": house})
            add("planet_house", f"What is the effect of {planet} in the {ORD[house]} house?", fact, expected)
            if len(tests) >= 24:
                break
        if len(tests) >= 24:
            break

    for planet in PLANETS:
        for sign in SIGNS:
            fact = {"type": "planet_sign", "planet": planet, "sign": sign}
            expected = ranked_candidate_ids(store, {"planet": planet, "sign": sign})
            add("planet_sign", f"What does {planet} in {sign} indicate?", fact, expected)
            if len([t for t in tests if t["category"] == "planet_sign"]) >= 12:
                break
        if len([t for t in tests if t["category"] == "planet_sign"]) >= 12:
            break

    for nak in NAKS:
        fact = {"type": "nakshatra", "nakshatra": nak}
        expected = ranked_candidate_ids(store, {"nakshatra": nak})
        add("nakshatra", f"What does {nak} nakshatra signify?", fact, expected)
        if len([t for t in tests if t["category"] == "nakshatra"]) >= 12:
            break

    for planet in PLANETS:
        for aspect in ASPECTS:
            fact = {"type": "aspect", "planet": planet, "aspect_type": aspect}
            expected = ranked_candidate_ids(store, {"planet": planet, "aspect_type": aspect})
            add("aspect", f"What is the significance of {planet}'s {aspect} aspect?", fact, expected)
            if len([t for t in tests if t["category"] == "aspect"]) >= 8:
                break
        if len([t for t in tests if t["category"] == "aspect"]) >= 8:
            break

    for planet in PLANETS:
        fact = {"type": "dasha", "dasha_lord": planet}
        expected = ranked_candidate_ids(store, {"dasha_lord": planet})
        add("dasha", f"What does {planet} Mahadasha indicate?", fact, expected)

    # Variations test the fact-normalized semantic leg against the same real
    # fact, while still preserving exact metadata as the candidate constraint.
    variations = [
        "Mars in the 10th house",
        "What does Mars in the tenth house mean?",
        "Mars placed in the tenth bhava",
        "Significance of Mars occupying the 10th house",
        "How is Mars interpreted when placed in the 10th?",
    ]
    expected = ranked_candidate_ids(store, {"planet": "Mars", "house": 10})
    for q in variations:
        add("semantic_variation", q, {"type": "planet_house", "planet": "Mars", "house": 10}, expected)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for t in tests:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for t in tests:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    print(f"Wrote {len(tests)} corpus-specific retrieval smoke cases -> {OUT_PATH}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
