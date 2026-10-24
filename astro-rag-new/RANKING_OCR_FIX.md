# Astro RAG - Ranking Precision + Sanskrit Unicode Fix

## What was fixed

### 1. Ranking precision

The previous metadata index treated a chunk as a match when the requested planet and house appeared anywhere in the same chunk. That creates false positives such as:

- query: `Mars in the 10th house`
- retrieved rule: `5th Lord in the 10th House`
- Mars was mentioned later in the example, so old metadata incorrectly looked like `Mars + 10th House`.

The new corpus records explicit relationship fields:

```json
"planet_house_pairs": ["Mars|10"]
"planet_house_example_pairs": ["Mars|10"]
"planet_sign_pairs": ["Sun|Aries"]
"planet_nakshatra_pairs": ["Moon|Ashwini"]
"planet_aspect_pairs": ["Mars|4th"]
"lord_house_pairs": ["5|10", "10|10"]
```

Hybrid retrieval now uses the explicit relation first and only falls back to loose entity co-occurrence for recall. Explicit relation hits receive the strongest ranking component. The corpus builder also indexes own-house lord placements (for example, `10|10`), using house-token spans and contextual guards to avoid linking a house lord to a later, unrelated planet placement in the same source sentence.

For `/ask`, chart-backed scoring also checks all indexed planet/house and house-lord/house conditions on direct candidates against the supplied chart. A compound rule is discounted for conditions whose relevant placement is present in the chart and demonstrably differs from the indexed requirement; a condition cannot be penalized if its chart placement is unavailable. This prevents a single matching relation from making a partially satisfied compound rule rank like a fully supported rule. `/retrieve` without a full chart keeps its existing explicit-relation behavior.

House-lord relation scoring considers other `lord_house_pairs` for house-lord queries, not `planet_house_pairs` that may have been extracted from a commentary/example. When the actual chart identifies H1's lord placement, Ascendant-lord-in-house BPHS Ch. 26 slokas mislabeled by a Karaka example receive an additional discount if that condition does not hold. Known conception/progeny-only chapters are down-ranked for questions outside children/family/marriage topics. When more than one chart fact has candidates, no individual fact can occupy more than two slots before deferred candidates are used to fill the final result.

This keeps `Mars in the 10th house` separate from `5th lord in the 10th house`.

The generation prompt now provides available planet placements and house lordships alongside the chart's house map. Matching house-lord rules receive a `[Chart Binding: ...]` annotation that names the lord planet and distinguishes it from the destination-house lord. If passage metadata contradicts a supplied placement, a chart-verification note is appended; both note types are retained by NPU prompt compaction.

Before `/ask` or the final `/ask/stream` event is returned, generated planet placement claims are repaired against available chart data, and a planet's house-lord list is repaired only if all 12 houses are present. Unsupported yoga names are replaced unless named in retrieved text or in the standard-name allowlist. This is deterministic post-generation grounding, not general semantic validation; streaming deltas can precede the repaired final answer.

### 2. Sanskrit / Devanagari Unicode

The corpus builder now applies:

- Unicode NFC normalization
- removal of control characters / soft hyphen noise
- defensive UTF-8 mojibake repair if an upstream extractor produces common Latin-1/UTF-8 corruption

The four supplied PDFs already contain a usable Unicode Devanagari text layer when extracted with PyMuPDF. The new build preserves that text and verifies that common mojibake markers are absent.

### 3. API JSON charset

JSON responses now explicitly advertise:

```text
application/json; charset=utf-8
```

This avoids ambiguity for clients that inspect the HTTP content type.

If Devanagari is still displayed as `à¤...` only in Windows PowerShell, configure the terminal for UTF-8; the corpus itself is checked independently with `scripts/check_unicode.py`.

## Rebuild / populate the fix

From the project root:

```bash
python scripts/build_corpus.py
python scripts/check_unicode.py
python scripts/verify_corpus.py
python -m pytest -q tests/test_facts_parser.py tests/test_metadata_search.py tests/test_hybrid_retrieval.py
```

Then restart Astro RAG:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Verify the fix

Run:

```bash
python rag/evaluation/evaluate_retrieval.py --strategy hybrid
```

The corpus-specific smoke set is now generated from the explicit relationship fields. The raw-question semantic channel uses a lower offset and scale than the initial ranking fix, and redundant section overviews are omitted when a same-book, same-chapter sloka is already among the leading results. The current 80-query smoke run reports:

```text
planet_house Hit@5 = 96.4%
overall      Hit@5 = 85.0%
```

This is a corpus-specific smoke benchmark, not a scholarly relevance benchmark.

For the important regression case:

```text
What does Mars in the 10th house mean?
```

the top result is:

```text
phaladeepika_ch16_sl028
```

and it contains the explicit statement that Sun or Mars occupy the 10th house.

For:

```text
What does the 5th lord in the 10th house mean?
```

the top result is:

```text
bphs_ch26_sl058
```

which contains the 5th-Lord-in-10th rule.

## PowerShell UTF-8 display

Before reading JSON directly in PowerShell:

```powershell
chcp 65001
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

Or use the browser UI / `curl.exe`.
