# Astro RAG — Project Guide

This copy of Astro RAG is configured around the **four actual OCR PDFs supplied for the project**. The chart engine remains a separate service.

## 1. Service boundary

```text
                    birth details
                         |
                         v
                  CHART ENGINE :8000
                         |
                     chart JSON
                         |
                         v
                    ASTRO RAG :8001
                         |
             +-----------+-----------+
             |                       |
        fact extraction         question text
             |                       |
             +-----------+-----------+
                         |
                 hybrid retrieval
                         |
                 evidence bundle
                         |
                         v
          selected answer provider
          OpenAI / mock / local Qwen LoRA / Intel NPU
                         |
                         v
                      answer
```

Chart Engine is responsible for deterministic chart construction. Astro RAG is responsible for finding what the source books say and synthesizing an answer from those source passages. This separation is also the basis of the supplied architecture plan.

## 2. Real source corpus

The repository contains these four PDFs under `rag/source_pdfs/`:

```text
brihat hora sastra OCR(1).pdf
Brihat Jataka 2nd Ed. by V Subrahmanya Sastri_OCR(1).pdf
Jataka Parijata_OCR(1).pdf
Mantreswaras-Phaladeepika_OCR(1).pdf
```

The source editions are identifiable from their scanned title pages and OCR. BPHS is a translation/commentary volume by Girish Chand Sharma; Brihat Jataka is the second edition with English translation and explanatory notes/examples; Phaladeepika is the translated work of Mantreswara.

### Important Jataka Parijata limitation

The uploaded OCR does not provide the complete work. The current parser detects the main text from Adhyaya 10 through Adhyaya 18 and records the omitted chapter coverage in `corpus_manifest.json`. Treat this as a versioned partial source, not a complete Jataka Parijata corpus.

## 3. Corpus hierarchy

The ingestion pipeline is structure-aware:

```text
Book
└── Chapter / Adhyaya
    └── Section / topic
        └── Rule / Sloka
            ├── translation/source text
            ├── commentary child chunks
            └── examples / long continuation
```

This is preferable to a generic fixed-size splitter for these books because the supplied plan explicitly calls for semantic chapter/sloka boundaries.

The builder keeps page provenance and separates the canonical OCR from its retrieval-clean representation.

## 4. Ingestion files

`scripts/build_corpus.py`

Reads the four PDFs with PyMuPDF and creates:

```text
rag/processed/pages.jsonl
rag/processed/corpus.jsonl
rag/processed/corpus_manifest.json
rag/evaluation/retrieval_gold.jsonl
```

Current build:

```text
Brihat Parasara Hora Sastra : 1,114 chunks / 47 chapters
Brihat Jataka                 :   454 chunks / 28 chapters
Jataka Parijata               :   568 chunks /  9 detected chapters (10-18)
Phaladeepika                  :   664 chunks / 28 chapters
TOTAL                         : 2,800 chunks
```

Chunk types are:

```text
rule
numbered_rule
commentary
section_overview
prose
```

Rule chunks are the primary retrieval units. Commentary becomes a child chunk when a rule segment is too long. This follows the supplied chunking plan rather than using blind overlap windows.

## 5. What is stored per chunk

A chunk has the shape:

```json
{
  "chunk_id": "bphs_ch20_sl024",
  "text": "...source-derived OCR/translation...",
  "retrieval_text": "Source + chapter + sloka + indexed context + text",
  "metadata": {
    "book_id": "bphs",
    "chapter_no": 20,
    "sloka_start": 24,
    "chunk_type": "rule",
    "planets": [],
    "houses": [],
    "signs": [],
    "nakshatras": [],
    "dasha_lords": [],
    "lord_of_houses": [],
    "planet_house_pairs": [],
    "lord_house_pairs": [],
    "topics": [],
    "relations": [],
    "source": {
      "pdf_page_start": 255,
      "pdf_page_end": 255,
      "printed_page_start": null,
      "printed_page_end": null,
      "pdf_sha256": "..."
    }
  }
}
```

`text` remains source-derived. `retrieval_text` is contextualized for search. The source plan specifically recommends separate raw, cleaned, and retrieval representations.

## 6. Retrieval flow

`app/facts/parser.py` first converts the user question and chart into concrete facts.

Examples:

```text
"Mars in 10th house"
 -> planet_house(Mars, 10)

"my career"
 -> house 10 / house 6 related chart facts

"when will I get married"
 -> marriage-related house facts + current Mahadasha

"how is my personality"
 -> 1st-house / placement facts
```

Timing questions explicitly add the running Mahadasha so the temporal part of the question has chart context.

`app/retrieval/hybrid.py` then:

```text
exact metadata candidate set
        |
        v
semantic ranking over those candidates
        |
        +---- direct semantic search over the question
        |
        +---- small life-area/topic boost
        +---- remove redundant same-chapter overviews when slokas are present
        v
merged + de-duplicated passages
```

This is deliberately source-first: retrieved doctrine is never generated during ingestion. Topic expansion prioritizes specific domain terms over generic helper wording and supports up to four topic houses when the question has no explicit planet/house constraint. Topic reranking discounts selected conception/progeny-specific chapters for unrelated questions, house-lord matches are not penalized for example planet metadata, and final passage merging limits a single fact to two of the first-pass slots when alternatives are available.

## 7. LLM generation

`app/generation/response_generator.py` routes final answer generation to GPT-5-mini/OpenAI by default, deterministic mock mode when no client is configured, or the optional local Qwen3-4B LoRA adapter / Intel NPU provider when selected. GPT requests use the configured intent mode. Local Qwen and NPU requests default to deterministic rules-based intent extraction so repeated questions resolve to the same chart facts; an explicit request `intent_mode` overrides that default. When LLM intent is combined with topic rules, deterministic rule facts are ordered first.

The local Transformers model uses the adapter's 4-bit bitsandbytes base, loaded lazily with low-memory Transformers/PEFT settings. CPU-only systems use an explicit all-CPU device map to avoid an unsupported mixed CPU/disk map; CUDA systems use automatic placement. The loader has no fixed 8 GB RAM precheck, though actual hardware memory still determines whether inference succeeds. Full-model Transformers CPU generation should be profiled on target hardware before relying on that provider.

The separate Intel NPU provider uses an exported OpenVINO GenAI IR. It sends the composed plain prompt with native chat templating disabled, uses repetition and stop-string controls, and defaults to a maximum of 1,024 new tokens via `OPENVINO_NPU_MAX_NEW_TOKENS`. That setting also configures OpenVINO's `MIN_RESPONSE_LEN` response-context reservation; it does not force the model to generate that many tokens. The default prompt limit is 1,536 tokens, with a fitting budget of `MAX_PROMPT_LEN - 128` (1,408 tokens). A compact NPU-specific instruction set reduces fixed prompt overhead. When the full prompt exceeds budget, the fitter removes repeated chart phrasing and duplicate house occupants while retaining chart relations, shortens citation headers to passage number and book, filters short OCR fragments, and allocates body tokens by retrieval rank (5:4:2 for the first two and subsequent passages), redistributing unused space. If citation headers and their chart notes alone exceed the budget, it drops the lowest-ranked passages one at a time, keeping notes for all retained passages. The prompt limit plus the 1,024-token response reservation sets a context of about 2,560 tokens. The NPU path has produced visible output in controlled tests and the frontend, but answer quality and performance should still be evaluated on target hardware.

The generation prompt receives:

```text
chart snapshot
relevant chart facts
retrieved passages with source/chapter/sloka metadata
user question
```

The generation system prompt uses a three-level evidence contract:

```text
DIRECT TEXT           -> explicitly stated in a retrieved passage
TRADITIONAL SYNTHESIS -> cautious interpretation combining chart facts + retrieved themes
UNSUPPORTED DOCTRINE  -> rules/claims requiring missing chart or source information (forbidden)
```

Interpretations are allowed only when supplied chart facts and retrieved evidence jointly support the theme; pretrained astrological knowledge must not fill evidence gaps. The answer length is question-dependent, unavailable divisional-chart data must be acknowledged rather than invented, and generic astrologer disclaimers are omitted unless a limitation matters. Personality questions are instructed to give the overall reading first, followed by supporting chart/source points and a brief limitation when needed.

The prompt includes the available 12-house map, planet placements, and house lordships from the chart schema and labels missing data as unavailable rather than inferring it. A retrieved house-lord relation that holds in the chart receives a `[Chart Binding: ...]` note naming the actual ruling planet and, when needed, distinguishing it from the destination house's different lord. Retrieved rule metadata is also checked against known chart placements; contradicted planet-house and lord-house conditions receive explicit verification notes, while missing data is not treated as disproving a condition. Before generation, a retrieval guard checks for an exact indexed chart relation or a passage score of at least `0.45`; otherwise the prompt warns the model not to use pretrained knowledge to fill evidence gaps. The prompts also explicitly require checking placements and yoga conditions and prohibit invented technical labels.

Before returning the completed `/ask` response or the final `/ask/stream` events, a deterministic post-generation guard repairs chart-backed planet house/sign and explicit dignity claims when the corresponding chart data is available, repairs two-house lordship claims only when the supplied chart contains all twelve houses, and replaces unsupported named yoga labels. Hypothetical textbook rules are not rewritten. Known standard names and names found in retrieved text are retained. Streaming deltas can precede this final guard; the `replace` event explicitly replaces them with the validated answer, and `done` carries the complete response. `generation_usage.retrieval_signature` records the retrieved chunk IDs, scores, and match types so runs can be compared with the prompt hash and model fingerprint. These measures improve grounding but are not a general semantic or citation validator. Local generation clients remove leaked `<tool_call>` markers, known internal-thought tags, echoed passage blocks, and internal chart notes.

## 8. API endpoints

```text
GET  /health
GET  /corpus/stats
POST /retrieve
POST /ask
POST /ask/stream
```

`/retrieve` is the main debugging endpoint because it isolates corpus + retrieval from chart generation and the LLM. `/ask/stream` accepts the same request as `/ask` and streams metadata, visible answer deltas, a validated-answer `replace` event, and a final response as Server-Sent Events; `/ask` remains the non-streaming JSON endpoint.

## 9. Correct replacement workflow

When replacing the four PDFs:

```bash
# 1. replace source PDFs in rag/source_pdfs/
python scripts/build_corpus.py
python scripts/verify_corpus.py
python -m pytest -q

# 2. restart the API so the in-memory index is rebuilt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Do not hand-edit `corpus.jsonl` after generation. The source PDFs + `build_corpus.py` are the reproducible input path.

## 10. Why there is no `.env` in the zip

A local API key that was present in the uploaded project copy has been removed from this deliverable. Use `.env.example` to create a local `.env`, and rotate/revoke that previously exposed key before using the repository again.
