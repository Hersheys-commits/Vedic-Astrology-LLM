# Project Progress

## Local answer-model progress (2026-10-22)

Astro RAG now exposes an optional local response-generation choice using the fine-tuned `XeroxFlynn/astro-rag-lora-2` adapter and its 4-bit `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` base. GPT-5-mini remains the default, and question-intent extraction continues to use the configured OpenAI client. The model and adapter are loaded lazily; optional local dependencies are isolated in `astro-rag-new/requirements-local-model.txt`.

The local loader uses low-memory Transformers/PEFT loading. On CPU-only systems it assigns the model wholly to the bitsandbytes CPU backend; CUDA systems keep automatic placement. The former fixed 8 GB free-RAM application gate was removed. This addresses the Transformers error caused by automatic mixed CPU/disk placement for this 4-bit model, but it does not lower the model's actual runtime memory requirement.

Validation so far: a small bitsandbytes 4-bit CPU forward pass succeeds on the local Intel CPU. Full Transformers CPU generation remains unverified. The separate OpenVINO NPU provider has returned visible output in controlled tests and in the frontend; the configured 2,048-token generation ceiling was not yet validated end-to-end at its maximum when this status was recorded. The planned project-trained ~1B model remains future work.

## OpenVINO NPU generation and corpus indexing (2026-10-22)

Controlled NPU comparisons found that the app's composed plain-text prompt returned empty output with OpenVINO's native chat-template wrapping enabled and visible output when that wrapping was disabled. The NPU client now explicitly disables it, uses repetition and stop-string controls, and marks output incomplete when generation reaches the configured limit. `OPENVINO_NPU_MAX_NEW_TOKENS` defaults to 2,048 and also configures OpenVINO's response-context reservation; it is a ceiling, not a requirement to emit 2,048 tokens. The prompt limit remains 1,024, so validate memory use, latency, and answer completeness on target hardware before increasing either reservation.

The answer prompt now includes an available 12-house map, labels missing house data instead of inferring it, and tightens instructions for passage conditions and irrelevant evidence. Local Transformers generation is greedy with a 700-token cap, repetition penalty, Qwen end-token stopping when available, and limited cleanup of internal tags/echoed passage blocks. These cleanups are not semantic claim or citation validation.

The corpus builder now indexes own-house lord relations while using house-token spans and contextual checks to avoid associating a lord with unrelated later placements. The processed corpus was rebuilt and verified from all four PDFs (2,800 chunks); the known partial Jataka Parijata coverage remains. Hybrid retrieval score weights were reduced for raw-question semantic matches, and same-chapter overviews are suppressed when a leading sloka already represents that chapter. The 80-case retrieval smoke evaluation remains at Hit@5 85%; it is a regression set, not an expert-adjudicated relevance benchmark. The current full Astro RAG test suite passes.

## End-to-end evaluation work (2026-10-21)

An initial 20-case evaluation set is in `astro-rag-new/rag/evaluation/end_to_end_gold.jsonl`. It uses distinct synthetic birth dates, times and locations, pinned calculation settings and a fixed `as_of_date`; 12 cases are development and 8 are holdout. The questions cover placements, life areas, source rules, dasha, unsupported corpus coverage and a medical-safety boundary. Gold passage IDs and metadata were checked against the processed corpus, with representative cited pages visually reviewed.

The PDF audit verified that all four PDFs match the manifest SHA-256 values. Across 2,003 pages, the extracted text layer is blank on 9 Brihat Jataka pages and 11 Phaladeepika pages; BPHS and Jataka Parijata have no blank pages. Reviewed blank pages were genuinely blank in the PDF. Some inventory differences are pages intentionally excluded from main text; Jataka Parijata has known partial detected coverage. The audit does not prove OCR semantic correctness.

Retrieval-only Hit@5 was 22.2% in the first run, then rose to 44.4% after parser/label fixes and a larger per-fact candidate pool. After raw-question ranking, exact nakshatra metadata scoring, and two source-confirmed alternative evidence labels, it reached 66.7% overall: 83.3% on development and 33.3% across the six evidence-bearing holdout cases. The existing 80-query retrieval smoke benchmark remains at Hit@5 85% and Recall@5 70.83%; these suites use different queries and labels and are not expert-adjudicated answer-quality scores.

The first full development baseline completed with generation model `gpt-5-mini` and temporary judge model `gpt-5.1`: judge means were 4.67 chart consistency, 4.08 evidence support, 4.50 citation validity, 4.67 relevance and 4.58 uncertainty/safety; 8/12 answers passed all current case checks. A post-fix development run passed 9/12, including improved career-question fact handling. It exposed loose citations and overextended comparison of adjacent aspect rules; citation and calculation prompt guidance was added afterwards. A consistent full development and holdout answer run using that latest prompt remains pending.

Chart Engine tests pass (42); Astro RAG's offline tests also pass. The private generation configuration is available, while `EVAL_JUDGE_MODEL` is blank in the default environment and must be set for full evaluation. Retrieval-only evaluation runs with both services online and makes no model calls.

## Browser chat UI (2026-10-22)

The Astro RAG test form has been replaced by a responsive chatbot interface served from the existing FastAPI static directory. First use requests birth date, local birth time and birthplace, with explicit place search and optional Leaflet map/coordinate selection. A returning browser opens the chat directly with the saved profile in the sidebar. Profile edits affect future questions while existing messages retain the profile snapshot used for their answer.

The client keeps chart/retrieval controls and saved-data management in a collapsed **Advanced settings** section, supports system/light/dark appearance, and stores its profile, settings, theme and transcript in versioned browser `localStorage`. Astro RAG now has a same-origin geocoding proxy with an in-process one-hour cache (128 entries) and serialized upstream requests. Backend tests cover caching, errors, UI asset serving and `/ask` request compatibility. Browser automation was unavailable in the development environment, so interactive visual checks remain to be completed in a browser.

## Vedic Astrology Question Answering System

### Project overview

This project explores how a language model can answer questions about a person's Vedic astrology chart while using both calculated chart information and passages from traditional astrology texts. It is currently implemented as two separate services: one calculates the chart, and the other retrieves textual evidence and prepares an answer.

The separation is intentional. Chart calculations are handled by deterministic software so the model does not need to invent planetary positions or house placements. The language model is used for understanding questions and explaining retrieved material in accessible language.

### Work completed so far

#### 1. Deterministic chart calculation service

The **Chart Engine** accepts a birth date, local birth time, and geographic coordinates. It resolves the timezone for the location, converts local time to a calculation time, and uses Swiss Ephemeris calculations to produce structured chart data.

The output includes planetary positions, ascendant and houses, nakshatras and padas, planetary aspects, and Vimshottari dasha periods. It also records calculation settings, timezone and ephemeris information, warnings, and internal consistency checks. This makes the chart result inspectable and helps identify which settings were used.

The service is available through a web API and a command-line interface. It also has location helper endpoints and a small browser interface for development.

#### 2. Question answering and retrieval service

The **Astro RAG** service accepts a user's question together with birth details, or it can accept chart JSON that has already been calculated. When given birth details, it requests the chart from Chart Engine.

Astro RAG then identifies relevant chart facts. It currently supports a deterministic, rule-based question parser and an optional language-model intent step. Proposed intents are checked against supported values and expanded using the chart data before retrieval.

The retrieval component searches for passages related to both the question and relevant chart facts. It combines structured metadata, such as planet-house relationships, with TF-IDF text ranking. Retrieved passages retain their book and page-related metadata so they can be inspected alongside the answer.

GPT-5-mini/OpenAI remains the default answer provider; deterministic mock mode allows the data flow to run without a configured key, and the optional local Qwen3-4B LoRA backend can be selected for final answer generation. Intent extraction remains on the configured OpenAI client. The API response includes the answer, facts considered, retrieved passages, chart summary, selected model and generation diagnostics.

#### 3. Source corpus and processing

Four OCR book editions are included in the current corpus: Brihat Parasara Hora Sastra, Brihat Jataka, Jataka Parijata, and Phaladeepika. A corpus-building utility extracts text, identifies structure such as chapters and numbered rules where possible, and creates searchable chunks with source metadata.

The current processed corpus contains 2,800 chunks. Jataka Parijata coverage is incomplete: the supplied OCR currently has detected main-text coverage for Adhyaya 10–18, and the missing coverage is recorded. OCR and structure detection can affect retrieval quality, so the chunk count should not be interpreted as complete or error-free coverage of the books.

#### 4. Developer and evaluation utilities

Both services include API documentation. Chart Engine retains a development chart console, while Astro RAG now provides the persistent chat UI described below. The repositories also contain tests for chart calculations and Astro RAG components, tools for rebuilding and checking the corpus, and scripts for comparing retrieval strategies.

These evaluation utilities provide an initial way to catch regressions. The existing retrieval cases are not yet a comprehensive, expert-adjudicated benchmark of answer quality.

### Current end-to-end flow

```text
Birth details + question
        ↓
Chart Engine calculates structured chart JSON
        ↓
Astro RAG extracts relevant chart facts
        ↓
Relevant passages are retrieved from the four-book corpus
        ↓
The generation model prepares an answer from facts and evidence
        ↓
Answer is returned with chart facts and retrieved passages
```

### Current status and limitations

The core prototype connects chart calculation, question understanding, text retrieval, and language-model generation. It is suitable for development and further research, but it is not yet a fully validated public product.

Key areas still needing work include broader expert review of chart calculations and interpretations, improved OCR and corpus coverage, human-reviewed retrieval and answer benchmarks, and checks that verify the generated answer's claims and citations. The generation prompt asks the model to ground its answer and communicate uncertainty, but prompt instructions alone cannot guarantee that it will always do so. The project also does not yet include the planned custom small language model.
