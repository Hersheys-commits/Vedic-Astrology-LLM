# Decisions and Rationale

This is an implementation record of decisions visible in the repository, followed by proposed decisions for the next phase. It does not claim that every choice has been benchmarked or independently verified.

## 1. Current implementation decisions

| Decision | Rationale visible in the design | Consequence / evidence in repository |
|---|---|---|
| Separate chart calculation from language generation | Chart arithmetic should be deterministic and independently debuggable; language generation should not invent positions. | `chart-engine/` is a standalone service; Astro RAG calls it via `app/chart_client/client.py`. The chart calculation pipeline contains no LLM call. |
| Return a structured, versioned chart contract | Downstream services need stable, inspectable values and calculation provenance. | Chart JSON includes schema/engine versions, settings, timezone, ephemeris provenance, validations, normalized fact keys/IDs and optional natural-language factual statements. |
| Resolve civil time using timezone rules, not a fixed offset | Local birth time must map to UTC using the location's timezone and date-specific rules; DST edge cases need explicit behavior. | Timezonefinder/IANA zoneinfo path, DST ambiguity/nonexistent-time errors, and optional first/second occurrence choice. |
| Keep tradition-dependent values in configuration | Calculation assumptions should be visible and changeable without scattering them through algorithms. | `chart-engine/app/config/` defines ayanamsa, house settings, nodes, nakshatras, dignities and planetary rules. Defaults are Lahiri, sidereal, whole-sign and mean nodes. |
| Use a source-derived corpus and preserve provenance | Readers and debugging tools should trace retrieved doctrine back to a book and page; ingestion should not silently rewrite sources. | Four supplied OCR PDFs feed `scripts/build_corpus.py`; outputs separate page material and retrieval corpus with hashes and provenance. |
| Segment by book structure rather than only fixed token windows | Astrological rules frequently depend on chapter/sloka context and should remain identifiable. | Builder detects books/chapters/rules/slokas and emits rule, numbered-rule, commentary and overview chunks. Detection quality still depends on OCR. |
| Start semantic ranking with TF-IDF | A deterministic, offline baseline avoids a model download and makes the small prototype easy to run. | `semantic_search.py` uses scikit-learn TF-IDF; no dense embedding dependency is required. This is a baseline, not a final quality claim. |
| Combine structured metadata and text relevance | Structured chart relations can be precise; lexical search can recover relevant passages not tagged perfectly. | Hybrid retrieval uses explicit relation pairs, semantic ranking, lower-weight co-occurrence fallback and a small global semantic tail. Hand-tuned scores are not calibrated probabilities. |
| Parse the question into bounded chart facts before generation | Passing raw chart JSON alone asks a generator to select relevant data opaquely; explicit facts improve debuggability. | Parser uses rules; optional model intent returns typed intents which are vocabulary-checked and expanded with deterministic chart readers. It falls back when the intent call fails. |
| Let the model synthesize cautiously from passages | Literal extractive answers may be unhelpful for broad questions; source themes can be combined with chart facts under explicit uncertainty. | Prompt distinguishes direct text, traditional synthesis and unsupported doctrine. This is a prompt instruction only; no output verifier currently enforces it. |
| Keep mock mode available | Developers can inspect chart/facts/retrieval without a paid API key. | `LLMClient` returns a deterministic preview when no OpenAI client is configured. It is not a substitute for answer-quality testing. |
| Make the final answer provider selectable while pinning intent extraction to the OpenAI client | Local generation is useful for adapter testing and private/local deployment; changing it should not affect chart-intent extraction. | `/ask` defaults to GPT-5-mini and accepts the optional Astro Qwen response model. Qwen selection routes only final synthesis; intent still uses the configured OpenAI `LLMClient`. |
| Use explicit CPU placement for the quantized local answer model | Automatic placement can split a bitsandbytes 4-bit checkpoint over CPU and disk on CPU-only machines, which Transformers rejects. | CPU-only Qwen uses `device_map={"": "cpu"}` and low-memory base/adapter loading; CUDA retains automatic placement. The fixed 8 GB application precheck was removed. |
| Keep Intel NPU inference as a separate OpenVINO provider | The prepared NPU IR and runtime constraints differ from the Transformers bitsandbytes path; silently falling back would obscure device/runtime failures. | NPU selection uses the merged int4 OpenVINO IR, plain prompts with native chat templating disabled, stop/repetition controls, and a configurable 2,048-token ceiling. `MIN_RESPONSE_LEN` reserves response context; it is not a requirement to generate that many tokens. |
| Sanitize known generation artifacts without claiming semantic verification | Internal thought tags and echoed retrieval blocks are presentation leaks, but removing them cannot establish whether answer claims are supported. | Local Transformers and OpenVINO outputs receive limited tag/passage cleanup. No semantic claim-to-citation validator is wired into `/ask`. |
| Mark corpus gaps in metadata/docs | Missing chapters must not be represented as complete coverage. | Manifest records the detected Jataka Parijata coverage as Adhyaya 10–18 and calls out missing ranges. |

## 2. Decisions to carry forward

### 2.1 Keep calculation and doctrinal authority outside the generator

**Decision:** Preserve Chart Engine as the authority for numeric chart values, and preserve versioned retrieved sources as the authority for textual doctrine. A custom model may learn how to select, explain and cite supplied evidence; it should not be expected to memorize charts or serve as the only store for changeable rules.

**Why:** Numerical calculations require deterministic reproducibility. Source books may be corrected, replaced, expanded, or have licensing/edition metadata updated. Retrieval keeps evidence inspectable and makes these updates independent of model training.

### 2.2 Improve measurement before adding model complexity

**Decision:** Build manually reviewed chart, retrieval and answer benchmark slices before adopting embeddings, a reranker or fine-tuning as a presumed improvement.

**Why:** Current retrieval smoke cases are generated from the same corpus and do not provide independent relevance labels. A small curated benchmark will reveal whether errors arise in chart calculations, question parsing, corpus tagging, retrieval or generation.

### 2.3 Use a ~1B model as an experiment, not a fixed assumption

**Decision:** Investigate a licensed instruction model near 1B parameters using LoRA/QLoRA SFT after the hosted baseline and untuned base are evaluated. Do not train from scratch as the initial route. Continue using retrieval and Chart Engine in the model's context.

**Why:** Parameter-efficient tuning is a practical way to test response style and structured domain behavior with limited resources. A 1B model may still fail at nuanced reasoning, so release depends on the locked benchmark and deployment hardware, not parameter count or training loss alone. QLoRA is a reference approach for memory-efficient adapter tuning, but the exact recipe and hardware need to be tested for the selected model.

### 2.4 Treat dataset provenance and expert review as product requirements

**Decision:** No training record enters a release dataset without source/permission provenance, review status and a defined task label. Do not use private birth charts for training without explicit informed consent and a retention/deletion policy.

**Why:** OCR-derived text can contain errors; translations and editions may have separate rights; unreviewed model-generated answers can amplify unsupported doctrine. Synthetic examples are useful for coverage but cannot replace subject-matter review.

### 2.5 Require layered guardrails and evaluations

**Decision:** Use input/schema validation, chart and intent validation, retrieval evidence checks, structured generation, deterministic citation checks and safety tests as separate controls. Maintain component-level metrics and human review for interpretation.

**Why:** A prompt alone cannot enforce the evidence contract. A single end-to-end score can obscure whether incorrect answers came from calculation, retrieval, or synthesis. Guardrails reduce failure probability but cannot promise an errorless system.

### 2.6 Avoid unsupported “first” claims

**Decision:** Describe novelty as the system's intended combination of auditable deterministic calculation, relation-aware source retrieval and evidence/uncertainty contracts. Do not claim that chart-grounded Vedic AI chat, RAG-based astrology, or custom-trained astrology models are new categories.

**Why:** Public product descriptions already advertise chart-based Vedic conversation and sources/classical rules. This market scan is not exhaustive and cannot prove that no competitor uses a particular internal technique.

## 3. Decisions still open

1. Which school/lineage and calculation conventions should be the supported default, and how will alternatives be represented?
2. Which source editions can legally be redistributed, indexed for a public product, and used to derive training examples?
3. Which ~1B base model meets language, license, context-length and device constraints?
4. Which quality thresholds and review process qualify a model/corpus release?
5. For a hosted multi-user release, should data remain request-only or be stored server-side? Define consent, encryption, access and deletion behavior first. The current local UI persists profile and chat only in browser storage.
6. Which high-stakes and crisis response policy should apply to astrology-related questions?

## 4. Evaluation suite decisions (2026-10-21)

- Use 20 distinct synthetic birth profiles with pinned calculation settings and `as_of_date`; divide them into 12 development cases and 8 holdout cases. This provides reproducible dasha timing and varied dates, times, and locations.
- Store expected chart facts, evidence chunk IDs, source/page references, acceptable answer points and prohibited claims in JSONL. Include unsupported corpus coverage and a medical-safety question to test abstention boundaries.
- Separate model-free retrieval evaluation from full answer evaluation. The latter uses the configured generation model plus a separately configured judge. Judge scores cover chart consistency, evidence support, citation validity, relevance and uncertainty, but are diagnostic rather than proof of interpretive truth.
- Audit all four PDFs against their page inventory and source hashes with PyMuPDF. Record blank text layers and detected chapter coverage; visual page review is still needed to judge semantic OCR quality.
- Fix the observed parser gaps: recognize `dasa` as a dasha term, and preserve life-topic houses when a question also mentions dasha/timing. Correct two test labels that asked about textual doctrine but incorrectly required unrelated natal facts.
- Increase the per-fact candidate pool from 3 to 5 after the first report; the existing 80-query retrieval smoke metrics remained unchanged. Add a separately scaled raw-question semantic channel because unscaled TF-IDF scores were being overwhelmed by chart-fact scores. Compare development and holdout reports before accepting that ranking adjustment.

The suite is an initial engineering benchmark, not a claim of expert validation. The full answer-generation baseline uses the private configured generator and a temporary separate judge-model setting; the default `.env.example` intentionally leaves `EVAL_JUDGE_MODEL` blank.

## 5. Browser UI decisions (2026-10-22)

- Keep the client as static HTML, CSS and JavaScript served by Astro RAG; the repository has no frontend build pipeline, and the current UI does not need one.
- Require birth date, local birth time and a selected birthplace before showing the chat. Support explicit place search and optional map/coordinate selection; rely on Chart Engine to resolve timezone from coordinates.
- Keep the transcript visible and retain the birth-profile snapshot used for each question/answer when the active profile changes, so history remains understandable without deleting prior messages.
- Persist profile, advanced settings, theme and conversation in versioned browser `localStorage`, with a clear-data control. This is browser/device-local storage, not account-backed persistence.
- Keep calculation and retrieval controls collapsed in the sidebar. Default the interface to the system light/dark appearance and allow a manual override.
- Route place searches through Astro RAG to the configured Chart Engine. Cache successful geocodes for one hour (up to 128 entries) and serialize upstream cache misses at one request per second for the default public geocoder.

## 6. Decision history / provenance

The current code and corpus are the source for “implemented” claims. Existing project notes mention the product plan and implementation history, but this record is written from the checked-in implementation and its supporting artifacts. New decisions should be added with date, status, context, alternatives considered, evidence and consequences rather than rewriting historical rationale as if it were known at implementation time.
