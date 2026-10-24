# Product Requirements: Vedic Astrology Question Answering

**Status:** Prototype exists; requirements below distinguish shipped scope from target scope.  
**Product intent:** Help a user ask a question about a Vedic birth chart and receive an understandable answer that exposes the chart facts and textual sources behind its interpretation.

## 1. Problem and product direction

Astrology answers are difficult to inspect when chart calculations, traditional rules, and fluent language are mixed inside one opaque model. This project separates those responsibilities: calculate the chart with deterministic software, retrieve passages from a declared source corpus, and use a language model to explain the combination. The current options include hosted GPT-5-mini, mock generation, and an optional local fine-tuned Qwen3-4B adapter. A future project-trained compact model may learn the response task and Jyotisha vocabulary; it must not replace the calculation engine or silently become the authority for source doctrine.

Astrology is a cultural and interpretive framework, not an empirically established guarantee of life outcomes. The product must present readings as tradition-based interpretation, avoid certainty about health, death, financial, legal, or other consequential events, and not encourage users to make high-stakes decisions solely from a reading.

## 2. Users and main job

**Primary user:** Someone who has birth date, local birth time, and birth location and wants to understand a chart in plain language.

**Primary job:** “When I ask about my chart or a life area, show me which chart facts and source passages the answer uses, and explain the interpretation with appropriate uncertainty.”

**Secondary users:** Developers inspecting the API; a Jyotisha reviewer validating chart rules, corpus passages, and answer examples.

## 3. Goals

1. Produce reproducible, versioned chart facts from explicit birth inputs and calculation settings.
2. Retrieve relevant, source-attributed doctrine from a versioned corpus.
3. Generate clear answers that distinguish source statements from interpretation and make evidence inspectable.
4. Detect missing, ambiguous, contradictory, or low-quality inputs and degrade safely.
5. Establish component-level and end-to-end evaluations before training or promoting a project-specific compact model.
6. Eventually assess whether a compact ~1B-parameter model can meet quality, safety, latency, and cost targets for the domain.

## 4. Non-goals

- Claiming scientifically validated prediction or guaranteed outcomes.
- Replacing a qualified astrologer, clinician, lawyer, or financial adviser.
- Training a foundation model from scratch as the first custom-model step.
- Treating model fine-tuning as a substitute for live source retrieval, deterministic chart math, or evaluation.
- Claiming complete coverage of the four books while the supplied Jataka Parijata OCR is partial.
- Supporting all Jyotisha lineages and calculation conventions in the first release.

## 5. Current scope (implemented)

### Chart Engine

- Accept birth date, local time, latitude/longitude and calculation settings.
- Resolve timezone, handle ambiguous/nonexistent local times explicitly, convert to UTC and Julian Day, and calculate planetary positions using pyswisseph.
- Emit chart JSON with positions, whole-sign semantic house placements, nakshatras/padas, aspects, Vimshottari periods, derived facts, validation checks, warnings and provenance.
- Provide `/chart`, health and location helper endpoints plus a CLI.

### Astro RAG

- Accept birth details and a question or an already-calculated `chart_json`.
- Extract question intents using a constrained LLM step when configured, validated against the supported vocabulary and chart; fall back to rule-based parsing.
- Retrieve from a 2,800-chunk processed corpus using metadata and TF-IDF search, with explicit relation metadata, limited recall fallback and deduplication.
- Generate answers with GPT-5-mini/OpenAI by default, deterministic mock mode without a configured client, or the optional Qwen3-4B LoRA adapter through either Transformers (CPU/CUDA) or a separately exported OpenVINO Intel NPU provider. Model selection affects final answer generation only; intent extraction remains on the configured OpenAI client.
- Load the existing Qwen 4-bit bitsandbytes base plus LoRA adapter on demand with Transformers/PEFT low-memory loading. CPU-only machines use explicit all-CPU placement; CUDA machines use automatic device placement. The application has no fixed free-RAM threshold, but actual model loading still depends on available system/GPU memory.
- Return the answer, facts, retrieved passages and metadata, chart summary and generation diagnostics.
- Provide retrieval and corpus inspection endpoints, corpus build/verify scripts, and smoke evaluation utilities.
- Accept optional `as_of_date` on `/ask` and forward it to Chart Engine for reproducible time-sensitive fields.
- Include an initial 20-case end-to-end evaluation suite (12 development, 8 holdout), a page-level PDF text audit, and a separate-model answer-judge runner.
- Serve a responsive browser chat UI with first-use birth-profile setup, profile editing, saved local conversation history, light/dark themes, and collapsed advanced calculation/retrieval settings.
- Provide a same-origin place-search proxy with in-memory result caching and explicit user-triggered searches.

## 6. Requirements for the next dependable release

### 6.1 Chart integrity

- Preserve input values, selected tradition/settings, timezone source, ephemeris mode and engine/schema versions with every result.
- Reject invalid coordinates, dates and times with stable machine-readable errors; require a user choice for ambiguous civil time.
- Make the effective calculation date explicit when returning a current dasha or time-sensitive result.
- Build independent reference fixtures and chart comparisons reviewed by a domain specialist; quantify discrepancies by body, setting and date instead of treating internal consistency checks as external accuracy proof.
- Do not claim support for a house model or derived calculation until that behavior is implemented and separately validated.

### 6.2 Corpus quality and retrieval

- Record edition, translator/commentator, source file hash, licensing/permission status, OCR/build version, coverage notes and page provenance.
- Preserve immutable source text separately from normalized and retrieval text; any OCR correction must be reviewable and traceable.
- Make missing chapter ranges, OCR uncertainty and conflicting source readings visible in retrieval metadata and answer context.
- Create a manually adjudicated evaluation set with question, expected chart facts, relevant source/chapter/page/sloka and acceptable evidence alternatives. Split by source section and question family to reduce leakage.
- Measure Recall@k, MRR/nDCG, citation/provenance accuracy, no-result behavior, and retrieval performance by book, topic, OCR quality and relation type.
- Compare metadata-only, TF-IDF, dense retrieval, and hybrid variants using the same fixed benchmark before choosing additional dependencies.
- Continue reviewing the initial synthetic 20-case suite against its cited PDF pages; its exact-passage labels are not yet a domain-expert adjudicated benchmark.
- Track retrieval Hit@k, Recall@k, and MRR separately from answer review dimensions; do not collapse them into a single score of correctness.

### 6.3 Answer quality

- Cite claims to returned passages with stable passage IDs and page/chapter/sloka details where recoverable.
- Separate: (a) chart facts, (b) claims directly stated in evidence, and (c) interpretive synthesis. Do not present category (c) as a quotation or deterministic fact.
- State when the source evidence is absent, incomplete, OCR-uncertain or contradictory; ask for clarification when birth data or the question is materially ambiguous.
- Avoid deterministic language for predictions. Never present astrology as a basis for medical diagnosis, self-harm decisions, legal/financial decisions, or guaranteed outcomes.
- Validate generated output structure and citations after generation; on validation failure, retry within a bounded policy or return a clear non-answer with available evidence.

### 6.4 Reliability, privacy and abuse resistance

- Define request limits, timeouts, retry policy and stable errors for both services. Ensure partial/incomplete model responses are detected and not mistaken for complete answers.
- Keep API credentials outside source control. Avoid logging full birth details, chart JSON, prompts or answers by default; set a clear retention policy before collecting telemetry.
- Before public deployment, add authentication, rate limiting, restricted CORS, TLS at the deployment edge, input-size limits, dependency/image scanning, and per-user isolation for any stored data.
- Treat user prompts and retrieved corpus text as untrusted input. Add prompt-injection handling, corpus integrity controls and tests showing that instructions found inside the question/source cannot override application policy.
- `include_debug` must be disabled or access-controlled outside development.

## 7. Future custom ~1B model

### Product role

The compact model should improve domain-specific intent handling, terminology, answer structure, evidence use and tone. The deterministic engine remains the source of chart values; retrieved source passages remain the changeable source of doctrinal evidence. A 1B model must be benchmarked against the current hosted-model baseline and an unmodified open-weight baseline. Do not assume it can match a much larger model on broad reasoning.

### Data workstream

Create a versioned, permission-cleared dataset from reviewed examples. Each record should include a stable example ID, task type, chart-engine/schema version, calculation settings, minimized synthetic or explicitly consented chart input, user query, evidence passage IDs and texts, reference answer, required citations, uncertainty/refusal labels, provenance, reviewer, review status and split. Never leak private user charts into training without explicit informed consent and data handling controls.

Build distinct task families:

1. **Intent/fact selection:** user query plus chart summary -> schema-valid intents or abstention; chart code expands intents into facts.
2. **Evidence-grounded response:** question + relevant facts + passages -> answer with passage IDs, claim labels and uncertainty.
3. **Source conflict and missing evidence:** conflicting edition passages or no supporting passage -> acknowledge limits, do not invent a rule.
4. **Safety and boundaries:** high-stakes requests, deterministic-fate requests, prompt injection, unrelated queries -> safe bounded response or redirect.
5. **Language and terminology:** English first, then separately reviewed Hindi/other language examples if explicitly in scope.

Training examples should be authored or reviewed by qualified domain readers, use faithful source paraphrase rather than fabricated interpretations, and include hard negatives (near-matching but wrong planet-house relation, unsupported claims, misleading OCR, irrelevant passages). Split by underlying source rule/chart archetype, not by paraphrased prompt, to limit train/test leakage. Maintain provenance and permissions for every training source.

### Training/evaluation sequence

1. Establish human-reviewed answer and retrieval baselines before training.
2. Select an open-weight instruction model near 1B after checking model license, language coverage, context length, inference targets and benchmark suitability.
3. Start with supervised fine-tuning using parameter-efficient LoRA/QLoRA; compare against prompting and retrieval improvements first. Only consider continued pretraining if vocabulary/domain comprehension tests show that SFT is insufficient and source rights permit it.
4. Keep training, validation and locked test sets separate. Track exact dataset/model/tokenizer/base revision, code, hyperparameters, seeds and evaluation outputs.
5. Add preference optimization only after reliable preference labels exist. It is not a substitute for factual evidence labels.
6. Quantize/deploy only after measuring quality, memory, latency, throughput and failure behavior on target hardware.

### Release gates

No custom model is promoted unless it meets predeclared thresholds on chart-fact fidelity, intent schema validity, retrieval-grounded citation precision, answer faithfulness, abstention on unsupported doctrine, safety cases, language quality and latency. Report per-category results and confidence intervals or sample counts; keep expert review for interpretation. Any model or corpus change reruns the locked suite and is compared to the previous release.

## 8. Evaluation plan

Evaluate layers independently before end-to-end quality:

| Layer | Example checks |
|---|---|
| Chart calculation | Fixed reference charts, input validation, timezone/DST edges, calculation settings, external comparison with tolerance |
| Intent/facts | Fact exact match, supported schema rate, abstention accuracy, chart-membership validation |
| Corpus | OCR/coverage audits, source hash checks, chapter/sloka/page mapping |
| Retrieval | Recall@k, MRR/nDCG, relation precision, per-book/topic slice, no-evidence behavior |
| Generation | Claim support, citation correctness, contradiction handling, completeness, calibration/hedging, readability, refusal quality |
| Security/privacy | Prompt injection, untrusted retrieved text, secret handling, PII/logging, debug exposure, request abuse |
| Operations | Availability, p50/p95 latency, timeout/error rates, model cost or local memory/throughput |

Automated judge scores are diagnostic aids, not substitutes for adjudicated relevance labels and expert review. Maintain a small manually reviewed set that is stable across releases and a separate growing challenge set.

## 9. Phased delivery

1. **Baseline reliability:** Confirm reproducible setup; fix schema drift, errors, current-date behavior and deployment configuration; preserve existing API contracts where possible.
2. **Data and evaluation foundation:** Document source permissions/editions; human-review retrieval labels; establish chart and answer evaluation sets; capture baseline metrics.
3. **Grounded output safeguards:** Add structured answer contract, citation checking, abstention/contradiction logic, prompt-injection defenses, privacy controls and security boundary for deployment.
4. **Retrieval improvement:** Benchmark corrected OCR, improved lexical/dense/hybrid search and reranking against the fixed retrieval set.
5. **Custom model feasibility:** Select candidate base model, build licensed expert-reviewed data, train a LoRA/QLoRA baseline, evaluate against hosted and base models.
6. **Product expansion:** Only after reliability gates, add validated chart features (e.g. divisional charts), broader sources/languages and user-facing workflows.

## 10. Open product decisions

- Which Jyotisha lineage and calculation defaults define the first supported product?
- Which source editions can be redistributed, indexed, and used to produce derivative fine-tuning data?
- What languages and deployment hardware are target requirements for a ~1B model?
- For a future hosted multi-user product, what account-level birth-data, conversation, consent, server-retention and deletion controls are required? The current local UI stores data only in the user's browser and includes a clear-data action.
- What quality thresholds are acceptable, and who is responsible for expert adjudication?
