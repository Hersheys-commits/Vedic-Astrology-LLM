# Astro RAG / Fine-tuned Qwen: Code and Output Audit

**Audit date:** 24 October 2026  
**Inputs reviewed:** 250-example dataset, Qwen3-4B fine-tuning notebook, chart engine and `chart-engine/chart.json`, and the 23 Oct and 24 Oct 40-question NPU answer runs.

## Executive verdict

**The system is not yet working reliably enough to call it a professional astrology LLM. Do not retrain it as the first response.** The largest visible failures are deterministic chart-grounding, source-condition checking, citation/retrieval quality, and response discipline. Most are best addressed in the input/prompt/retrieval/post-generation code before spending effort on another fine-tune.

The saved test runs are especially concerning: the two runs have the same 40 questions, chart snapshots, chart facts used, and retrieved-passage objects, and all requests completed successfully, yet **0 of 40 answer texts match exactly**. The mean whitespace-normalized `difflib.SequenceMatcher` similarity is **0.1458** (median **0.1486**, range **0.0287–0.2550**). This score measures text stability only, not correctness. The JSON files do not contain effective prompt hashes or a reliable model-artifact identity for both historical runs, so the divergence cannot be attributed solely to sampling; prompt, export, or runtime changes between those runs remain possible.

The code patches in the accompanying ZIP make chart corrections stricter, tighten the NPU prompt, make progress frames appear before slow preparation work, and add prompt/model diagnostic fingerprints. **They have passed unit tests and the existing retrieval benchmark; the actual NPU model has not been rerun here**, so new answer quality and live browser streaming must still be confirmed on the user's machine.

## Audit results

### 1. Repeatability and service health

| Measurement | 23 Oct run | 24 Oct run |
|---|---:|---:|
| Requests | 40 | 40 |
| HTTP 200 | 40 | 40 |
| Generation status `completed` | 40 | 40 |
| Incomplete reasons | 0 | 0 |
| Provider mismatch | 0 | 0 |
| Passage sets identical between runs | 40/40 | 40/40 |
| Chart-fact sets identical between runs | 40/40 | 40/40 |
| Mean latency | 57.26 s | 51.59 s |
| Median latency | 53.20 s | 48.74 s |
| Mean generated tokens | 286.3 | 269.6 |
| Answers matching exactly between runs | 0/40 | 0/40 |

All calls completing means the endpoint/runtime is broadly functioning; it does **not** mean the generated interpretations are correct. Identical chart and retrieval outputs isolate much of the observed variation to generation/output processing, although the missing effective-prompt/export fingerprints prevent a definitive cause attribution.

### 2. Quality rubric

This is a qualitative review of the supplied 40-answer test, not a validated benchmark score.

| Criterion | Rating | Evidence / interpretation |
|---|---|---|
| Chart factuality | **Poor** | Repeatedly confuses lordship and occupancy, changes the wrong house/sign, or asserts a house is occupied by the wrong planet. |
| Source-rule application | **Poor** | Frequently cites a passage whose stated condition/topic does not support the conclusion, or silently applies a rule despite a missing condition. |
| Reasoning consistency | **Poor** | Contradictory statements inside an answer, including conditions described as both met and unmet. The simple contradiction-pattern scan matched 13 older and 15 newer answers; these are flags, not a perfect contradiction detector. |
| Relevance to the question | **Mixed to poor** | Some answers attempt the right topic, but often drift into unrelated children, health, property, generic career, or yoga claims. |
| Simplicity/readability | **Mixed to poor** | Repetitive headings and long passage dumps make answers look structured while obscuring the actual conclusion. Several malformed book names and labels occur. |
| Citation reliability | **Poor** | Only 14/40 saved answers in each run explicitly use a `Passage N` citation, despite the prompt's instruction. Several book titles/abbreviations are malformed; some chapter/sloka references do not match the retrieved passage's subject. |
| Calibration / abstention | **Mixed** | The answers to exact child-sex and spouse-initial questions appropriately refuse certainty. Other answers make unsupported predictions about wealth, relationships, health, and lifespan. |
| Repeatability | **Poor** | No exact answer match across the two saved runs despite identical recorded chart facts and retrieved passages. |
| Latency | **Slow but usable** | About 49–57 seconds median per response in the supplied runs. This is not itself evidence of quality failure; it is a UX issue. |

### 3. Dataset and fine-tuning design

- The JSON has **250 examples split into 200 train and 50 test**; the notebook is not training on all 250 at once.
- The mix is 170 `grounded_synthesis`, 40 `conditional_check_or_partial`, and 40 `clean_abstention`. The variety is useful, but target answers are often long and contain technical labels, multi-paragraph reasoning, and source-citation patterns that the deployed prompt then suppresses or changes.
- The dataset's system prompt is about **5,870 characters**. Runtime `app/generation/prompt.py` has a materially different prompt (about **7,634 characters**), and the NPU path substitutes a separate compact prompt when the full prompt exceeds 2,000 characters. That three-way mismatch is likely to weaken instruction-following and style consistency. A compact stricter NPU prompt is now in code, but it is not byte-identical to the training prompt; exact prompt parity should be a deliberate future training/evaluation choice.
- Two semantically identical/similar questions occur with different target labels across train/test: `astro_v2_015` vs `astro_v2_221`, and `astro_v2_109` vs `astro_v2_191`/`astro_v2_192`. This is not necessarily fatal, but it can teach contradictory response modes and makes the test split less independent. Deduplicate or group by normalized question/template before any future train/test split.
- Six examples were flagged for manual inspection for possible internal/meta text: `astro_v2_080`, `astro_v2_121`, `astro_v2_147`, `astro_v2_154`, `astro_v2_157`, `astro_v2_160`. These are **review candidates**, not confirmed leakage.
- Fine-tuning uses Qwen3-4B-Instruct-2507 with 4-bit/LoRA (rank 16, alpha 32), three epochs, learning rate 1e-4, response-only loss and greedy inference. These are plausible parameters; this audit found no reason to declare the training configuration itself the primary defect.
- The notebook's lightweight custom evaluator checks formatting/citation/abstention-style properties but does not reliably measure chart-fact accuracy, source-condition correctness, or repeatability. These must be explicit independent evaluation dimensions before another training run.

**Recommendation:** no retraining now. First fix the prompts and deterministic guards, then rerun the exact same 40-question set. If errors persist with identical fingerprints, clean the data and align the training/runtime instruction contract before considering a small controlled retrain.

### 4. Chart engine and chart JSON

The inspected fixture has a coherent whole-sign map for a Taurus Ascendant: H1 Taurus/Venus, H2 Gemini/Mercury, H3 Cancer/Moon, H4 Leo/Sun, H5 Virgo/Mercury, H6 Libra/Venus, H7 Scorpio/Mars, H8 Sagittarius/Jupiter, H9 Capricorn/Saturn, H10 Aquarius/Saturn, H11 Pisces/Jupiter, H12 Aries/Mars. The listed planet occupancies also align internally with its summary (H3 has Moon/Venus/Saturn; H4 Sun/Mars/Mercury; H5 Jupiter; H6 Ketu; H12 Rahu).

This is an **internal-consistency check**, not an independent ephemeris/astronomical validation of the birth chart. For this test fixture, the dominant errors are in the LLM's use of the supplied chart, not an obvious mismatch within the chart JSON. The chart-engine test suite was also attempted: **29 passed and 13 failed because `timezonefinder` is not installed in this audit environment**. The package's `requirements.txt` does pin `timezonefinder==6.5.8`, so install the chart-engine requirements in the target environment and rerun; these failures do not establish a source-code defect in timezone resolution.

The intended distinction needs to be enforced mechanically:

- **Lord** = the planet in the house map's `lord` field.
- **Occupant** = the planet whose `placement.house` equals that house.
- They are different fields; occupying H12 does not make Rahu the 12th lord. Here Mars is 12th lord and occupies H4, while Rahu occupies H12.

A post-generation guard already existed but did not cover common phrasing consistently. It now repairs chart-specific lord labels, compact `H#` placement claims and house/sign claims; it avoids rewriting hypothetical source rules that begin with `if`/`when`/`suppose`/`for example`; it also reduces one class of duplicate house-correction phrasing. It remains a deterministic safety net, **not** a full semantic verifier: unsupported claims that are phrased indirectly can still get through.

### 5. Concrete test failures (representative, not exhaustive)

| Test(s) | Failure pattern in saved answers | Why it is wrong / fix |
|---|---|---|
| Q1 financial comfort | Uses BPHS Ch. 31, Sl. 3 (a Pada calculation passage) to support a wealth claim. | Topic/retrieval mismatch; use wealth-relevant evidence or narrow the answer. Conditional topic-aware reranking is patched, but the passage still needs a relevance gate. |
| Q3 and Q33 abroad/work | Calls Rahu the 12th/9th lord in variants. | Occupant-vs-lord confusion. Deterministic chart-specific lord repair is added. |
| Q5, Q9, Q26, Q32 education/career | Calls Mercury the 4th lord or Jupiter the 5th lord in Taurus rising chart. | Incorrect house lordship. New verifier uses the complete Houses map to correct common explicit forms. |
| Q6 education | One run says Mercury is exalted; the other calls it an ordinary sign. | Direct inconsistency: Mercury in Leo is an ordinary sign under the project's dignity table. Compact NPU prompt now includes a dignity table; answer output still must be retested. |
| Q7–Q10, Q28, Q30–Q31 career | Claims a 6th-lord-in-6th or 10th-lord-in-10th condition while Saturn is in H3; labels Ketu as 6th lord; invents a “Mercury Mahapurusha Dharma” or “Nabhasa Yoga”. | Wrong conditions/technical labels and ungrounded conclusions. Prompt now prohibits invented yogas and unseen divisional charts; verifier removes unsupported yoga names unless supported by evidence. |
| Q8, Q18, Q23 | Claims a house is occupied solely by the wrong planet or swaps the 4th/5th lord. | Must check all occupants and lords against the chart map, not infer from prose. Guard fixes explicit placements/lordship, but broad occupancy prose remains a residual risk to validate. |
| Q11–Q12 marriage | Corrects false Ketu/Saturn-in-7th premise in part, then adds an unsupported “protective triangle” or irrelevant health claims. | Correcting the premise does not license unsupported doctrine. Keep reading narrow and cite a relevant rule. |
| Q15, Q21–Q22 property/relationships | Introduces D-5/D-6/D-7/D-9/D-10 information absent from the D-1 chart. | Unseen divisional-chart facts were invented. NPU prompt now forbids using those divisional charts unless data is supplied. |
| Q19–Q20 children/property | Brings in conception/sex-related conditions for a child-success or property question. | Irrelevant evidence and topic drift. Retrieval relevance gate and direct-answer-first format need more work. |
| Q34 foreign settlement | States the 12th-lord-in-4th rule, then says it does not apply despite Mars (the 12th lord) being in H4. | Reasoning contradicts the chart and the cited condition. Mechanical placement/lord repair helps, but rule-condition checking remains a model-quality criterion. |
| Q35–Q36 longevity / over-85 | Discusses longevity based on a chart rule, but the specific age threshold cannot be established from these inputs. | Avoid precise lifespan predictions; NPU prompt now requires a limitation on exact lifespan questions. |
| Q38 dropping out | Correctly cautions against dropping out solely on astrology, but misstates the lord of Pisces/9th house. | Helpful decision advice does not excuse false chart facts. Lordship correction is included. |
| Q39 mother's health | Includes a medical disclaimer, then still implies a health pattern from poorly grounded chart claims. | Avoid diagnosis-like interpretations and keep medical matters outside predictive claims. |
| Q40 spouse initial | Properly abstains from predicting exact initials, but cites a malformed source name. | The refusal is good; source-name cleanup is patched. |

A full row-by-row answer comparison is stored in `astro-rag-new/audit/manual_run_comparison.json`; the standalone script `scripts/audit_manual_runs.py` lets you compare future results the same way.

### 6. Retrieval evaluation

The existing 80-query retrieval smoke benchmark was rerun after the change.

| Strategy | hit@1 | hit@3 | hit@5 | precision@3 | recall@5 |
|---|---:|---:|---:|---:|---:|
| Hybrid (current after patch) | 0.550 | 0.775 | 0.850 | 0.425 | 0.708 |
| Metadata-only reference | 0.800 | 0.875 | 0.888 | 0.658 | 0.833 |

Metadata-only performs better on this **particular set of 80 mostly direct chart lookup queries**, but switching all user queries to metadata-only would be an unjustified global decision because broad life-area questions need semantic topic relevance. An earlier blanket reduction of relation scoring harmed hit@1 (0.55 → 0.4625), so it was rejected. The patched reranker changes weighting only for broad interpretive questions while preserving the original direct-lookup formula; the measured hybrid benchmark returned to its original scores.

**Known unresolved retrieval defect:** all **9/9 dasha benchmark queries have hit@1 = hit@3 = hit@5 = 0** in both hybrid and metadata-only. This needs a focused fix to query expansion, dasha metadata/indexing, and test labels. Do not infer dasha correctness from successful HTTP responses.

### 7. Streaming diagnosis and patch

The UI already reads `/ask/stream` and dispatches `progress`, `metadata`, `delta`, `done`, and `error` SSE frames. The key backend UX problem was that preparation of chart facts and retrieval happened before the server started sending its stream. That can look like “streaming is broken” for the first ~seconds even if token generation later streams.

Patched:

1. `/ask/stream` now yields a `progress` event **before** `_prepare_ask_context` and another before generation; it sets `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`.
2. The UI updates the visible progress message on `progress` frames.
3. OpenVINO's documented Python `LLMPipeline.generate` API accepts a streamer callback; see the official [streaming guide](https://openvinotoolkit.github.io/openvino.genai/docs/guides/streaming/) and [API reference](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.LLMPipeline.html). The code tries supported callback call forms before a last-resort non-streaming fallback; if it must fall back, it logs a warning. `generation_usage.streaming_callback_used` now records whether token callbacks actually arrived.
4. NPU generation usage now captures a prompt SHA-256 prefix, a **metadata-based model/export fingerprint** (paths, file sizes and modification times; not a cryptographic weights hash), `do_sample`, repetition penalty and streaming-callback status. This makes future run comparisons more diagnosable.

**How to validate streaming on your machine:** restart both services and use the Astro RAG frontend at `http://localhost:8001`. Select `astro-qwen-npu` for the converted NPU model. Open DevTools → Network → `/ask/stream` → Response; you should see `event: progress`, `event: metadata`, then multiple `event: delta` frames before `event: done`. Inspect the final `generation_usage.streaming_callback_used`; it should be `true`. If it is `false`, the server log explains that the OpenVINO callback was not accepted and the version/API needs alignment.

You can also send a request with `curl.exe -N` to the same endpoint. `-N` disables curl's output buffering. Do not use a client that waits for the final JSON body to finish before displaying its text: it can make a working SSE stream look non-streaming.

**Validation status:** the Astro RAG service test suite passes locally in this audit environment (124 tests). The 80-query retrieval benchmark also retained its prior hybrid scores after the conditional reranking patch. The chart-engine tests were not fully green here: 29 passed and 13 could not import `timezonefinder`; that pinned dependency is listed in the chart-engine requirements. No repaired outputs were generated by the actual NPU model in this environment.

### 8. Changes included in the patched project

- `app/api/ask.py`: early SSE progress frames and no-buffer headers.
- `app/static/app.js`: visible progress-event handling.
- `app/generation/openvino_npu_client.py`: stricter compact NPU instruction contract, stream callback compatibility attempts, fallback warning, prompt/export diagnostics.
- `app/generation/qwen_client.py`: stronger chart-specific repair for explicit house lordship and `H#` placement statements, context protection for hypothetical source conditions, common book-title/abbreviation cleanup and duplicate correction phrasing cleanup.
- `app/retrieval/hybrid.py`: question-topic-aware weighting only for broad interpretive questions; direct lookup scoring stays unchanged.
- `tests/`: regression checks for lord/occupancy errors, hypothetical-rule preservation, SSE progress order, frontend event handling and inference diagnostics.
- `scripts/audit_manual_runs.py` + `audit/manual_run_comparison.json`: a reusable run-comparison diagnostic.

### 9. What to do next (in order)

1. **Run the patched ZIP locally** and repeat the same 40 questions on the same chart. Save both JSON logs including the newly added `prompt_sha256`, `model_fingerprint`, and `streaming_callback_used` fields. The model's outputs have not yet been re-generated in this environment.
2. Review all 40 against the deterministic chart map and retrieved text; score chart facts, condition matching, source traceability, relevance, contradiction, readability and abstention independently. Do not use BLEU/text similarity as correctness.
3. Fix dasha retrieval, question/passage relevance and explicit chart-occupancy claims before new training.
4. Remove or reconcile duplicate question templates and inspect the six dataset leakage candidates. Keep a held-out set that is grouped by normalized question/template and ideally chart seed, rather than randomly splitting near-identical examples.
5. Only consider another fine-tune after these changes if the same chart-grounding errors persist with identical effective prompt/model fingerprints. If retraining is warranted, align a compact canonical system prompt across training and deployment and evaluate on a held-out set with chart-claim and evidence-condition checks—not just format/abstention checks.

## Bottom line

The chart engine and API are structurally useful, and the application can complete requests. However, the supplied results show that the answer-generation layer is **not yet consistently grounded or professionally reliable**. Fixing prompt parity, the chart verifier, question-relevant retrieval, and streaming observability is the correct first move. Retraining now would risk teaching the model more of the same unsupported answer patterns without fixing the deterministic sources of error.
