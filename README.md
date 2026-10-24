# Vedic Astrology LLM

A two-service prototype for answering questions about a Vedic birth chart. **Chart Engine** calculates a structured chart from birth details. **Astro RAG** reads that chart, finds relevant passages in a four-book OCR corpus, and generates a cited, cautious explanation.

```text
Birth date, local time, location
          |
          v
Chart Engine (FastAPI :8000)
          | deterministic chart JSON
          v
Astro RAG (FastAPI :8001) <- question
   question/fact parsing -> metadata + TF-IDF retrieval
          | chart facts + source passages
          v
OpenAI Responses API, local mock mode, or optional local Qwen LoRA inference
          |
          v
Answer + retrieved evidence + chart snapshot
```

GPT-5-mini remains the default answer generator. Astro RAG offers an optional local Transformers backend using the existing fine-tuned Qwen3-4B LoRA adapter and its 4-bit bitsandbytes base, and an optional Intel NPU backend using a separately exported OpenVINO GenAI model. The selected response model affects final answer generation only; intent extraction remains on the configured OpenAI client. The Transformers CPU path uses explicit CPU placement and low-memory loading, but can be slow and needs sufficient physical memory; the NPU path uses its own prompt, generation, and memory limits. See the [Astro Qwen NPU setup and diagnostics](astro-rag-new/README.md#optional-run-astro-qwen-on-an-intel-npu). The planned project-trained ~1B model has not been built. The system does not establish that astrology predicts real-world outcomes; readings should be treated as interpretations within a named tradition.

## Repository layout

| Path | Purpose |
|---|---|
| `chart-engine/` | Birth-time validation, timezone resolution, Swiss Ephemeris calculations, chart API and CLI |
| `astro-rag-new/` | Question understanding, corpus ingestion, retrieval, generation, API and evaluation utilities |
| `PRD.md` | Product requirements and phased plan |
| `ARCHITECTURE.md` | Current architecture and target-state design |
| `DECISION.md` | Recorded implementation decisions and proposed future decisions |
| `NOVELTY.md` | Qualified market scan and differentiation thesis |

## Run locally

Use Python 3.11 or later. Run each service in its own terminal and virtual environment. Commands below are from the repository root.

### 1. Install Chart Engine

```powershell
cd chart-engine
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The chart engine can use the included Swiss Ephemeris data in `chart-engine/data/ephe/`. If unavailable, it falls back to Moshier and identifies that mode in chart provenance and warnings. Check the ephemeris data terms before redistributing it.

For place search, configure `GEOCODER_USER_AGENT` in `chart-engine/.env` with an application identifier and contact address; the default public geocoder may reject a placeholder identity.

Start it:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful URLs: [Chart Engine UI](http://localhost:8000/), [OpenAPI docs](http://localhost:8000/docs), and `GET http://localhost:8000/health`.

### 2. Install Astro RAG

In another terminal:

```powershell
cd astro-rag-new
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `CHART_ENGINE_URL=http://localhost:8000`. To use hosted generation, set `OPENAI_API_KEY` in `astro-rag-new/.env`; optionally set `OPENAI_MODEL` (default `gpt-5-mini`). Leave the key empty for deterministic mock mode. `INTENT_MODE=auto` uses the configured model for fact extraction when available and falls back to rules; set `INTENT_MODE=rules` to keep question parsing local and deterministic. Keep `.env` private and never commit keys.

Start Astro RAG from its directory so the `.env` and relative paths resolve correctly:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Useful URLs: [Astro RAG UI](http://localhost:8001/), [OpenAPI docs](http://localhost:8001/docs), `GET /health`, and `GET /corpus/stats`.

The Astro RAG UI starts with birth date, local birth time, and birthplace setup, then opens a chat with the saved profile in its sidebar. Profile, settings, theme, and chat history are stored in the current browser. Place search is available at Astro RAG's same-origin `POST /location/geocode` endpoint and is forwarded to Chart Engine with in-process caching and serialized uncached lookups; the sidebar keeps optional chart and retrieval settings collapsed under **Advanced settings**.

To enable local Qwen answer generation, install `astro-rag-new/requirements-local-model.txt` in the Astro RAG virtual environment and authenticate with Hugging Face for read access to the private adapter. See the [Astro RAG Transformers setup notes](astro-rag-new/README.md#optional-run-the-astro-qwen-answer-model-locally). The separate Intel NPU option requires the OpenVINO export/runtime dependencies and prepared IR; see the [NPU setup notes](astro-rag-new/README.md#optional-run-astro-qwen-on-an-intel-npu). Model selection changes final answer generation only; GPT-5-mini remains the default and intent extraction stays on the OpenAI client.

The processed corpus is already included. To rebuild from the supplied PDFs, run from `astro-rag-new/`:

```powershell
python scripts/build_corpus.py
python scripts/verify_corpus.py
python scripts/check_unicode.py
```

Restart Astro RAG after rebuilding; it loads the corpus into memory at startup. Check that you have rights to use the source editions and OCR for your intended distribution and model-training use.

## Ask a question

The browser page on port 8001 provides a simple manual interface. The API accepts birth details (and calls Chart Engine) or a previously computed chart JSON.

PowerShell example:

```powershell
$body = @{
  birth_date = "2004-05-17"
  birth_time = "14:35:00"
  location = @{
    mode = "map_click"
    latitude = 25.4358
    longitude = 81.8463
    display_name = "Prayagraj, India"
  }
  settings = @{ ayanamsa = "lahiri"; house_system = "whole_sign"; node_type = "mean" }
  as_of_date = "2026-10-01"
  question = "What themes are associated with my career?"
  strategy = "hybrid"
  top_k = 5
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Method Post -Uri http://localhost:8001/ask `
  -ContentType "application/json" -Body $body
```

API endpoints:

| Service | Method and path | Use |
|---|---|---|
| Chart Engine | `POST /chart` | Compute chart JSON |
| Chart Engine | `POST /location/geocode` | Resolve an address to candidate coordinates (network/provider dependent) |
| Chart Engine | `POST /location/reverse` | Resolve coordinates to a display name |
| Chart Engine | `POST /location/timezone` | Resolve IANA timezone offline |
| Chart Engine | `GET /health` | Engine/schema/ephemeris status |
| Astro RAG | `POST /ask` | End-to-end answer from birth details or `chart_json` |
| Astro RAG | `POST /location/geocode` | Cached same-origin proxy to Chart Engine place search |
| Astro RAG | `POST /retrieve` | Inspect retrieval without generation |
| Astro RAG | `GET /corpus/stats` | Loaded corpus counts |
| Astro RAG | `GET /health` | Corpus, generator mode, and best-effort chart-service reachability |

`/ask` returns answer text, parsed chart facts, retrieved passages and metadata, retrieval strategy, chart snapshot, generation model/mode, and generation diagnostics. `include_debug=true` returns the exact generation prompts; use it only in a trusted development environment because prompts contain chart and question data.

`as_of_date` is optional. Supplying an ISO date makes time-dependent details such as the current Vimshottari dasha repeatable; omitting it preserves the existing current-date behavior.

For retrieval debugging, call `/retrieve` with a query, a fact, or both. A direct fact looks like `{"type":"planet_house","planet":"Mars","house":10}`. Strategies are `hybrid` (default), `metadata_only`, and `semantic_only`.

## Chart Engine inputs and outputs

Birth time is local civil time at the submitted coordinates. The engine resolves an IANA timezone from coordinates unless one is supplied to its `/chart` API; the Astro RAG request currently forwards date, time, coordinates, and basic settings, letting Chart Engine resolve the timezone. Ambiguous daylight-saving times require an explicit `dst_choice`. Supported settings include sidereal/tropical zodiac, five ayanamsas, whole-sign/equal/Placidus house settings, and mean/true nodes. The semantic house objects currently use whole-sign placement logic; see [known limitations](chart-engine/README.md#9-known-limitations).

The chart response includes planetary positions, houses, nakshatras and padas, aspects, Vimshottari dasha, derived facts, validation results, settings, warnings, and calculation provenance. `chart_id` is deterministic for the same submitted chart input/settings/timezone; `computed_at_utc` is naturally time-varying. The chart includes the current period as of the calculation date unless `as_of_date` is supplied to Chart Engine directly.

## Corpus and evidence

Astro RAG uses Brihat Parasara Hora Sastra, Brihat Jataka, Jataka Parijata, and Phaladeepika. The checked-in manifest reports 2,800 chunks: 1,114 BPHS, 454 Brihat Jataka, 568 Jataka Parijata, and 664 Phaladeepika. Jataka Parijata is partial: the detected main text is chapters 10–18, and coverage gaps are recorded in the manifest. Do not present this corpus as complete coverage of that book.

The corpus builder keeps page provenance, detected chapter/sloka structure, raw OCR, cleaned text and contextual retrieval text separate. Retrieval combines explicit relation metadata with scikit-learn TF-IDF ranking and a limited semantic tail; it does not currently use dense vector embeddings. The generator prompt distinguishes direct textual support from cautious synthesis and instructs the model to avoid unsupported doctrine. These prompt instructions are not a complete safety or correctness guarantee.

PyMuPDF is installed from `astro-rag-new/requirements.txt`. Run `python scripts/audit_pdfs.py` from `astro-rag-new/` to compare PDF text extraction with the page inventory, validate source hashes, and review empty/short text pages and detected chapter coverage. The audit checks the text layer and extraction consistency; it cannot establish semantic OCR correctness. Its report is written to `astro-rag-new/rag/evaluation/pdf_ocr_audit.json`.

### End-to-end evaluation

The curated suite at `astro-rag-new/rag/evaluation/end_to_end_gold.jsonl` has 20 synthetic birth profiles: 12 development cases and 8 holdout cases. Each case pins chart settings and an `as_of_date`, defines expected chart facts and answer constraints, and points to source passages by corpus ID and PDF page. It includes unsupported-source and medical-safety questions to exercise abstention behavior.

Start both services, then run retrieval-only evaluation from `astro-rag-new/` (it calls Chart Engine but makes no model calls):

```powershell
python rag/evaluation/evaluate_end_to_end.py --split all --mode retrieval-only
```

Timestamped reports are written under `rag/evaluation/reports/` and include chart facts, retrieved passages, retrieval Hit@k/Recall@k/MRR, and separate development/holdout summaries. A full answer-quality run calls the configured generation model and a separate `EVAL_JUDGE_MODEL`; configure these in the private `.env`, and ensure the judge differs from `OPENAI_MODEL`:

```powershell
python rag/evaluation/evaluate_end_to_end.py --split development --mode full
python rag/evaluation/evaluate_end_to_end.py --split holdout --mode full
```

Judge scores for chart consistency, evidence support, citation validity, relevance, and uncertainty are diagnostic. They do not establish that an astrological interpretation is correct; source review and domain-expert adjudication remain necessary. The standard `.env.example` leaves `EVAL_JUDGE_MODEL` blank; configure a separate model in the private environment (or set it for the evaluation process) before running full mode.

The included retrieval gold set/evaluation reports are useful regression aids, but their cases are generated from corpus material and are not yet a human-adjudicated quality benchmark. The chart-engine tests and Astro RAG tests are in each project’s `tests/` directory; run them from that project directory with `python -m pytest -q` when you want to check the code.

## Current limitations and operational notes

- This is a local prototype: both APIs allow permissive CORS and have no authentication, per-user data isolation, or rate limiting. Do not expose them publicly as-is.
- `include_debug` may return full prompts. Birth data is sensitive; avoid logging or sharing it unnecessarily.
- The chart service has deterministic calculations and internal consistency checks, but the current internal validation does not prove agreement with every Jyotisha school or third-party chart calculator.
- House placements are built with whole-sign logic. Although other house settings and raw cusps are available, the full derived house semantics are not implemented for each model.
- Divisional charts, Shadbala, Ashtakavarga and many yoga calculations are not implemented.
- No semantic claim/citation validator, prompt-injection filter, or trained safety classifier is currently wired into `/ask`. Local generation clients remove certain known internal-thought tags and echoed passage blocks, but that text cleanup does not validate answer claims or citations.
- Timing answers are especially sensitive to incomplete chart features and corpus coverage. They should not be presented as certain event predictions.

## Research references

The intended fine-tuning plan and market positioning are documented in [PRD.md](PRD.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [NOVELTY.md](NOVELTY.md). Research anchors include the [QLoRA paper](https://arxiv.org/abs/2305.14314), [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework), and public product descriptions linked in [NOVELTY.md](NOVELTY.md).
