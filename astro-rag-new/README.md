# Astro RAG — Real Four-PDF Corpus

Astro RAG is the downstream service that takes a user's astrology question and a chart produced by the separate chart-engine service, retrieves relevant doctrine from the **four supplied OCR PDFs**, and gives those chart facts + retrieved passages to an LLM.

The boundary is deliberate:

```text
Birth date + time + place
        |
        v
   CHART ENGINE (:8000)
        |
        | chart JSON
        v
   ASTRO RAG (:8001)
        |
        +--> question / fact decomposition
        |
        +--> metadata-constrained retrieval
        |       + semantic ranking (TF-IDF baseline)
        |
        +--> source-tagged evidence bundle
        |
        v
   GPT-5-mini / mock / optional local Astro Qwen generation
        |
        v
   Grounded answer
```

The four source books are stored under `rag/source_pdfs/` and are the only doctrinal corpus used by the shipped `rag/processed/corpus.jsonl`:

- Brihat Parasara Hora Sastra
- Brihat Jataka
- Jataka Parijata
- Phaladeepika

The source files identify BPHS as a translation/commentary volume, Brihat Jataka as an English translation with explanatory notes/examples, and Phaladeepika as the translated work of Mantreswara.

## What is actually shipped

The corpus is **not synthetic anymore**. `scripts/build_corpus.py` reads the four PDFs in `rag/source_pdfs/` and generates:

- `rag/processed/pages.jsonl` — page-preserving raw/clean OCR
- `rag/processed/corpus.jsonl` — retrieval chunks with source/chapter/sloka provenance
- `rag/processed/corpus_manifest.json` — source hashes, coverage and build statistics
- `rag/evaluation/retrieval_gold.jsonl` — corpus-specific retrieval smoke cases

Current build: **2,800 chunks from 2,003 main-text pages**:

| Book | PDF pages | Detected main-text chapters | Chunks |
|---|---:|---:|---:|
| Brihat Parasara Hora Sastra | 657 | 47 | 1,114 |
| Brihat Jataka | 588 | 28 | 454 |
| Jataka Parijata | 682 | 9 | 568 |
| Phaladeepika | 476 | 28 | 664 |

The uploaded Jataka Parijata OCR is **partial**: this build detects Adhyaya 10–18 and explicitly records missing early/later chapter coverage in the manifest. Do not treat that PDF as a complete Jataka Parijata edition. The supplied project plan also calls out this partial coverage.

The books have strong internal structure, so the corpus builder preserves `book -> chapter/adhyaya -> rule/sloka -> commentary/example` instead of applying a blind fixed-size splitter. That matches the supplied RAG plan.

## Corpus design

The primary retrieval unit is a rule/sloka segment. Each chunk keeps:

```text
raw OCR page provenance
chapter / Adhyaya
sloka number when recoverable
translation / source text as extracted
commentary child chunks when a rule segment is very long
planet / sign / house / nakshatra / dasha / relationship metadata
PDF page + printed-page provenance
```

Three text representations are intentionally separated:

```text
raw_text            -> preserved source OCR
text_clean          -> deterministic layout cleanup
retrieval_text      -> contextualized text used by the semantic index
```

No LLM is used to silently rewrite or “correct” the source during ingestion. The supplied plan explicitly calls for preserving the raw OCR and using a separate retrieval representation.

For long source segments, the builder keeps the core rule together and creates commentary child chunks instead of blindly cutting the source every N characters.

## Retrieval

`hybrid` is the default.

1. The question is converted into concrete chart facts.
2. Exact metadata gives a candidate set when possible.
3. TF-IDF semantic search ranks those exact candidates.
4. The original question is also searched directly so topic words such as `marriage`, `job`, or `personality` can recover relevant doctrine.
5. A small topic-aware boost favors passages whose indexed topics match the user's life area; a downscaled raw-question semantic channel helps recover relevant doctrine without overpowering verified chart-relation matches.
6. Redundant section overviews are removed when a leading result already contains a sloka from the same book and chapter, then duplicate chunk IDs are removed before the final `top_k` evidence set is returned.

For the local Astro Qwen and Intel NPU answer providers, intent resolution defaults to deterministic rules so repeated questions use the same chart facts; callers can explicitly request another supported intent mode. Compound rule candidates are discounted only for indexed planet/house or house-lord placements that the supplied chart can verify are unmet. Missing chart data is treated as unknown, not as a mismatch.

Topic-house expansion prioritizes specific domain terms such as media, design, and communications over generic helper words such as "succeed", and can include up to four topic houses when no explicit planet/house limits the expansion. Retrieval also applies a modest off-topic penalty to known conception/progeny chapters for unrelated questions, ignores commentary planet-pair examples when scoring a house-lord fact, and caps the first-pass contribution of an individual fact so it cannot crowd out all other chart relations.

The implementation deliberately does not use `house_lookup` / `sign_lookup` as doctrinal retrieval targets. Those are chart-engine facts and remain chart context.

## Generation / interpretation contract

`app/generation/prompt.py` uses a three-level evidence contract so the LLM can perform a grounded astrology reading instead of only repeating sentences from the retrieved corpus:

```text
DIRECT TEXT           -> explicitly stated in a retrieved passage
TRADITIONAL SYNTHESIS -> cautious interpretation combining chart facts + retrieved themes
UNSUPPORTED DOCTRINE  -> rules/claims requiring missing chart or source information (forbidden)
```

The model is therefore allowed to translate technical source themes into practical tendencies when that connection is reasonably supported. It must still keep chart facts anchored to the snapshot, cite passages it relies on, surface genuine source disagreements, hedge traditional tendencies, and avoid inventing unstated astrological rules. Personality questions are instructed to lead with the reading, then give 3–5 supporting points, and add a brief limitation only where necessary. The prompt includes the available 12-house map and planet placements/lordships, annotates verified house-lord rules with the actual chart planet and disambiguates the destination-house lord, and annotates retrieved conditions contradicted by the chart. Unavailable chart data remains unverified.

Before `/ask` and `/ask/stream` return their final answer, a deterministic post-generation guard repairs contradicted planet house/sign claims when those placements are available in the chart, and corrects a planet's two-house lordship statement only when the chart includes all 12 houses. Unsupported named yoga labels are replaced unless they are present in retrieved evidence or belong to the small standard-name allowlist. The guard improves chart consistency but is not a general fact or citation validator.

Before generation, a retrieval guard checks whether any passage's indexed planet/house, planet/sign, nakshatra, aspect, or dasha relation exactly matches a supplied chart fact. If none does, the prompt warns the model not to infer relevance from keywords or chapter titles and directs it to abstain unless a passage itself directly answers the question. The prompt also explicitly requires checking planet placements and yoga conditions and disallows invented technical labels. These checks improve grounding but do not replace review of model answers. Both local Qwen output paths remove leaked `<tool_call>` markers along with known thought tags and echoed passage text.

The supplied architecture describes the intended high-level flow as question + chart JSON → fact decomposition → retrieval → evidence bundle → LLM synthesis.

### About embeddings

The shipped semantic leg uses scikit-learn TF-IDF so the repository installs and runs without downloading an embedding model. It is a deterministic baseline, not the final semantic retriever. Replacing it later with a sentence-embedding model only requires changes in the retrieval layer.

## Question types

The deterministic parser already handles questions such as:

```text
What does Mars in the 10th house mean?
What does my Mars mean?
How will my career be?
When will I get my job?
When will I get married?
How is my personality?
What does Sun Mahadasha mean?
```

Open-ended life-area questions map to relevant houses, while timing questions also add the current running Mahadasha to the retrieval context.

An optional OpenAI intent step exists in `app/facts/intent.py`. For GPT-5-mini requests it is used when the effective intent mode is `auto` and an `OPENAI_API_KEY` is available; deterministic rules remain the fallback. For local Qwen and NPU requests, rule-based intent is the default regardless of global `INTENT_MODE`, unless the request explicitly supplies `intent_mode`.


## Verify live answer streaming

The UI uses server-sent events from `POST /ask/stream`. The first event is a `progress`
frame emitted before chart preparation and retrieval; visible answer chunks then arrive as
`delta` frames, followed by a `replace` frame containing the validated answer and a final
`done` frame containing the complete response.

1. Start the chart engine on port `8000` and Astro RAG on port `8001` as above.
2. Open `http://localhost:8001`, select `astro-qwen-npu` when testing the OpenVINO export, and ask a question.
3. In browser DevTools → **Network** → `/ask/stream` → **Response**, look for `event: progress`, `event: metadata`, multiple `event: delta` frames, then `event: replace` and `event: done`.
4. Inspect `generation_usage.streaming_callback_used` in the final event. `true` means OpenVINO delivered callback tokens. If it is `false`, check the server log for the callback compatibility warning and align `openvino-genai` with `requirements-openvino-npu.txt`.

For command-line checks, use `curl.exe -N` (the `-N` disables curl output buffering). A client that buffers the response until `done` can make a valid stream appear non-streaming. If the UI remains stuck before `metadata`, inspect chart-engine/network requests and the Astro RAG server log; progress should now be sent before those slower calls.

## Running the project

### 1. Create the environment

Linux / macOS / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Set at least:

```env
CHART_ENGINE_URL=http://localhost:8000
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
INTENT_MODE=auto
```

Leaving `OPENAI_API_KEY` blank is valid. The service will run in `mock` generation mode, which is useful for testing chart → facts → retrieval without an external API call.

### 3. Build the corpus

The shipped processed corpus is already generated, but the **reproducible source of truth is the four PDFs** in `rag/source_pdfs/`.

```bash
python scripts/build_corpus.py
python scripts/verify_corpus.py
```

### 4. Run the tests

```bash
python -m pytest -q
```

### 5. Start Astro RAG

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Open `http://localhost:8001/` for the Astro RAG chat UI. On first visit, enter a birth date, local birth time, and birthplace using place search or the optional map. The UI saves the profile, settings, theme, and conversation in browser-local storage; use **Advanced settings** in the sidebar to change calculation/retrieval options or clear that data. Each answer retains the birth profile that was active when its question was sent, and its chart facts and retrieved passages can be expanded below the answer.

### Optional: run the Astro Qwen answer model locally

The answer model selector is in **Advanced settings**. GPT-5-mini remains the default. Choosing **Astro Qwen (local)** uses the `XeroxFlynn/astro-rag-lora-v2` LoRA adapter with the 4-bit base model named in its adapter config: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`. For repeatability, local Qwen and Intel NPU requests default to deterministic rule-based question-to-chart-fact analysis, even if global `INTENT_MODE=auto`; callers can explicitly set `intent_mode` to override this default. GPT-5-mini requests continue to use the configured intent mode.

Install the optional dependencies in the Astro RAG environment:

```bash
pip install -r requirements-local-model.txt
```

For NVIDIA GPU acceleration, install the PyTorch build matching the machine's CUDA setup using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the optional requirements. CPU inference uses bitsandbytes' CPU backend with an explicit all-CPU device map, which avoids an unsupported mixed CPU/disk placement for 4-bit weights. It can be slow and memory-intensive. The base and adapter load with Transformers and PEFT low-memory loading; there is no fixed free-RAM threshold, though actual available RAM still determines whether model loading and generation can complete. The base model and adapter are loaded only on the first Qwen request, then cached in the Astro RAG process. The first request downloads model files from Hugging Face into its normal local cache; allow several gigabytes of disk space. If optional dependencies are absent or loading/generation fails, the API returns a clear error and does not fall back to GPT.

The Transformers CPU/CUDA answer-generation path is separate from the OpenVINO NPU path; full Transformers generation should be validated on the target hardware before relying on it. The OpenVINO NPU path has returned visible output in controlled tests and in the frontend. The offline Astro RAG test suite passes. These checks do not establish that every prompt produces a complete or high-quality answer.

The adapter repository is private. Authenticate with `hf auth login` as the same Windows user that runs Astro RAG, using a Hugging Face account with read access to `XeroxFlynn/astro-rag-lora-v2`. The app reads the cached Hugging Face token and never accepts it in the chat request.

To download both repositories before using the UI, run this from `astro-rag-new` after activating its virtual environment (or replace `python` with that environment's Python executable):

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('XeroxFlynn/astro-rag-lora-v2'); snapshot_download('unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit')"
```

This requires Hugging Face access to the private adapter and several GB of free disk space. On Windows, restart Astro RAG after installing the optional packages. On a CPU-only machine, closing memory-heavy applications can improve the chance that Qwen loads successfully.

### Optional: run Astro Qwen on an Intel NPU

The **Astro Qwen (Intel NPU)** answer-model option uses OpenVINO GenAI and a separately converted, merged model artifact. It does not use the existing bitsandbytes 4-bit checkpoint directly, does not change the CPU/CUDA Transformers path, and does not move question/intent analysis to the NPU. NPU selection is strict: if the OpenVINO NPU device or prepared artifact is unavailable, the request returns a clear error rather than falling back to CPU.

Install the standard local-model requirements and OpenVINO NPU/export tools in the Astro RAG environment:

```powershell
pip install -r requirements-local-model.txt
pip install -r requirements-openvino-npu.txt
hf auth login
python scripts/prepare_npu_model.py
```

Model preparation downloads the public unquantized Qwen3 base and the `XeroxFlynn/astro-rag-lora-v2` LoRA adapter, merges the adapter, then uses Optimum Intel to export an int4 OpenVINO IR. It accepts a local adapter directory when `ADAPTER_MODEL` is set to one. The adapter may supply its chat template as `chat_template.jinja` or in tokenizer configuration; the export uses the standalone template when present and otherwise preserves the tokenizer's configured template. Changing the adapter constant does not update an already-exported NPU IR: prepare the model again into a clean output directory and point `OPENVINO_NPU_MODEL_DIR` to that artifact before selecting NPU. The script refuses to overwrite a non-empty output directory, so move or back up an existing export and choose a clean output directory before preparing the new version. Export may use several GB of working RAM and disk; close memory-heavy applications first. Set `OPENVINO_NPU_MODEL_DIR` if you want the artifact stored outside the default ignored `models/astro-qwen-npu/` directory.

After export, install/update the Intel NPU driver if needed, restart Astro RAG, choose **Astro Qwen (Intel NPU)** under Advanced settings, and send a test question. The recommended configuration sets OpenVINO's `MAX_PROMPT_LEN` to 1,536 tokens and budgets the complete rendered prompt at `MAX_PROMPT_LEN - 128` (1,408 tokens), leaving room for runtime context. It removes the redundant full-house map, preserves passage headers and short verse notes, filters OCR fragments, and trims passage bodies only when needed. The NPU output limit defaults to 1,024 tokens (`OPENVINO_NPU_MAX_NEW_TOKENS`); the same value configures OpenVINO's `MIN_RESPONSE_LEN` response-context reservation. Together, these settings reserve a context of about 2,560 tokens. The output limit is an upper bound, not a target: the model may stop naturally earlier. If it reaches the cap, the API marks the response incomplete and the UI warns that it may be truncated.

Controlled tests found empty output when OpenVINO applied its native chat-template wrapper to the composed prompt, while disabling that wrapper returned visible output. The NPU client now renders system/user messages through the tokenizer's ChatML template with thinking disabled, then sends that rendered prompt with `GenerationConfig.apply_chat_template=False` so the runtime does not wrap it a second time. It also configures repetition penalty and stop strings to reduce repetition and stop at Qwen's end marker or an echoed passage block. The logged `prompt_tokens` count is the tokenizer count for the exact rendered prompt sent to the pipeline. Runtime/client-setting changes do not require re-exporting the model; restart the service to load them. Override `OPENVINO_NPU_MAX_PROMPT_LEN` only when needed, keeping within the exported model's context length and device memory limits. Larger prompt/context reservations can increase shared-memory use. Task Manager can confirm NPU activity during generation, but a successful export alone does not prove inference works on the device.

The local Transformers path uses greedy decoding, a 700-token cap, a repetition penalty, Qwen end-token stopping when available, and cleanup of known internal-thought and echoed-passage tags. The OpenVINO NPU path uses its separate configured 1,024-token limit. These limits apply to different providers.

#### Troubleshooting empty NPU responses

Run Uvicorn in a visible terminal and check that terminal's output for `[NPU]` messages. The client uses Python logging, which Uvicorn normally emits to the console/stderr; the repository's `logs\\uvicorn.stderr.log` and `logs\\uvicorn.stdout.log` files are not automatically connected to the application logger and only contain current output if the process launcher redirects to them. Empty-response diagnostics report the prompt token count, pipeline length and response settings, OpenVINO result type, number of returned texts, generated-token metric (when available), and output length without logging the prompt or answer contents. If OpenVINO reports that the configured `max_new_tokens` limit was reached, Astro RAG marks the answer incomplete so the UI can warn that it may be truncated.

A local comparison harness can rebuild the `astro_sample_109` prompt using the actual `build_prompt()` function and compare it with the copied GPT-log prompt without printing either prompt or generated answer:

```powershell
python scripts/diagnose_npu_generation.py
python scripts/diagnose_npu_generation.py --client-smoke-test
```

Stop the Uvicorn process first so the standalone diagnostics do not compete for NPU memory. A successful short standalone prompt confirms that the model can generate in that test configuration, but it does not prove the longer `/ask` prompt and its config are equivalent. Compare the exact compacted `/ask` prompt, model directory, pipeline config, generation config, and list/string input shape before changing the exported INT4 group size or re-exporting the model. Keep the NPU option's strict error behavior; it does not silently fall back to another provider.

The separate chart-engine should be running on port `8000` for requests that contain birth details or place searches. Astro RAG's `POST /location/geocode` endpoint forwards explicit place searches to Chart Engine and caches successful results in memory for one hour (up to 128 queries); uncached searches are serialized to stay within the public Nominatim default's one-request-per-second policy. Configure Chart Engine's `GEOCODER_USER_AGENT` with an application identifier and contact before using its public geocoder. Alternatively, pass `chart_json` directly to `/ask`.

## How to replace the corpus later

To replace the four source PDFs with another edition or OCR version:

```text
1. Put the four PDFs into rag/source_pdfs/
2. Keep/update their BookConfig entries in scripts/build_corpus.py
3. Run: python scripts/build_corpus.py
4. Run: python scripts/verify_corpus.py
5. Run: python -m pytest -q
6. Restart uvicorn
```

You do **not** manually edit `rag/processed/corpus.jsonl`. It is generated output. The manifest stores a SHA-256 for every source PDF, so `verify_corpus.py` catches accidental mismatches between source PDFs and the processed corpus.

The application loads the corpus into memory at startup. After rebuilding the JSONL, restart the Astro RAG process so the new index is loaded.

## Verify the actual corpus from the API

```bash
curl -s http://localhost:8001/corpus/stats | python -m json.tool
curl -s http://localhost:8001/health | python -m json.tool
```

For retrieval-only debugging:

```bash
curl -s http://localhost:8001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "chart_facts": {"type": "planet_house", "planet": "Mars", "house": 10},
    "query": "How will my career be?",
    "strategy": "hybrid",
    "top_k": 5
  }' | python -m json.tool
```

## End-to-end `/ask`

Birth details + question:

```bash
curl -s http://localhost:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "birth_date": "2004-05-17",
    "birth_time": "14:35:00",
    "location": {
      "mode": "map_click",
      "latitude": 25.4358,
      "longitude": 81.8463,
      "display_name": "Prayagraj, India"
    },
    "settings": {
      "ayanamsa": "lahiri",
      "house_system": "whole_sign",
      "node_type": "mean"
    },
    "question": "When will I get my job?",
    "strategy": "hybrid",
    "top_k": 5,
    "response_model": "astro-qwen-lora"
  }' | python -m json.tool
```

`response_model` is optional and defaults to `gpt-5-mini`; use `astro-qwen-lora` only after installing the optional local model dependencies.

Already have chart JSON? Skip the chart-engine call:

```bash
curl -s http://localhost:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "chart_json": { ... },
    "question": "How is my personality?",
    "strategy": "hybrid",
    "top_k": 5
  }' | python -m json.tool
```

The response includes the answer, facts used, retrieved passages, source metadata, retrieval strategy, chart snapshot, model, and generation mode. `include_debug=true` adds the exact prompts passed to the generator.

### Streaming answers

`POST /ask/stream` accepts the same JSON request as `/ask` and streams Server-Sent Events. It sends `metadata` after chart/fact/retrieval preparation, `delta` events as visible answer text is generated, a `replace` event with the post-validated answer, and a final `done` event containing the complete `/ask`-shaped response. A mid-stream provider failure is reported as an `error` event; clients should treat `done` as authoritative.

```bash
curl -N http://localhost:8001/ask/stream \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "chart_json": { ... },
    "question": "How is my personality?",
    "strategy": "hybrid",
    "top_k": 5,
    "response_model": "astro-qwen-npu"
  }'
```

The browser chat uses `/ask/stream` and falls back to `/ask` when the server or a test mock does not provide SSE. The manual 40-question runner uses `/ask/stream` by default and prints question progress/status only; it never prints streamed answer text. Pass `--no-stream` to use the regular JSON endpoint. `/ask` itself remains available with its existing JSON response for scripts and evaluation clients.

## Retrieval smoke tests

```bash
python scripts/generate_gold_set.py
python rag/evaluation/evaluate_retrieval.py --strategy hybrid
python rag/evaluation/compare_strategies.py
```

These files are regression tools for the current corpus, **not a manually adjudicated scholarly relevance benchmark**. A retrieved chunk can be a valid passage even when it is not the one a human annotator would rank first.

## Dataset and NPU evaluation

Validate a training dataset without changing it; the validator reports schema,
duplicate, passage-reference, internal-reasoning, and possible chart-placement
issues:

```powershell
python rag/evaluation/validate_training_dataset.py path\to\training-data.json
```

After starting Astro RAG with the intended provider, run the 40-question NPU
check. The runner saves a timestamped JSON file under
`manual-test/results/` unless `--output` is supplied:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
python manual-test/run_npu_question_set.py path\to\chart.json
```

Evaluate the saved run and optionally compare it with a previous run:

```powershell
python rag/evaluation/evaluate_npu_run.py manual-test\results\npu-answers-TIMESTAMP.json --out manual-test\results\npu-evaluation-TIMESTAMP.json
python rag/evaluation/evaluate_npu_run.py manual-test\results\npu-answers-CURRENT.json --baseline manual-test\results\npu-answers-BASELINE.json
```

The report scores observable checks such as latency, citation references, and
reasoning-heading leaks. It does not claim interpretive accuracy without
adjudicated labels. Hash comparisons include comparable counts; missing hashes
are reported as not comparable rather than treated as matches.

## Security note

`.env` is intentionally not included in the repository/zip. Set your own `OPENAI_API_KEY` locally. Never commit it.

The uploaded project copy contained an API key in its local `.env`; that secret has been removed from this deliverable. Rotate/revoke that key in your OpenAI account before using the project again.


## Precision + Unicode fixes

The corpus builder now records explicit relationship pairs such as:

```text
planet_house_pairs: ["Mars|10"]
lord_house_pairs: ["5|10", "10|10"]
planet_sign_pairs: ["Sun|Aries"]
planet_nakshatra_pairs: ["Moon|Ashwini"]
planet_aspect_pairs: ["Mars|4th"]
```

Hybrid retrieval gives these explicit relationships a strong ranking bonus. Loose planet/house co-occurrence is retained only as a lower-ranked recall fallback. House-token spans allow own-house lord rules such as `10|10` to be indexed while avoiding false links to unrelated later planet placements in the same sentence. This keeps a rule such as `5th Lord in the 10th House` distinct from one that explicitly says `Mars ... in the 10th House`.

For chart-backed `/ask` retrieval, direct candidates are also checked against the supplied chart's known planet placements and house-lord placements. A compound passage is discounted when one of its indexed conditions is verified to be unmet; unavailable placement data is treated as unknown and is not penalized. Standalone `/retrieve` calls without a full chart retain the ordinary explicit-relation ranking. The prompt includes available planet placements and house lordships and adds a chart verification note to passages with contradicted conditions. NPU prompt fitting preserves those notes, and local Qwen output cleanup removes leaked verification tags and the known unsupported pseudo-yoga names `Dwipaka Yoga`, `Dvivida Yoga`, and `Drishti Yoga`.

The corpus normalization also applies Unicode NFC normalization, removes control characters, and includes a defensive UTF-8 mojibake repair path. The supplied PDFs' Devanagari text is already extractable as Unicode through PyMuPDF; `scripts/check_unicode.py` verifies that the generated corpus contains no common mojibake markers.

After pulling this version, repopulate the generated corpus with:

```bash
python scripts/build_corpus.py
python scripts/check_unicode.py
python scripts/verify_corpus.py
python -m pytest -q
```

Restart Astro RAG after rebuilding because the corpus is loaded into memory at startup.

### PowerShell display

If Devanagari looks like `à¤...` only in the Windows terminal while the corpus check passes, that is terminal rendering/encoding, not corpus corruption. Before inspecting JSON in PowerShell, run:

```powershell
chcp 65001
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

The browser UI or `curl.exe` can also be used to inspect the API response without PowerShell console encoding issues.
