# Manual NPU question run

Run all 40 questions sequentially against Astro RAG using the Intel NPU
provider. The runner uses the SSE `/ask/stream` endpoint by default and prints
question progress and generation status only; answer text is never printed.
It uses deterministic rule-based intent extraction with a previously generated
chart JSON file. Pass `--no-stream` to use the regular JSON `/ask` endpoint
instead.

From the `astro-rag-new` directory:

```powershell
python manual-test\run_npu_question_set.py path\to\chart.json
```

Optional arguments:

```powershell
python manual-test\run_npu_question_set.py path\to\chart.json `
  --base-url http://localhost:8001 `
  --output manual-test\results\my-run.json `
  --timeout 900
```

Both modes write the same timestamped JSON result format. Streaming is only a
transport change; the final saved answer comes from the completed response
event.

By default, the script writes a timestamped JSON file under `manual-test/results/`
and atomically saves progress after each question. Each item includes the question,
answer, provider and generation status, chart facts used, retrieved passages, and
elapsed time. HTTP errors are recorded and the run continues; connection failures
save partial results and stop. The output can contain chart-derived information
and retrieved text, so keep it local.

The runner reads UTF-8 chart JSON and also accepts legacy Windows-1252 files.
New chart exports from the Chart Engine CLI are written as UTF-8.
