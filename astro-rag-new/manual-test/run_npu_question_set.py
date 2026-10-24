"""Run the manual question set sequentially against Astro RAG's Intel NPU provider."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

QUESTIONS = [
    "Does my chart show financial comfort and support from people around me?",
    "People say Saturn in the wealth house ruins finances—will I be poor or can I build wealth?",
    "Do I have strong chances of traveling abroad or working with foreign organizations?",
    "Could marriage or a business partnership lead me to move to another country?",
    "Will I succeed in competitive entrance exams and higher education?",
    "I heard a debilitated Mercury hurts education—does my chart mean I will struggle in studies?",
    "Why is my current period considered so important for my career and professional growth?",
    "Is this a good time for a promotion or career advancement based on my running period?",
    "Can I succeed in media, design, communications, or starting my own venture?",
    "Why have I gone through sudden career changes, and what kind of work actually fits me?",
    "I have Ketu in my marriage house—does that mean my marriage will fail, or is it protected?",
    "Will I have a stable and happy marriage despite Saturn being in my 7th house?",
    "Does my chart suggest I will marry someone from a different cultural or social background?",
    "Will my spouse be financially successful and supportive?",
    "Why do I struggle to maintain long-term romantic relationships?",
    "Will I have children, and what does my chart say about my relationship with them?",
    "Will I have a son or daughter first?",
    "Does my chart indicate difficulty or delay in having children?",
    "Will my children be successful in life?",
    "Will I inherit property from my family, or will I buy a house on my own?",
    "Does my chart indicate owning multiple properties?",
    "Will I live comfortably in my own house or face domestic instability?",
    "What does my chart say about my relationship with my mother?",
    "Will I become wealthy through my career or through inheritance?",
    "Why do I experience financial ups and downs despite having a good income?",
    "Will I ever become extremely wealthy or a millionaire?",
    "Does my chart support success in business compared with a salaried career?",
    "Will I become famous or widely recognized for my work?",
    "What kind of career is most suitable for me according to my chart?",
    "Will I get a government job or work in the private sector?",
    "Will I change my career multiple times in life?",
    "Does my chart indicate success in technology or engineering?",
    "Will I have opportunities to work abroad?",
    "Will I settle permanently in a foreign country?",
    "Does my chart indicate a long life?",
    "Will I live past 85 years old according to my 8th house?",
    "Why do my friends always borrow money from me and never pay it back?",
    "Should I drop out of college to start a YouTube channel based on my 3rd house?",
    "Why does my mother's health keep fluctuating according to my chart?",
    "Can you tell me the exact name initial of my future spouse from my 7th house?",
]


def _write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(results, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_chart_json(path: Path) -> dict[str, Any]:
    try:
        raw_chart = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"Could not read chart JSON from {path}: {exc}") from exc

    try:
        chart_text = raw_chart.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            chart_text = raw_chart.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise SystemExit(
                f"Chart JSON is neither UTF-8 nor Windows-1252 encoded: {path}: {exc}"
            ) from exc
        print(
            "Notice: chart JSON is Windows-1252 encoded; for future exports, "
            "regenerate it with the updated chart CLI to produce UTF-8.",
            file=sys.stderr,
        )

    try:
        chart = json.loads(chart_text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse chart JSON from {path}: {exc}") from exc
    if not isinstance(chart, dict):
        raise SystemExit("The chart JSON root must be an object.")
    return chart


def _ask(
    base_url: str,
    chart: dict[str, Any],
    question: str,
    timeout_seconds: float,
    stream: bool = True,
) -> tuple[int, dict[str, Any]]:
    payload = {
        "chart_json": chart,
        "question": question,
        "top_k": 5,
        "strategy": "hybrid",
        "intent_mode": "rules",
        "response_model": "astro-qwen-npu",
        "include_debug": False,
    }
    if stream:
        request = Request(
            f"{base_url.rstrip('/')}/ask/stream",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            event_type = "message"
            data_lines: list[str] = []
            final_data: dict[str, Any] | None = None
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if data_lines:
                        event_data = json.loads("\n".join(data_lines))
                        if event_type == "metadata":
                            print(
                                "    Context ready; answer generation in progress.",
                                flush=True,
                            )
                        elif event_type == "done":
                            final_data = event_data
                        elif event_type == "error":
                            message = (
                                event_data.get("error", {}).get("message")
                                or "Streaming generation failed."
                            )
                            raise RuntimeError(message)
                    event_type = "message"
                    data_lines = []
                elif line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                event_data = json.loads("\n".join(data_lines))
                if event_type == "metadata":
                    print(
                        "    Context ready; answer generation in progress.",
                        flush=True,
                    )
                elif event_type == "done":
                    final_data = event_data
                elif event_type == "error":
                    error_data = event_data.get("error")
                    message = (
                        error_data.get("message")
                        if isinstance(error_data, dict)
                        else None
                    )
                    raise RuntimeError(message or "Streaming generation failed.")
            if not isinstance(final_data, dict):
                raise ValueError("Stream ended without a final `done` payload.")
            return response.status, final_data

    request = Request(
        f"{base_url.rstrip('/')}/ask",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("The /ask response was not a JSON object.")
        return response.status, data


def run(
    chart_path: Path,
    output_path: Path,
    base_url: str = "http://localhost:8001",
    timeout_seconds: float = 900,
    stream: bool = True,
) -> int:
    chart = _load_chart_json(chart_path)
    chart_sha256 = hashlib.sha256(
        json.dumps(chart, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    question_set_sha256 = hashlib.sha256(
        json.dumps(QUESTIONS, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_started_at = datetime.now().astimezone().isoformat()

    results: list[dict[str, Any]] = []
    _write_results(output_path, results)
    for index, question in enumerate(QUESTIONS, start=1):
        print(f"[{index}/{len(QUESTIONS)}] Sending question to NPU...")
        started_at = time.perf_counter()
        record: dict[str, Any] = {
            "index": index,
            "question": question,
            "chart_sha256": chart_sha256,
            "question_set_sha256": question_set_sha256,
            "run_started_at": run_started_at,
            "streaming_requested": stream,
        }
        try:
            status_code, response_data = _ask(
                base_url,
                chart,
                question,
                timeout_seconds,
                stream=stream,
            )
            record.update(
                {
                    "http_status": status_code,
                    "answer": response_data.get("answer"),
                    "model": response_data.get("model"),
                    "generation_mode": response_data.get("generation_mode"),
                    "generation_status": response_data.get("generation_status"),
                    "generation_incomplete_reason": response_data.get(
                        "generation_incomplete_reason"
                    ),
                    "generation_usage": response_data.get("generation_usage"),
                    "intent_source": response_data.get("intent_source"),
                    "chart_facts_used": response_data.get("chart_facts_used"),
                    "retrieved_passages": response_data.get("retrieved_passages"),
                }
            )
            record["provider_mismatch"] = (
                response_data.get("generation_mode") != "openvino_npu"
            )
            print(
                f"    HTTP {status_code}; mode={response_data.get('generation_mode')}; "
                f"status={response_data.get('generation_status')}"
            )
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            try:
                error_detail: Any = json.loads(error_body)
            except json.JSONDecodeError:
                error_detail = error_body
            record.update(
                {
                    "http_status": exc.code,
                    "error": error_detail,
                }
            )
            print(f"    HTTP {exc.code}; error recorded.")
        except (URLError, TimeoutError, OSError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_seconds"] = round(time.perf_counter() - started_at, 2)
            results.append(record)
            _write_results(output_path, results)
            print(f"    Request failed; partial results saved to {output_path}")
            raise SystemExit(
                "Could not reach the Astro RAG server or the request timed out. "
                "Check the server and rerun the script to continue a fresh test."
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print("    Invalid response; error recorded.")
        record["elapsed_seconds"] = round(time.perf_counter() - started_at, 2)
        results.append(record)
        _write_results(output_path, results)

    errors = sum("error" in result for result in results)
    provider_mismatches = sum(
        result.get("provider_mismatch", False) for result in results
    )
    print(
        f"Finished {len(results)} questions: {errors} request errors, "
        f"{provider_mismatches} provider mismatches."
    )
    print(f"Results saved to: {output_path}")
    return 1 if errors or provider_mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chart_json",
        type=Path,
        help="Path to a chart JSON file previously generated by Chart Engine.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8001",
        help="Astro RAG API base URL (default: http://localhost:8001).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: manual-test/results/npu-answers-TIMESTAMP.json).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="Per-question HTTP timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use the non-streaming /ask endpoint (answer text is never printed).",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_path = args.output or (
        Path(__file__).resolve().parent
        / "results"
        / f"npu-answers-{timestamp}.json"
    )
    return run(
        args.chart_json.expanduser().resolve(),
        output_path.expanduser().resolve(),
        args.base_url,
        args.timeout,
        stream=not args.no_stream,
    )


if __name__ == "__main__":
    sys.exit(main())
