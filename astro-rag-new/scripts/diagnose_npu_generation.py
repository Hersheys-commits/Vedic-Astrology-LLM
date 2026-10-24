"""Compare NPU generation variants without printing prompt or answer contents."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings  # noqa: E402
from app.generation.openvino_npu_client import (  # noqa: E402
    NPU_SYSTEM_PROMPT,
    OpenVINONPUClient,
)
from app.generation.prompt import SYSTEM_PROMPT, build_prompt  # noqa: E402

logger = logging.getLogger("npu_diagnostic")


def _load_gpt_log_user_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^user\n(?=CHART SNAPSHOT\n)", text)
    if match is None:
        raise ValueError(
            f"Could not find a user prompt beginning with CHART SNAPSHOT in {path}"
        )
    return text[match.end():].rstrip() + "\n"


def _ordinal_house_fact(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(\d+)(?:st|nd|rd|th) house:\s*([A-Za-z]+)", line)
    return (int(match.group(1)), match.group(2)) if match else None


def _example_prompt(example: dict[str, Any]) -> tuple[str, str]:
    chart_summary: dict[str, Any] = {}
    chart_facts: list[dict[str, Any]] = []

    for line in example.get("chart_facts", []):
        ascendant = re.match(r"^Ascendant:\s*([A-Za-z]+)", line)
        if ascendant:
            chart_summary["ascendant_sign"] = ascendant.group(1)
            continue

        house = _ordinal_house_fact(line)
        if house:
            chart_facts.append(
                {"type": "house_lookup", "house": house[0], "sign": house[1]}
            )
            continue

        lord = re.match(
            r"^(\d+)(?:st|nd|rd|th) lord:\s*([A-Za-z]+)\s+placed in "
            r"(\d+)(?:st|nd|rd|th) house",
            line,
        )
        if lord:
            chart_facts.append(
                {
                    "type": "planet_house",
                    "planet": lord.group(2),
                    "house": int(lord.group(3)),
                    "role": f"lord of house {lord.group(1)}",
                }
            )
            continue

        placement = re.match(
            r"^([A-Za-z]+):\s*Placed in (\d+)(?:st|nd|rd|th) house",
            line,
        )
        if placement:
            chart_facts.append(
                {
                    "type": "planet_house",
                    "planet": placement.group(1),
                    "house": int(placement.group(2)),
                }
            )

    passages = []
    for context in example.get("retrieved_context", []):
        passages.append(
            {
                "text": context["text"],
                "metadata": {
                    "source": {"book": context.get("source")},
                },
                "match_type": "training-example",
            }
        )

    chart = {"chart_summary": chart_summary}
    return build_prompt(
        chart,
        chart_facts,
        passages,
        example["question"],
    )


def _metrics(result: Any) -> tuple[str, int, int | None, str]:
    texts = getattr(result, "texts", None)
    if isinstance(texts, str):
        count = 1
        text = texts
    elif texts is not None:
        count = len(texts)
        text = texts[0] if count else ""
    else:
        count = 1 if isinstance(result, str) else 0
        text = result if isinstance(result, str) else ""

    generated_tokens = None
    perf_metrics = getattr(result, "perf_metrics", None)
    get_count = getattr(perf_metrics, "get_num_generated_tokens", None)
    if callable(get_count):
        generated_tokens = get_count()
    return type(result).__name__, count, generated_tokens, text if isinstance(text, str) else ""


def _run_case(
    client: OpenVINONPUClient,
    label: str,
    prompt: str,
    input_shape: str,
    config_mode: str,
    template_mode: str,
) -> dict[str, Any]:
    pipeline = client._pipeline
    get_config = getattr(pipeline, "get_generation_config", None)
    if not callable(get_config):
        raise RuntimeError("This OpenVINO pipeline does not expose get_generation_config().")

    config = get_config()
    default_sampling = getattr(config, "do_sample", None)
    config.max_new_tokens = 256
    if config_mode == "greedy":
        config.do_sample = False
    elif config_mode != "pipeline-default":
        raise ValueError(f"Unknown config mode: {config_mode}")

    default_template = getattr(config, "apply_chat_template", None)
    if template_mode == "disabled":
        if not hasattr(config, "apply_chat_template"):
            return {
                "case": label,
                "skipped": "GenerationConfig has no apply_chat_template setting",
            }
        config.apply_chat_template = False

    input_value: Any = [prompt] if input_shape == "list" else prompt
    started_at = perf_counter()
    result = pipeline.generate(input_value, config)
    elapsed = perf_counter() - started_at
    result_type, returned_text_count, generated_tokens, text = _metrics(result)
    if "</think>" in text:
        visible_text = text.split("</think>", 1)[-1].strip()
    elif re.match(r"^\s*<think(?:\s[^>]*)?>", text):
        visible_text = ""
    else:
        visible_text = re.sub(r"</?think(?:\s[^>]*)?>", "", text).strip()
    record = {
        "case": label,
        "input_shape": input_shape,
        "config_mode": config_mode,
        "template_mode": template_mode,
        "pipeline_default_do_sample": default_sampling,
        "pipeline_default_apply_chat_template": default_template,
        "effective_do_sample": getattr(config, "do_sample", None),
        "effective_apply_chat_template": getattr(config, "apply_chat_template", None),
        "result_type": result_type,
        "returned_text_count": returned_text_count,
        "generated_tokens": generated_tokens,
        "raw_output_chars": len(text),
        "visible_output_chars_after_app_filter": len(visible_text),
        "has_think_open": "<think>" in text,
        "has_think_close": "</think>" in text,
        "elapsed_seconds": round(elapsed, 2),
    }
    print(json.dumps(record, ensure_ascii=True), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROJECT_ROOT.parent / "astro_finetuning_160_examples.json",
    )
    parser.add_argument(
        "--gpt-log",
        type=Path,
        default=PROJECT_ROOT.parent / "prompt_copied_from_gpt_logs.txt",
    )
    parser.add_argument("--example-id", default="astro_sample_109")
    parser.add_argument(
        "--python-template-only",
        action="store_true",
        help="Run only Python-chat-template prompts with OpenVINO templating disabled.",
    )
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="Run short-prompt and fixed-context controls only.",
    )
    parser.add_argument(
        "--no-template-only",
        action="store_true",
        help="Run plain fitted prompts with the OpenVINO chat template disabled.",
    )
    parser.add_argument(
        "--client-smoke-test",
        action="store_true",
        help="Run the matching dataset example through OpenVINONPUClient.generate().",
    )
    args = parser.parse_args()

    examples = json.loads(args.examples.read_text(encoding="utf-8"))
    example = next(
        (item for item in examples if item.get("id") == args.example_id),
        None,
    )
    if example is None:
        raise ValueError(f"Example id not found in {args.examples}: {args.example_id}")

    client = OpenVINONPUClient()
    client._ensure_loaded()
    tokenizer = client._tokenizer
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Loaded tokenizer does not support encode().")

    copied_user_prompt = _load_gpt_log_user_prompt(args.gpt_log)
    copied_prompt = client._fit_prompt_to_npu_budget(
        SYSTEM_PROMPT,
        copied_user_prompt,
    )
    sample_system_prompt, sample_user_prompt = _example_prompt(example)
    sample_prompt = client._fit_prompt_to_npu_budget(
        sample_system_prompt,
        sample_user_prompt,
    )
    print(
        json.dumps(
            {
                "model_dir": str(settings.openvino_npu_model_dir),
                "openvino_npu_max_prompt_len": settings.openvino_npu_max_prompt_len,
                "gpt_log_user_prompt_tokens_after_app_fitting": len(
                    encode(copied_prompt, add_special_tokens=False)
                ),
                "dataset_example_id": example["id"],
                "dataset_domain": example.get("domain"),
                "dataset_question_matches_gpt_log": (
                    example["question"].strip()
                    in copied_user_prompt
                ),
                "dataset_chart_fact_count": len(example.get("chart_facts", [])),
                "dataset_retrieved_context_count": len(
                    example.get("retrieved_context", [])
                ),
                "dataset_prompt_tokens_after_app_fitting": len(
                    encode(sample_prompt, add_special_tokens=False)
                ),
                "prompt_contents_logged": False,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    if args.python_template_only:
        for label, prompt in (
            ("copied_gpt_log_prompt", copied_prompt),
            ("dataset_example_app_prompt", sample_prompt),
        ):
            _, compact_user = prompt.split("\n\n", 1)
            user_start = compact_user.find("CHART SNAPSHOT\n")
            if user_start < 0:
                raise RuntimeError(
                    f"Could not recover compacted user prompt for {label}."
                )
            compact_user = compact_user[user_start:]
            chat_formatted_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": NPU_SYSTEM_PROMPT},
                    {"role": "user", "content": compact_user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            print(
                json.dumps(
                    {
                        "case": f"{label}_python_chat_template_prompt",
                        "prompt_tokens": len(
                            encode(chat_formatted_prompt, add_special_tokens=False)
                        ),
                        "prompt_contents_logged": False,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            _run_case(
                client,
                f"{label}_list_greedy_python_template_native_disabled",
                chat_formatted_prompt,
                "list",
                "greedy",
                "disabled",
            )
        return

    if args.controls_only:
        for input_shape in ("string", "list"):
            _run_case(
                client,
                f"hello_current_config_{input_shape}_native",
                "Hello",
                input_shape,
                "greedy",
                "native",
            )

        effective_system, compact_user = copied_prompt.split("\n\n", 1)
        passages_marker = "RETRIEVED REFERENCE PASSAGES\n"
        question_marker = "\nQUESTION\n"
        prefix, rest = compact_user.split(passages_marker, 1)
        _, question_suffix = rest.rsplit(question_marker, 1)
        fixed_context_prompt = (
            f"{effective_system}\n\n{prefix}{passages_marker}"
            f"{question_marker}{question_suffix}"
        )
        print(
            json.dumps(
                {
                    "case": "copied_prompt_without_passage_bodies",
                    "prompt_tokens": len(
                        encode(fixed_context_prompt, add_special_tokens=False)
                    ),
                    "prompt_contents_logged": False,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        _run_case(
            client,
            "fixed_context_current_config_list_native",
            fixed_context_prompt,
            "list",
            "greedy",
            "native",
        )
        return

    if args.no_template_only:
        for label, prompt in (
            ("copied_gpt_log_prompt", copied_prompt),
            ("dataset_example_app_prompt", sample_prompt),
        ):
            _run_case(
                client,
                f"{label}_plain_list_greedy_template_disabled",
                prompt,
                "list",
                "greedy",
                "disabled",
            )
        return

    if args.client_smoke_test:
        answer = client.generate(sample_system_prompt, sample_user_prompt)
        print(
            json.dumps(
                {
                    "case": "dataset_example_openvino_npu_client",
                    "answer_chars": len(answer),
                    "generation_usage": client.last_diagnostics.get("usage"),
                    "prompt_contents_logged": False,
                    "answer_contents_logged": False,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return

    if settings.openvino_npu_max_prompt_len != 1024:
        raise RuntimeError(
            "Diagnostic expects OPENVINO_NPU_MAX_PROMPT_LEN=1024; "
            f"found {settings.openvino_npu_max_prompt_len}."
        )

    # This control matches the successful standalone CLI call shape.
    standalone_started_at = perf_counter()
    standalone_result = client._pipeline.generate("Hello", max_new_tokens=32)
    standalone_elapsed = perf_counter() - standalone_started_at
    result_type, text_count, generated_tokens, text = _metrics(standalone_result)
    print(
        json.dumps(
            {
                "case": "standalone_hello_string_kwargs",
                "result_type": result_type,
                "returned_text_count": text_count,
                "generated_tokens": generated_tokens,
                "raw_output_chars": len(text),
                "elapsed_seconds": round(standalone_elapsed, 2),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    for label, prompt in (
        ("copied_gpt_log_prompt", copied_prompt),
        ("dataset_example_app_prompt", sample_prompt),
    ):
        _run_case(client, f"{label}_list_greedy_native", prompt, "list", "greedy", "native")
        _run_case(client, f"{label}_string_greedy_native", prompt, "string", "greedy", "native")
        _run_case(
            client,
            f"{label}_list_pipeline_sampling_native",
            prompt,
            "list",
            "pipeline-default",
            "native",
        )
        _run_case(
            client,
            f"{label}_list_greedy_template_disabled",
            prompt,
            "list",
            "greedy",
            "disabled",
        )
        _, compact_user = prompt.split("\n\n", 1)
        user_start = compact_user.find("CHART SNAPSHOT\n")
        if user_start < 0:
            raise RuntimeError(f"Could not recover compacted user prompt for {label}.")
        compact_user = compact_user[user_start:]
        chat_formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": NPU_SYSTEM_PROMPT},
                {"role": "user", "content": compact_user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(
            json.dumps(
                {
                    "case": f"{label}_python_chat_template_prompt",
                    "prompt_tokens": len(
                        encode(chat_formatted_prompt, add_special_tokens=False)
                    ),
                    "prompt_contents_logged": False,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        _run_case(
            client,
            f"{label}_list_greedy_python_template_native_disabled",
            chat_formatted_prompt,
            "list",
            "greedy",
            "disabled",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
