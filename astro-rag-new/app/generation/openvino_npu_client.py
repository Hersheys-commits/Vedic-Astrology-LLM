"""Lazy OpenVINO GenAI inference on the Intel NPU."""
from __future__ import annotations

import gc
import hashlib
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from app.config.settings import settings
from app.generation.prompt import CORE_GROUNDING_POLICY
from app.generation.qwen_client import (
    ADAPTER_MODEL,
    _clean_model_output,
    _stream_visible_text,
)
from app.utils.errors import RAGError

logger = logging.getLogger(__name__)


def _prompt_token_budget() -> int:
    return max(1, settings.openvino_npu_max_prompt_len - 128)


def _allocate_passage_token_limits(
    entries: list[tuple[str, list[int], str]],
    total_tokens: int,
) -> list[int]:
    """Allocate passage tokens by rank, redistributing unused capped capacity."""
    if not entries or total_tokens <= 0:
        return [0] * len(entries)

    capacities = [len(body_ids) for _, body_ids, _ in entries]
    limits = [0] * len(entries)
    remaining = min(total_tokens, sum(capacities))
    active = [index for index, capacity in enumerate(capacities) if capacity]
    weights = [5, 4] + [2] * max(0, len(entries) - 2)

    while remaining and active:
        weight_sum = sum(weights[index] for index in active)
        shares = {
            index: remaining * weights[index] // weight_sum
            for index in active
        }
        unassigned = remaining - sum(shares.values())
        for index in active[:unassigned]:
            shares[index] += 1

        allocated = 0
        next_active = []
        for index in active:
            amount = min(capacities[index] - limits[index], shares[index])
            limits[index] += amount
            allocated += amount
            if limits[index] < capacities[index]:
                next_active.append(index)
        if not allocated:
            break
        remaining -= allocated
        active = next_active

    return limits


def _model_artifact_fingerprint(model_dir: Path) -> str:
    """Cheap manifest hash to detect changed local exports between test runs.

    This intentionally hashes names, sizes and modification times rather than
    reading multi-gigabyte weight tensors. It is a reproducibility diagnostic,
    not a cryptographic proof that two model files contain identical weights.
    """
    digest = hashlib.sha256()
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or ".npucache" in path.parts:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        record = f"{path.relative_to(model_dir).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()[:16]


def _generation_config_snapshot(config: Any) -> dict[str, Any]:
    stop_strings = getattr(config, "stop_strings", ())
    if isinstance(stop_strings, str):
        normalized_stop_strings = [stop_strings]
    else:
        try:
            normalized_stop_strings = sorted(str(value) for value in stop_strings)
        except TypeError:
            normalized_stop_strings = []
    return {
        "max_new_tokens": getattr(config, "max_new_tokens", None),
        "do_sample": bool(getattr(config, "do_sample", False)),
        "repetition_penalty": getattr(config, "repetition_penalty", None),
        "apply_chat_template": getattr(config, "apply_chat_template", None),
        "include_stop_str_in_output": getattr(
            config, "include_stop_str_in_output", None
        ),
        "stop_strings": normalized_stop_strings,
    }


NPU_SYSTEM_PROMPT = f"""\
You are a careful Vedic astrology (Jyotisha) assistant. Explain traditional interpretations,
not scientific facts or guaranteed predictions.

{CORE_GROUNDING_POLICY}

DIGNITY: Exalted—Sun Aries, Moon Taurus, Mars Capricorn, Mercury Virgo, Jupiter Cancer,
Venus Pisces, Saturn Libra. Debilitated—Sun Libra, Moon Scorpio, Mars Cancer, Mercury Pisces,
Jupiter Capricorn, Venus Virgo, Saturn Aries. Own—Sun Leo, Moon Cancer, Mars Aries/Scorpio,
Mercury Gemini/Virgo, Jupiter Sagittarius/Pisces, Venus Taurus/Libra, Saturn Capricorn/Aquarius.
All other signs are ordinary.

STYLE: Answer the question first. Narrow facts: 2–5 sentences; ordinary interpretation:
80–140 words; broad life-area: 120–180 words. Be concise, cite relevant sources, and hedge
traditional tendencies. Do not pad, add generic disclaimers unless relevant, or output
internal reasoning. Do not diagnose health conditions or predict exact lifespan, a child's
sex, or a future spouse's exact initials."""


class OpenVINONPUClient:
    """Run the prepared Astro Qwen model on NPU without a CPU fallback."""

    mode = "openvino_npu"
    model = ADAPTER_MODEL
    device = "NPU"

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._tokenizer: Any = None
        self._load_lock = Lock()
        self._generation_lock = Lock()
        self._model_fingerprint: str | None = None
        self.last_diagnostics: dict[str, Any] = {
            "status": "not_called",
            "incomplete_reason": None,
            "usage": None,
        }

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._pipeline is not None and self._tokenizer is not None:
                return

            model_dir = Path(settings.openvino_npu_model_dir)
            if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_npu_model_missing",
                    "The OpenVINO NPU model is not prepared. Run "
                    "`python scripts/prepare_npu_model.py` from astro-rag-new "
                    "and check OPENVINO_NPU_MODEL_DIR.",
                    503,
                )

            self._model_fingerprint = _model_artifact_fingerprint(model_dir)
            try:
                import openvino as ov
                import openvino_genai
                from transformers import AutoTokenizer
            except ImportError as exc:
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_npu_dependencies_missing",
                    "Astro Qwen NPU inference needs OpenVINO GenAI and Transformers. "
                    "Install requirements-openvino-npu.txt, then restart Astro RAG.",
                    503,
                ) from exc

            try:
                gc.collect()
                core = ov.Core()
                if self.device not in core.available_devices:
                    raise RAGError(
                        "qwen_npu_device_unavailable",
                        "OpenVINO cannot see an NPU device. Install or update the "
                        "Intel NPU driver, restart Windows, and confirm `NPU` appears "
                        "in OpenVINO Core.available_devices.",
                        503,
                    )
                cache_dir = model_dir / ".npucache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                pipeline_config = {
                    "MAX_PROMPT_LEN": settings.openvino_npu_max_prompt_len,
                    "MIN_RESPONSE_LEN": settings.openvino_npu_max_new_tokens,
                    "CACHE_DIR": str(cache_dir),
                }
                pipeline = openvino_genai.LLMPipeline(
                    str(model_dir),
                    self.device,
                    pipeline_config,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                )
                self._pipeline = pipeline
                self._tokenizer = tokenizer
            except RAGError:
                self.last_diagnostics["status"] = "error"
                raise
            except Exception as exc:
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_npu_model_load_failed",
                    "Could not load Astro Qwen on the Intel NPU. Check the converted "
                    "OpenVINO model, NPU driver/runtime compatibility, and available "
                    f"shared memory. Details: {exc}",
                    503,
                ) from exc

    def _fit_prompt_to_npu_budget(self, system_prompt: str, user_prompt: str) -> str:
        effective_system = (
            NPU_SYSTEM_PROMPT if len(system_prompt) > 2000 else system_prompt
        )
        passages_marker = "RETRIEVED REFERENCE PASSAGES\n"
        question_marker = "\nQUESTION\n"

        def clean_passages(text: str) -> str:
            text = re.sub(r"[\u0900-\u097F\u200b-\u200d]+", " ", text)
            cleaned_lines = []
            for line in text.splitlines():
                line = re.sub(r"[ \t]{2,}", " ", line).strip()
                line = re.sub(
                    r"^(?:Sioku|Sloku|Sloka|Slipper)\s+[A-Z0-9]+"
                    r"(?:/[A-Z0-9]+)?\.?\s*[-—:]+\s*",
                    "",
                    line,
                    flags=re.IGNORECASE,
                )
                if not line:
                    continue
                if line.startswith("[Passage "):
                    cleaned_lines.append(line)
                    continue
                if line.startswith(
                    ("[Chart Binding:", "[Chart Verification Note:")
                ):
                    cleaned_lines.append(line)
                    continue
                if len(line) < 18 and not re.search(
                    r"\b(if|when|the|in|of|and|to|is|are|will|should|house|lord|"
                    r"sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu|"
                    r"ascendant|lagna|effects|chapter|sloka|rule)\b",
                    line,
                    re.IGNORECASE,
                ):
                    continue
                alpha_count = sum(character.isalpha() for character in line)
                if len(line) > 15 and alpha_count / len(line) < 0.35:
                    continue
                cleaned_lines.append(line)
            return "\n".join(cleaned_lines)

        has_passage_section = passages_marker in user_prompt and question_marker in user_prompt
        if has_passage_section:
            prefix, rest = user_prompt.split(passages_marker, 1)
            passages_block, question_suffix = rest.rsplit(question_marker, 1)
            prefix = self._compact_fixed_context(prefix)
            passages_block = clean_passages(passages_block)
            passage_entries = [
                entry.strip()
                for entry in re.split(r"(?=\[Passage \d+[^\]\n]*\])", passages_block)
                if entry.strip()
            ]
            trimmed_entries = []
            for entry in passage_entries:
                parts = re.split(r"\bNotes\s*:", entry, maxsplit=1)
                main_part = parts[0].rstrip()
                if len(parts) > 1 and len(main_part.splitlines()) <= 3:
                    note = parts[1].strip()
                    preview = note[:240].rstrip()
                    if len(note) > len(preview):
                        preview = preview.rsplit(" ", 1)[0].rstrip() + " ..."
                    trimmed_entries.append(f"{main_part}\nNote: {preview}")
                else:
                    trimmed_entries.append(main_part)
            passage_entries = trimmed_entries
        else:
            prefix = user_prompt
            question_suffix = ""
            passage_entries = []

        def render(content: str) -> str:
            chat_template = getattr(self._tokenizer, "chat_template", None)
            apply_chat_template = getattr(
                self._tokenizer, "apply_chat_template", None
            )
            if isinstance(chat_template, str) and callable(apply_chat_template):
                messages = [
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": content},
                ]
                try:
                    rendered = apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    rendered = apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str) and not rendered.endswith(
                        "</think>\n\n"
                    ):
                        rendered += "<think>\n\n</think>\n\n"
                if not isinstance(rendered, str):
                    raise RAGError(
                        "qwen_npu_chat_template_invalid",
                        "The tokenizer chat template did not return a string prompt.",
                        503,
                    )
                if re.search(r"<\|im_start\|>assistant\s*$", rendered):
                    rendered = f"{rendered}<think>\n\n</think>\n\n"
                elif re.search(
                    r"<\|im_start\|>assistant\s*<think>\s*$",
                    rendered,
                    flags=re.IGNORECASE,
                ):
                    rendered = f"{rendered}</think>\n\n"
                return rendered
            return f"{effective_system}\n\n{content}"

        encode = getattr(self._tokenizer, "encode", None)
        if not callable(encode):
            raise RAGError(
                "qwen_npu_tokenizer_invalid",
                "The NPU tokenizer does not support token counting, so the prompt "
                "cannot be safely checked against the configured token budget.",
                503,
            )

        max_tokens = _prompt_token_budget()

        def token_count(prompt: str) -> int:
            return len(encode(prompt, add_special_tokens=False))

        if not has_passage_section:
            prompt = render(user_prompt)
            if token_count(prompt) > max_tokens:
                raise RAGError(
                    "qwen_npu_prompt_too_long",
                    "The prompt exceeds the configured NPU token budget and has "
                    "no retrieved-passages section that can be safely shortened.",
                    422,
                )
            return prompt

        joined_passages = "\n\n".join(passage_entries)
        full_passages_user = (
            f"{prefix}{passages_marker}{joined_passages}"
            f"{question_marker}{question_suffix}"
        )
        full_prompt = render(full_passages_user)
        if token_count(full_prompt) <= max_tokens:
            return full_prompt

        fixed_user = f"{prefix}{passages_marker}{question_marker}{question_suffix}"
        fixed_prompt = render(fixed_user)
        if token_count(fixed_prompt) > max_tokens:
            raise RAGError(
                "qwen_npu_prompt_too_long",
                "The chart facts and question exceed the configured NPU token "
                "budget even without retrieved passages. Shorten the question or "
                "reduce the chart context.",
                422,
            )

        headers_and_bodies: list[tuple[str, list[int], str]] = []
        for entry in passage_entries:
            match = re.match(r"(\[Passage \d+[^\]\n]*\])(?:\n|$)", entry)
            if match:
                header = match.group(1)
                body = entry[match.end():].strip()
            else:
                header = entry.splitlines()[0]
                body = "\n".join(entry.splitlines()[1:]).strip()
            header = re.sub(
                r"\s*\|\s*Chapter (\d+) — [^|\]]+",
                r" | Chapter \1",
                header,
            )
            header = re.sub(r"\s*\|\s*PDF p\. [^,\]]+", "", header)
            header = re.sub(r", match: [^\]]+", "", header)
            header_match = re.match(
                r"^\[Passage\s+(\d+)\b.*?\bsource:\s*(.*?)\]$",
                header,
            )
            if header_match:
                source_label = header_match.group(2).split("|", 1)[0]
                source_label = re.sub(r",\s*match:.*$", "", source_label).strip()
                header = (
                    f"[Passage {header_match.group(1)} — {source_label}]"
                )
            verification_notes = re.findall(
                r"\[Chart (?:Binding|Verification Note):[^\]]*\]", body
            )
            body = re.sub(
                r"\[Chart (?:Binding|Verification Note):[^\]]*\]",
                "",
                body,
            ).strip()
            note_suffix = (
                "\n" + "\n".join(verification_notes)
                if verification_notes
                else ""
            )
            headers_and_bodies.append(
                (header, encode(body, add_special_tokens=False), note_suffix)
            )

        def build_headers_prompt(
            items: list[tuple[str, list[int], str]],
        ) -> str:
            headers_only = "\n\n".join(
                f"{header}{note_suffix}"
                for header, _, note_suffix in items
            )
            return render(
                f"{prefix}{passages_marker}{headers_only}"
                f"{question_marker}{question_suffix}"
            )

        headers_prompt = build_headers_prompt(headers_and_bodies)
        while token_count(headers_prompt) > max_tokens and len(headers_and_bodies) > 1:
            dropped_header = headers_and_bodies.pop()[0]
            logger.warning(
                "NPU prompt budget: dropped lower-ranked passage header %s; "
                "remaining_passages=%d",
                dropped_header,
                len(headers_and_bodies),
            )
            headers_prompt = build_headers_prompt(headers_and_bodies)

        if token_count(headers_prompt) > max_tokens:
            fixed_tokens = token_count(headers_prompt)
            logger.warning(
                "NPU fixed prompt context exceeds budget after passage reduction "
                "(tokens=%d budget=%d passages=%d)",
                fixed_tokens,
                max_tokens,
                len(headers_and_bodies),
            )
            raise RAGError(
                "qwen_npu_prompt_too_long",
                "The compacted chart facts, question, and passage citations need "
                f"{fixed_tokens} tokens, exceeding the configured prompt budget "
                f"of {max_tokens}. Increase OPENVINO_NPU_MAX_PROMPT_LEN if the "
                "exported model supports it, or reduce the chart context.",
                422,
            )

        def render_passages(total_body_token_limit: int) -> str:
            entries = []
            limits = _allocate_passage_token_limits(
                headers_and_bodies,
                total_body_token_limit,
            )
            for (header, body_ids, note_suffix), body_token_limit in zip(
                headers_and_bodies,
                limits,
            ):
                body = self._tokenizer.decode(
                    body_ids[:body_token_limit], skip_special_tokens=True
                ).rstrip()
                if len(body_ids) > body_token_limit and body:
                    body += " ..."
                body += note_suffix
                entries.append(f"{header}\n{body}" if body else header)
            passage_text = "\n\n".join(entries)
            return render(
                f"{prefix}{passages_marker}{passage_text}"
                f"{question_marker}{question_suffix}"
            )

        low = 0
        high = sum(len(body_ids) for _, body_ids, _ in headers_and_bodies)
        best_prompt = headers_prompt
        while low <= high:
            total_body_token_limit = (low + high) // 2
            candidate_prompt = render_passages(total_body_token_limit)
            if token_count(candidate_prompt) <= max_tokens:
                best_prompt = candidate_prompt
                low = total_body_token_limit + 1
            else:
                high = total_body_token_limit - 1

        if token_count(best_prompt) > max_tokens:
            raise RAGError(
                "qwen_npu_prompt_too_long",
                "Could not fit the passage headers within the configured NPU "
                "token budget.",
                422,
            )
        logger.info(
            "NPU prompt fitted: tokens=%d budget=%d passages=%d",
            token_count(best_prompt),
            max_tokens,
            len(headers_and_bodies),
        )
        return best_prompt

    @staticmethod
    def _compact_fixed_context(prefix: str) -> str:
        """Remove repeated wording while retaining chart placements and relations."""
        compacted_lines = []
        for line in prefix.splitlines():
            if line.startswith("Houses: "):
                line = re.sub(
                    r"\b(H\d+):([^()\s]+)\(([^()/]+)(?:/([^()]+))?\)",
                    lambda match: (
                        f"{match.group(1)} sign={match.group(2)} "
                        f"lord={match.group(3)} "
                        f"occupants={match.group(4) or 'none'}"
                    ),
                    line,
                )
            elif line.startswith("Planets: "):
                line = re.sub(
                    r"\b([\w-]+):H(\d+)/([^[]+)"
                    r"(?:\[lord of ([^]]+)\])?",
                    lambda match: (
                        f"{match.group(1)} sign={match.group(3)} "
                        f"house=H{match.group(2)}"
                        + (
                            f" lord_of={match.group(4)}"
                            if match.group(4)
                            else ""
                        )
                    ),
                    line,
                )
            elif line.startswith("- "):
                line = re.sub(
                    r" — (?P<planet>[\w-]+) is placed in house "
                    r"(?P<house>\d+)(?: \(in (?P<sign>[^)]+)\))?",
                    lambda match: (
                        f"; in {match.group('sign')}"
                        if match.group("sign")
                        else ""
                    ),
                    line,
                )
                line = re.sub(
                    r" — (?P<planet>[\w-]+) is placed in "
                    r"(?P<sign>[^()]+)",
                    "",
                    line,
                )
                line = re.sub(r" in this chart", "", line)
                line = re.sub(r" \(it is the [^)]+\)", "", line)
                line = re.sub(
                    r"^-\s*(H\d+):\s*([^—]+) — House \d+ is [^,]+",
                    r"- \1: \2",
                    line,
                )
                line = re.sub(r", ruled by ([^,.;]+)", r"; lord \1", line)
                line = re.sub(
                    r", occupied by (.+)\.$",
                    r"; occupants \1",
                    line,
                )
                line = re.sub(
                    r", empty / no planets occupying this house\.",
                    "; no occupants",
                    line,
                )
                line = re.sub(
                    r" \(current running mahadasha\)",
                    "",
                    line,
                )
                line = re.sub(r"\s{2,}", " ", line).strip()
            compacted_lines.append(line)
        return "\n".join(compacted_lines)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        self._ensure_loaded()
        try:
            prompt = self._fit_prompt_to_npu_budget(system_prompt, user_prompt)
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            import openvino_genai

            get_generation_config = getattr(
                self._pipeline, "get_generation_config", None
            )
            config = (
                get_generation_config()
                if callable(get_generation_config)
                else openvino_genai.GenerationConfig()
            )
            if not hasattr(config, "apply_chat_template"):
                raise RAGError(
                    "qwen_npu_chat_template_control_unavailable",
                    "This OpenVINO GenAI version does not expose "
                    "`GenerationConfig.apply_chat_template`; the NPU prompt "
                    "cannot safely disable the native wrapper that produced "
                    "empty output.",
                    503,
                )
            config.apply_chat_template = False
            config.max_new_tokens = (
                settings.openvino_npu_max_new_tokens
                if max_tokens is None
                else max_tokens
            )
            config.do_sample = False
            config.repetition_penalty = 1.15
            config.stop_strings = {
                "<|im_end|>",
                "<|endoftext|>",
                "\n[Passage ",
                "**End Response**",
                "**End of Response**",
                "**End of Abstain Response**",
                "** End Response**",
                "\n**Final Answer:**",
                "\n**Source Check:**",
            }
            config.include_stop_str_in_output = False
            prompt_tokens = len(
                self._tokenizer.encode(prompt, add_special_tokens=False)
            )
            logger.info(
                "[NPU] Generation started "
                "(input_shape=list[1], prompt_tokens=%d, prompt_budget=%d, "
                "max_prompt_len=%d, min_response_len=%d, max_new_tokens=%d, "
                "sampling=%s, native_chat_template=disabled, prompt_sha256=%s, model_fingerprint=%s)",
                prompt_tokens,
                _prompt_token_budget(),
                settings.openvino_npu_max_prompt_len,
                settings.openvino_npu_max_new_tokens,
                config.max_new_tokens,
                config.do_sample,
                prompt_sha256,
                self._model_fingerprint,
            )
            generation_config_snapshot = _generation_config_snapshot(config)
            generation_started_at = perf_counter()
            with self._generation_lock:
                result = self._pipeline.generate([prompt], config)
            elapsed = perf_counter() - generation_started_at
            texts = getattr(result, "texts", None)
            if isinstance(texts, str):
                returned_text_count = 1
                raw_answer = texts
            elif texts is not None:
                try:
                    returned_text_count = len(texts)
                    raw_answer = texts[0] if returned_text_count else ""
                except (TypeError, IndexError, KeyError) as exc:
                    raise RuntimeError(
                        "OpenVINO returned an unsupported `texts` result shape."
                    ) from exc
            elif isinstance(result, str):
                returned_text_count = 1
                raw_answer = result
            else:
                returned_text_count = 0
                raw_answer = ""
            if not isinstance(raw_answer, str):
                raise RuntimeError(
                    "OpenVINO returned a non-string first generated text "
                    f"({type(raw_answer).__name__})."
                )
            raw_answer = raw_answer.strip()
            generated_tokens: int | None = None
            perf_metrics = getattr(result, "perf_metrics", None)
            get_generated_tokens = getattr(
                perf_metrics, "get_num_generated_tokens", None
            )
            if callable(get_generated_tokens):
                try:
                    metric_value = get_generated_tokens()
                    if isinstance(metric_value, int):
                        generated_tokens = metric_value
                    else:
                        logger.warning(
                            "[NPU] Generated-token metric had unexpected type %s",
                            type(metric_value).__name__,
                        )
                except Exception:
                    logger.warning(
                        "[NPU] Could not read generated-token metric",
                        exc_info=True,
                    )

            answer = _clean_model_output(raw_answer)
            logger.info(
                "[NPU] Generation finished "
                "(elapsed=%.1fs, result_type=%s, returned_text_count=%d, "
                "generated_tokens=%s, raw_output_chars=%d, visible_output_chars=%d)",
                elapsed,
                type(result).__name__,
                returned_text_count,
                generated_tokens,
                len(raw_answer),
                len(answer),
            )
        except RAGError:
            self.last_diagnostics["status"] = "error"
            raise
        except Exception as exc:
            logger.exception("OpenVINO NPU generation failed")
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": {"device": self.device},
            }
            raise RAGError(
                "qwen_npu_generation_failed",
                f"Astro Qwen NPU generation failed: {exc}",
                502,
            ) from exc

        if not answer:
            logger.warning(
                "[NPU] Returned no visible output "
                "(prompt_tokens=%d, max_prompt_len=%d, min_response_len=%d, "
                "max_new_tokens=%d, sampling=%s, result_type=%s, "
                "returned_text_count=%d, generated_tokens=%s, raw_output_chars=%d)",
                prompt_tokens,
                settings.openvino_npu_max_prompt_len,
                settings.openvino_npu_max_new_tokens,
                config.max_new_tokens,
                config.do_sample,
                type(result).__name__,
                returned_text_count,
                generated_tokens,
                len(raw_answer),
            )
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": "no_visible_output",
                "usage": {
                    "device": self.device,
                    "prompt_tokens": prompt_tokens,
                    "raw_output_chars": len(raw_answer),
                    "result_type": type(result).__name__,
                    "returned_text_count": returned_text_count,
                    "generated_tokens": generated_tokens,
                },
            }
            raise RAGError(
                "qwen_npu_empty_output",
                "Astro Qwen returned no visible answer on the NPU. Check the "
                "[NPU] server log entries for result type, returned text count, "
                "and generated-token metrics before changing the exported model.",
                502,
            )

        hit_token_limit = (
            generated_tokens is not None
            and generated_tokens >= config.max_new_tokens
        )
        if hit_token_limit:
            logger.warning(
                "[NPU] Output reached max_new_tokens; answer may be truncated "
                "(generated_tokens=%d, max_new_tokens=%d)",
                generated_tokens,
                config.max_new_tokens,
            )
        self.last_diagnostics = {
            "status": "incomplete" if hit_token_limit else "completed",
            "incomplete_reason": "max_new_tokens" if hit_token_limit else None,
            "usage": {
                "device": self.device,
                "runtime": "OpenVINO GenAI",
                "max_prompt_len": settings.openvino_npu_max_prompt_len,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": config.max_new_tokens,
                "generated_tokens": generated_tokens,
                "returned_text_count": returned_text_count,
                "prompt_sha256": prompt_sha256,
                "model_fingerprint": self._model_fingerprint,
                "do_sample": bool(config.do_sample),
                "repetition_penalty": float(config.repetition_penalty),
                "streaming_callback_used": False,
                "generation_config": generation_config_snapshot,
            },
        }
        return answer

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Yield visible NPU output while generation runs on a worker thread."""
        import queue
        import threading

        self._ensure_loaded()
        try:
            prompt = self._fit_prompt_to_npu_budget(system_prompt, user_prompt)
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            import openvino_genai

            get_generation_config = getattr(
                self._pipeline, "get_generation_config", None
            )
            config = (
                get_generation_config()
                if callable(get_generation_config)
                else openvino_genai.GenerationConfig()
            )
            if not hasattr(config, "apply_chat_template"):
                raise RAGError(
                    "qwen_npu_chat_template_control_unavailable",
                    "This OpenVINO GenAI version does not expose "
                    "`GenerationConfig.apply_chat_template`; native prompt "
                    "wrapping cannot be safely disabled.",
                    503,
                )
            config.apply_chat_template = False
            config.max_new_tokens = (
                settings.openvino_npu_max_new_tokens
                if max_tokens is None
                else max_tokens
            )
            config.do_sample = False
            config.repetition_penalty = 1.15
            config.stop_strings = {
                "<|im_end|>",
                "<|endoftext|>",
                "\n[Passage ",
                "**End Response**",
                "**End of Response**",
                "**End of Abstain Response**",
                "** End Response**",
                "\n**Final Answer:**",
                "\n**Source Check:**",
            }
            config.include_stop_str_in_output = False
            prompt_tokens = len(
                self._tokenizer.encode(prompt, add_special_tokens=False)
            )
            generation_config_snapshot = _generation_config_snapshot(config)
        except RAGError:
            self.last_diagnostics["status"] = "error"
            raise
        except Exception as exc:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": {"device": self.device},
            }
            raise RAGError(
                "qwen_npu_generation_failed",
                f"Could not prepare NPU streaming generation: {exc}",
                502,
            ) from exc

        token_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        callback_chunks: list[str] = []
        streaming_status = getattr(openvino_genai, "StreamingStatus", None)

        def should_stop_early(text: str) -> bool:
            answer_headers = re.findall(
                r"\*\*(?:Answer|Final Answer):\*\*",
                text,
                flags=re.IGNORECASE,
            )
            _, blocked_section_started = _stream_visible_text(text)
            return len(answer_headers) > 1 or blocked_section_started

        def streamer(subword: str):
            callback_chunks.append(subword)
            token_queue.put(("token", subword))
            if should_stop_early("".join(callback_chunks)):
                return getattr(streaming_status, "STOP", True)
            return getattr(streaming_status, "RUNNING", False)

        def worker() -> None:
            started_at = perf_counter()
            try:
                with self._generation_lock:
                    try:
                        result = self._pipeline.generate([prompt], config, streamer)
                    except TypeError:
                        if callback_chunks:
                            raise
                        try:
                            result = self._pipeline.generate(
                                prompt, config, streamer
                            )
                        except TypeError:
                            if callback_chunks:
                                raise
                            try:
                                result = self._pipeline.generate(
                                    [prompt], generation_config=config, streamer=streamer
                                )
                            except TypeError:
                                if callback_chunks:
                                    raise
                                logger.warning(
                                    "[NPU] OpenVINO pipeline does not accept the stream callback; "
                                    "falling back to non-streaming generation. Upgrade/align openvino-genai."
                                )
                                result = self._pipeline.generate([prompt], config)
                                texts = getattr(result, "texts", None)
                                if isinstance(texts, str):
                                    fallback_text = texts
                                elif texts is not None:
                                    fallback_text = (
                                        str(texts[0]) if len(texts) else ""
                                    )
                                else:
                                    fallback_text = (
                                        result if isinstance(result, str) else ""
                                    )
                                if fallback_text:
                                    token_queue.put(("token", fallback_text))
                token_queue.put(("result", (result, perf_counter() - started_at)))
            except Exception as exc:
                token_queue.put(("error", exc))

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        raw_chunks: list[str] = []
        emitted_visible = ""
        result = None
        elapsed = 0.0

        while True:
            kind, value = token_queue.get()
            if kind == "token":
                raw_chunks.append(value)
                current_raw = "".join(raw_chunks)
                if (
                    re.search(r"<think\b", current_raw, flags=re.IGNORECASE)
                    and not re.search(
                        r"</think\s*>", current_raw, flags=re.IGNORECASE
                    )
                ) or (
                    re.search(r"<tool_call\b", current_raw, flags=re.IGNORECASE)
                    and not re.search(
                        r"</tool_call\s*>", current_raw, flags=re.IGNORECASE
                    )
                ):
                    continue
                safe_raw, _ = _stream_visible_text(current_raw)
                cleaned = _clean_model_output(safe_raw)
                if len(cleaned) > len(emitted_visible):
                    delta = cleaned[len(emitted_visible):]
                    emitted_visible = cleaned
                    yield delta
            elif kind == "result":
                result, elapsed = value
                break
            else:
                self.last_diagnostics = {
                    "status": "error",
                    "incomplete_reason": None,
                    "usage": {"device": self.device},
                }
                if isinstance(value, RAGError):
                    raise value
                raise RAGError(
                    "qwen_npu_generation_failed",
                    f"Astro Qwen NPU generation failed: {value}",
                    502,
                ) from value

        worker_thread.join()
        raw_answer = "".join(raw_chunks).strip()
        if not raw_answer and result is not None:
            texts = getattr(result, "texts", None)
            if isinstance(texts, str):
                raw_answer = texts.strip()
            elif texts is not None:
                try:
                    raw_answer = str(texts[0]).strip() if len(texts) else ""
                except (TypeError, IndexError, KeyError) as exc:
                    raise RAGError(
                        "qwen_npu_generation_failed",
                        "OpenVINO returned an unsupported `texts` result shape.",
                        502,
                    ) from exc
            elif isinstance(result, str):
                raw_answer = result.strip()

        generated_tokens = None
        perf_metrics = getattr(result, "perf_metrics", None)
        get_generated_tokens = getattr(
            perf_metrics, "get_num_generated_tokens", None
        )
        if callable(get_generated_tokens):
            try:
                metric_value = get_generated_tokens()
                if isinstance(metric_value, int):
                    generated_tokens = metric_value
                else:
                    logger.warning(
                        "[NPU] Stream generated-token metric had unexpected type %s",
                        type(metric_value).__name__,
                    )
            except Exception:
                logger.warning(
                    "[NPU] Could not read streamed generated-token metric",
                    exc_info=True,
                )

        safe_raw, _ = _stream_visible_text(raw_answer, final=True)
        answer = _clean_model_output(safe_raw)
        if len(answer) > len(emitted_visible):
            yield answer[len(emitted_visible):]
        if not answer:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": "no_visible_output",
                "usage": {
                    "device": self.device,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                },
            }
            raise RAGError(
                "qwen_npu_empty_output",
                "Astro Qwen returned no visible answer on the NPU.",
                502,
            )

        hit_token_limit = (
            generated_tokens is not None
            and generated_tokens >= config.max_new_tokens
        )
        self.last_diagnostics = {
            "status": "incomplete" if hit_token_limit else "completed",
            "incomplete_reason": "max_new_tokens" if hit_token_limit else None,
            "usage": {
                "device": self.device,
                "runtime": "OpenVINO GenAI",
                "max_prompt_len": settings.openvino_npu_max_prompt_len,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": config.max_new_tokens,
                "generated_tokens": generated_tokens,
                "returned_text_count": 1,
                "elapsed_seconds": round(elapsed, 2),
                "prompt_sha256": prompt_sha256,
                "model_fingerprint": self._model_fingerprint,
                "do_sample": bool(config.do_sample),
                "repetition_penalty": float(config.repetition_penalty),
                "streaming_callback_used": bool(callback_chunks),
                "generation_config": generation_config_snapshot,
            },
        }


_client: OpenVINONPUClient | None = None
_client_lock = Lock()


def get_openvino_npu_client() -> OpenVINONPUClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenVINONPUClient()
    return _client
