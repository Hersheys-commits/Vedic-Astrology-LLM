"""Lazy OpenVINO GenAI inference on the Intel NPU."""
from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from app.config.settings import settings
from app.generation.qwen_client import ADAPTER_MODEL, _clean_model_output
from app.utils.errors import RAGError

logger = logging.getLogger(__name__)


def _prompt_token_budget() -> int:
    return max(1, settings.openvino_npu_max_prompt_len - 128)


NPU_SYSTEM_PROMPT = """\
You explain Vedic astrology (Jyotisha) using only the supplied chart facts and retrieved passages.
1. Treat every line in CHART SNAPSHOT and RELEVANT CHART FACTS FOR THIS QUESTION as ground truth. Never invent, swap, or alter a planet's house, sign, or lordship.
2. Verify every condition in a retrieved passage against the chart facts before applying it. If a user's question assumes a placement that contradicts the chart facts, state the actual placement from the chart facts first.
3. Cite each source-based claim with its 1-based passage number and book (e.g. "Passage 1 — Brihat Parasara Hora Sastra").
4. Make reasonable traditional synthesis when a passage covers the relevant house, lordship, or planet placement; translate classic themes into cautious practical tendencies ("may indicate", "traditionally associated with"). Only abstain when the question asks for a non-astrological detail (like an exact name initial) or when no passage touches the queried houses/lords.
5. Format your response with a direct "**Answer:**" paragraph followed by a concise "**Reasoning:**" paragraph, then stop immediately after one complete response."""


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
                if not line:
                    continue
                if line.startswith("[Passage "):
                    cleaned_lines.append(line)
                    continue
                alpha_count = sum(character.isalpha() for character in line)
                if len(line) > 15 and alpha_count / len(line) < 0.35:
                    continue
                if len(line) < 18 and not re.search(
                    r"\b(if|when|the|in|of|and|to|is|are|will|should|house|lord|"
                    r"sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu|"
                    r"ascendant|lagna|effects|chapter|sloka|rule)\b",
                    line,
                    re.IGNORECASE,
                ):
                    continue
                cleaned_lines.append(line)
            return "\n".join(cleaned_lines)

        has_passage_section = passages_marker in user_prompt and question_marker in user_prompt
        if has_passage_section:
            prefix, rest = user_prompt.split(passages_marker, 1)
            prefix = re.sub(
                r"\n(?:Full House Map|Houses):[^\n]*",
                "",
                prefix,
                flags=re.IGNORECASE,
            )
            passages_block, question_suffix = rest.rsplit(question_marker, 1)
            passages_block = clean_passages(passages_block)
            passage_entries = [
                entry.strip()
                for entry in re.split(r"(?=\[Passage \d+[^\]\n]*\])", passages_block)
                if entry.strip()
            ]
            # Keep the first sentence of Notes when the sloka body before Notes is very short,
            # otherwise trim verbose editorial Notes so core sloka translations fit the budget.
            trimmed_entries = []
            for entry in passage_entries:
                verif_matches = re.findall(r"(\[Chart (?:Binding|Verification Note):[^\]]*\])", entry)
                verif_suffix = ("\n" + "\n".join(verif_matches)) if verif_matches else ""
                entry_no_verif = re.sub(r"\n?\[Chart (?:Binding|Verification Note):[^\]]*\]", "", entry)
                parts = re.split(r"\bNotes\s*:", entry_no_verif, maxsplit=1)
                main_part = parts[0].rstrip()
                if len(parts) > 1:
                    note_preview = parts[1].strip()[:220].rstrip()
                    trimmed_entries.append(f"{main_part}\nNote: {note_preview}{verif_suffix}")
                else:
                    trimmed_entries.append(f"{main_part}{verif_suffix}")
            passage_entries = trimmed_entries
        else:
            prefix = user_prompt
            question_suffix = ""
            passage_entries = []

        def render(content: str) -> str:
            # Format with Qwen's chat template (thinking disabled) when a real
            # HuggingFace tokenizer with chat_template is loaded, while keeping
            # plain `system\n\nuser` fallback for lightweight unit test doubles.
            chat_template = getattr(self._tokenizer, "chat_template", None)
            apply_fn = getattr(self._tokenizer, "apply_chat_template", None)
            if isinstance(chat_template, str) and callable(apply_fn):
                try:
                    return apply_fn(
                        [
                            {"role": "system", "content": effective_system},
                            {"role": "user", "content": content},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    rendered = apply_fn(
                        [
                            {"role": "system", "content": effective_system},
                            {"role": "user", "content": content},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str):
                        if not rendered.endswith("</think>\n\n"):
                            rendered += "<think>\n\n</think>\n\n"
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

        headers_and_bodies: list[tuple[str, list[int]]] = []
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
            headers_and_bodies.append(
                (header, encode(body, add_special_tokens=False))
            )

        headers_only = "\n\n".join(header for header, _ in headers_and_bodies)
        headers_prompt = render(
            f"{prefix}{passages_marker}{headers_only}"
            f"{question_marker}{question_suffix}"
        )
        if token_count(headers_prompt) > max_tokens:
            raise RAGError(
                "qwen_npu_prompt_too_long",
                "The chart facts, question, and passage citation headers exceed "
                "the configured NPU token budget. Shorten the question or reduce "
                "the chart context.",
                422,
            )

        def render_passages(body_token_limit: int) -> str:
            entries = []
            for header, body_ids in headers_and_bodies:
                body = self._tokenizer.decode(
                    body_ids[:body_token_limit], skip_special_tokens=True
                ).rstrip()
                if len(body_ids) > body_token_limit and body:
                    body += " ..."
                entries.append(f"{header}\n{body}" if body else header)
            passage_text = "\n\n".join(entries)
            return render(
                f"{prefix}{passages_marker}{passage_text}"
                f"{question_marker}{question_suffix}"
            )

        low = 0
        high = max((len(body_ids) for _, body_ids in headers_and_bodies), default=0)
        best_prompt = headers_prompt
        while low <= high:
            per_passage_limit = (low + high) // 2
            candidate_prompt = render_passages(per_passage_limit)
            if token_count(candidate_prompt) <= max_tokens:
                best_prompt = candidate_prompt
                low = per_passage_limit + 1
            else:
                high = per_passage_limit - 1

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
                "sampling=%s, chat_template=disabled)",
                prompt_tokens,
                _prompt_token_budget(),
                settings.openvino_npu_max_prompt_len,
                settings.openvino_npu_max_new_tokens,
                config.max_new_tokens,
                config.do_sample,
            )
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
            },
        }
        return answer


    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ):
        """
        Stream tokens live from the Intel NPU using OpenVINO GenAI's streamer
        callback while preserving the exact prompt fitting, stop controls,
        output cleanup, and diagnostics of `generate()`.
        """
        import queue
        import threading

        self._ensure_loaded()
        prompt = self._fit_prompt_to_npu_budget(system_prompt, user_prompt)
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
            "[NPU] Streaming generation started "
            "(prompt_tokens=%d, prompt_budget=%d, max_new_tokens=%d)",
            prompt_tokens,
            _prompt_token_budget(),
            config.max_new_tokens,
        )

        token_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        raw_chunks: list[str] = []
        streaming_status = getattr(openvino_genai, "StreamingStatus", None)

        def _should_stop_early(accumulated: str) -> bool:
            # Stop early if the model starts a second repeated Answer or Reasoning block
            if len(accumulated) > 40:
                if re.search(
                    r"(?:\n|\s{2,})(?:\*\*Answer:\*\*|\*\*Final Answer:\*\*)",
                    accumulated[25:],
                    flags=re.IGNORECASE,
                ):
                    return True
                if len(re.findall(r"\*\*Reasoning:\*\*", accumulated, flags=re.IGNORECASE)) > 1:
                    return True
            return False

        def _streamer(subword: str):
            raw_chunks.append(subword)
            token_queue.put(("token", subword))
            accumulated = "".join(raw_chunks)
            if _should_stop_early(accumulated):
                return streaming_status.STOP if streaming_status is not None else True
            return streaming_status.RUNNING if streaming_status is not None else False

        def _worker() -> None:
            started_at = perf_counter()
            try:
                with self._generation_lock:
                    try:
                        # OpenVINO GenAI streamer works with single string prompt or list[1]
                        result = self._pipeline.generate(prompt, config, _streamer)
                    except TypeError:
                        try:
                            result = self._pipeline.generate([prompt], config, _streamer)
                        except TypeError:
                            # Fallback for test doubles that do not accept a streamer arg
                            result = self._pipeline.generate([prompt], config)
                            texts = getattr(result, "texts", None)
                            fallback_text = (
                                texts[0]
                                if isinstance(texts, list) and texts
                                else (texts if isinstance(texts, str) else "")
                            )
                            if fallback_text:
                                raw_chunks.append(fallback_text)
                                token_queue.put(("token", fallback_text))
                elapsed = perf_counter() - started_at
                token_queue.put(("result", (result, elapsed)))
            except Exception as exc:
                token_queue.put(("error", exc))

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        emitted_visible = ""
        result_obj = None
        elapsed = 0.0

        while True:
            kind, payload = token_queue.get()
            if kind == "token":
                current_raw = "".join(raw_chunks)
                # Suppress incomplete <think> or <tool_call> blocks while streaming
                if "<think" in current_raw and "</think>" not in current_raw:
                    continue
                if "<tool_call" in current_raw and "</tool_call>" not in current_raw:
                    continue
                cleaned_so_far = _clean_model_output(current_raw)
                if len(cleaned_so_far) > len(emitted_visible):
                    delta = cleaned_so_far[len(emitted_visible):]
                    emitted_visible = cleaned_so_far
                    yield delta
            elif kind == "result":
                result_obj, elapsed = payload
                break
            elif kind == "error":
                self.last_diagnostics = {
                    "status": "error",
                    "incomplete_reason": None,
                    "usage": {"device": self.device},
                }
                if isinstance(payload, RAGError):
                    raise payload
                raise RAGError(
                    "qwen_npu_generation_failed",
                    f"Astro Qwen NPU generation failed: {payload}",
                    502,
                ) from payload

        worker_thread.join(timeout=1.0)

        raw_answer = "".join(raw_chunks).strip()
        if not raw_answer and result_obj is not None:
            texts = getattr(result_obj, "texts", None)
            if isinstance(texts, str):
                raw_answer = texts.strip()
            elif isinstance(texts, list) and texts:
                raw_answer = str(texts[0]).strip()
            elif isinstance(result_obj, str):
                raw_answer = result_obj.strip()

        generated_tokens: int | None = None
        perf_metrics = getattr(result_obj, "perf_metrics", None)
        get_generated_tokens = getattr(
            perf_metrics, "get_num_generated_tokens", None
        )
        if callable(get_generated_tokens):
            try:
                metric_value = get_generated_tokens()
                if isinstance(metric_value, int):
                    generated_tokens = metric_value
            except Exception:
                pass

        final_answer = _clean_model_output(raw_answer)
        if len(final_answer) > len(emitted_visible):
            yield final_answer[len(emitted_visible):]

        if not final_answer:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": "no_visible_output",
                "usage": {
                    "device": self.device,
                    "prompt_tokens": prompt_tokens,
                    "raw_output_chars": len(raw_answer),
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