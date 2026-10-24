"""Lazy local inference for the Astro Qwen LoRA adapter."""
from __future__ import annotations

import re
from pathlib import Path
from threading import Lock
from typing import Any

from app.config.settings import settings
from app.utils.errors import RAGError

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
ADAPTER_MODEL = "XeroxFlynn/astro-rag-lora-v2"


def _load_adapter_tokenizer(
    tokenizer_loader: Any,
    adapter_dir: Path,
    token: str | None,
    base_model: str = BASE_MODEL,
) -> Any:
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
    )
    tokenizer_source = (
        adapter_dir
        if any((adapter_dir / name).is_file() for name in tokenizer_files)
        else base_model
    )
    tokenizer = tokenizer_loader.from_pretrained(
        str(tokenizer_source),
        token=token,
        trust_remote_code=True,
    )
    chat_template_path = adapter_dir / "chat_template.jinja"
    if chat_template_path.is_file():
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    if not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise RuntimeError(
            "A string chat template was not found in the adapter or base "
            f"tokenizer for {ADAPTER_MODEL}."
        )
    return tokenizer


def _dedupe_repeated_sections(text: str) -> str:
    """
    Truncate multi-turn generation loops where the model starts a second
    `**Answer:**`, `**Final Answer:**`, `**Conclusion:**`, or a second
    `**Reasoning:**` block after completing the first response.
    """
    # Normalize jammed "**Answer:** **Reasoning:**" or ": **Answer:**" opening
    text = re.sub(r"^\s*:\s*(?=\*\*Answer:\*\*)", "", text)
    text = re.sub(
        r"^\s*\*\*Answer:\*\*\s*\*\*(?:Reasoning|Response):\*\*\s*",
        "**Answer:**\n",
        text,
    )

    # Cut at a second "**Answer:**" occurrence (after index 20)
    second_ans = re.search(
        r"(?:\n|\s{2,})(?:\*\*Answer:\*\*|\*\*Final Answer:\*\*|Answer:\s*\n)",
        text[20:],
        flags=re.IGNORECASE,
    )
    if second_ans:
        text = text[: 20 + second_ans.start()].rstrip()

    # Cut at a second "**Reasoning:**" occurrence
    reas_matches = list(
        re.finditer(r"\*\*Reasoning:\*\*", text, flags=re.IGNORECASE)
    )
    if len(reas_matches) > 1:
        text = text[: reas_matches[1].start()].rstrip()

    return text


def _clean_model_output(text: str) -> str:
    text = re.sub(
        r"<tool_call(?:\s[^>]*)?>.*?</tool_call\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"</?tool_call(?:\s[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<thought_process(?:\s[^>]*)?>.*?(?:</thought_process>|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"<think(?:\s[^>]*)?>.*?(?:</think>|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"^\s*(?:Answer:|ANSWER)\s*", "", text)
    text = re.sub(r"^\s*\*\*Response:\*\*", "**Answer:**", text)
    # Normalize corrupted book title hallucinations seen in NPU int4 runs
    text = re.sub(
        r"Brihat\s+Paras(?:h)?ara\s+(?:Bhashya|Hora\s+Sastry|Hora\s+Samhasta|Samhita)",
        "Brihat Parasara Hora Sastra",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Brihad\s+Parashar\s+Hora\s+Sastra|Brihadaranyananda",
        "Brihat Parasara Hora Sastra",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\(\*\*BP\*\*\)\*\*", "(**BPHS**)", text)
    text = re.sub(r"\(\*\*BPBH\*\*", "(**BPHS**", text)
    # Remove hallucinated outer planet slash-insertions (e.g. "Mercury/Pluto")
    text = re.sub(r"/(?:Pluto|Uranus|Neptune)\b", "", text, flags=re.IGNORECASE)
    # Replace hallucinated pseudo-Sanskrit yoga names on BPHS Ch. 41 / Ch. 18 rules
    text = re.sub(r"\b(?:Dwipaka|Dvivida|Drishti)\s+Yoga\b", "classical combination", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Chart (?:Binding|Verification Note):[^\]]*\]", "", text)
    text = re.split(r"(?:^|\n)\[Passage\s+\d+\s*[—-]", text, maxsplit=1)[0]
    text = re.split(r"\nNote:\s+Per the prompt", text, maxsplit=1)[0]
    text = re.split(
        r"\*\*\s*End\s+(?:of\s+)?(?:Abstain\s+)?Response\s*\*\*"
        r"|\n\*\*Final Answer:\*\*|\n\*\*Source Check:\*\*"
        r"|\s+\*\*Final Answer:\*\*|\s+\*\*Source Check:\*\*",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = _dedupe_repeated_sections(text)
    text = re.sub(
        r"\s*(?:\()?\*?(?:According to|Per)\s+Rule\s+\d+[^\n]*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return text.strip()


def verify_and_repair_chart_claims(
    text: str,
    chart: dict[str, Any] | None = None,
    passages: list[dict[str, Any]] | None = None,
) -> str:
    """
    Deterministic post-generation chart-grounding guard:
    1. Cross-checks any `<nth> lord <Planet> in/occupies the <mth> house` or
       `<Planet> occupies the <mth> house (<Sign>/H<m>)` claim against the
       immutable `chart` JSON and repairs contradicted house/sign assignments.
    2. Cross-checks any `<Planet> rules both the <nth> and <mth> houses` claim
       against the planet's actual house lordships in `chart`.
    3. Strips any invented `<Name> Yoga` title that does not appear in the
       retrieved passages.
    """
    if not text:
        return text

    # 1. Strip any invented "<Name> Yoga" not present in retrieved passages
    passage_blob = " ".join((p.get("text") or "") for p in (passages or [])).lower()
    allowed_yogas = {"raja yoga", "raj yoga", "dhana yoga", "viparita raja yoga", "parivartana yoga", "gajakesari yoga", "budhaditya yoga"}

    def _check_yoga(match: re.Match) -> str:
        full_yoga = match.group(0)
        if full_yoga.lower() in allowed_yogas or full_yoga.lower() in passage_blob:
            return full_yoga
        return "classical combination"

    text = re.sub(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+Yoga\b",
        _check_yoga,
        text,
    )

    if not chart or not (chart.get("planets") and chart.get("houses")):
        return text

    # Build immutable lookup tables from chart JSON
    planet_info: dict[str, tuple[int, str]] = {}
    for p in chart.get("planets", []):
        body = p.get("body")
        h = (p.get("placement") or {}).get("house")
        s = (p.get("position") or {}).get("zodiac_sign")
        if body and h is not None and s:
            planet_info[body] = (int(h), str(s))

    house_info: dict[int, tuple[str, str]] = {}
    lord_houses: dict[str, list[ tuple[int, str] ]] = {}
    for h_obj in chart.get("houses", []):
        h_num = h_obj.get("number")
        s = h_obj.get("sign")
        lord = h_obj.get("lord")
        if h_num is not None and s and lord:
            house_info[int(h_num)] = (str(s), str(lord))
            lord_houses.setdefault(str(lord), []).append((int(h_num), str(s)))

    ordinals = {
        1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th",
        7: "7th", 8: "8th", 9: "9th", 10: "10th", 11: "11th", 12: "12th",
    }

    # Repair false "<Planet> in/occupies the <N>th house (<Sign>/H<N>)" claims
    for planet, (true_h, true_s) in planet_info.items():
        true_ord = ordinals.get(true_h, f"{true_h}th")
        pattern = re.compile(
            rf"\b({planet}\b[^.\n]{{0,30}}?\b(?:in|occupies|sits in|placed in)\s+the\s+)"
            rf"(\d{{1,2}})(?:st|nd|rd|th)\s+house"
            rf"(?:\s*\(([^)]*)\))?",
            flags=re.IGNORECASE,
        )

        def _fix_placement(m: re.Match, p=planet, th=true_h, ts=true_s, tord=true_ord) -> str:
            claimed_h = int(m.group(2))
            if claimed_h == th:
                return m.group(0)
            return f"{m.group(1)}{tord} house ({ts}/H{th})"

        text = pattern.sub(_fix_placement, text)

        # Also repair "<Planet> rules both the Xth (...) and Yth houses (...)"
        ruled_list = lord_houses.get(planet, [])
        if len(ruled_list) == 2:
            (h1, s1), (h2, s2) = sorted(ruled_list)
            o1, o2 = ordinals[h1], ordinals[h2]
            dual_rule_pat = re.compile(
                rf"\b{planet}\s+rules\s+both\s+the\s+\d{{1,2}}(?:st|nd|rd|th)\s*(?:\([^)]*\)\s*)?"
                rf"and\s+\d{{1,2}}(?:st|nd|rd|th)\s+houses(?:\s*\([^)]*\))?",
                flags=re.IGNORECASE,
            )
            text = dual_rule_pat.sub(
                f"{planet} rules the {o1} ({s1}) and {o2} ({s2}) houses",
                text,
            )

    return text


class QwenLocalClient:
    """Load Qwen3 + the Astro LoRA adapter once, on the first selected request."""

    mode = "transformers"
    model = ADAPTER_MODEL

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = Lock()
        self._generation_lock = Lock()
        self.last_diagnostics: dict[str, Any] = {
            "status": "not_called",
            "incomplete_reason": None,
            "usage": None,
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                import bitsandbytes  # noqa: F401 - required by the base's NF4 weights
                from huggingface_hub import get_token, snapshot_download
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_dependencies_missing",
                    "Astro Qwen needs PyTorch, Transformers, PEFT, and Accelerate. "
                    "Install the optional dependencies listed in "
                    "requirements-local-model.txt, then restart Astro RAG.",
                    503,
                ) from exc

            try:
                hub_token = get_token()
                adapter_candidate = Path(ADAPTER_MODEL).expanduser()
                if adapter_candidate.is_dir():
                    adapter_path = adapter_candidate.resolve()
                elif not hub_token:
                    raise RAGError(
                        "qwen_hf_auth_required",
                        "The Astro Qwen adapter is in a private Hugging Face repo. "
                        "Run `hf auth login` as the same Windows user that runs Astro RAG, "
                        f"using an account with read access to {ADAPTER_MODEL}.",
                        503,
                    )
                else:
                    adapter_path = Path(
                        snapshot_download(ADAPTER_MODEL, token=hub_token)
                    )
                tokenizer = _load_adapter_tokenizer(
                    AutoTokenizer,
                    adapter_path,
                    hub_token,
                )
                device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    torch_dtype="auto",
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(
                    base,
                    adapter_path,
                    low_cpu_mem_usage=True,
                )
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
            except RAGError:
                self.last_diagnostics["status"] = "error"
                raise
            except ImportError as exc:
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_dependencies_missing",
                    "A local Qwen dependency is missing. Install the optional "
                    "dependencies listed in requirements-local-model.txt, "
                    "then restart Astro RAG.",
                    503,
                ) from exc
            except Exception as exc:
                self.last_diagnostics["status"] = "error"
                raise RAGError(
                    "qwen_model_load_failed",
                    "Could not load Astro Qwen locally. Check network access to "
                    "Hugging Face, available system memory, and the model cache. "
                    f"Details: {exc}",
                    503,
                ) from exc

    def unload(self) -> None:
        with self._load_lock:
            if self._model is not None or self._tokenizer is not None:
                import gc

                self._model = None
                self._tokenizer = None
                gc.collect()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        self._ensure_loaded()
        try:
            import torch

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_device = self._model.get_input_embeddings().weight.device
            inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            eos_ids = []
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                eos_ids.append(eos_token_id)
            convert_tokens_to_ids = getattr(
                self._tokenizer, "convert_tokens_to_ids", None
            )
            if callable(convert_tokens_to_ids):
                im_end_id = convert_tokens_to_ids("<|im_end|>")
                unk_token_id = getattr(self._tokenizer, "unk_token_id", None)
                if (
                    isinstance(im_end_id, int)
                    and im_end_id != unk_token_id
                    and im_end_id not in eos_ids
                ):
                    eos_ids.append(im_end_id)
            generation_options: dict[str, Any] = {
                "max_new_tokens": min(
                    max_tokens or settings.generation_max_tokens, 700
                ),
                "do_sample": False,
                "repetition_penalty": 1.15,
            }
            if eos_ids:
                generation_options["eos_token_id"] = eos_ids
            if isinstance(eos_token_id, int):
                generation_options["pad_token_id"] = eos_token_id
            with self._generation_lock, torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    **generation_options,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            answer = _clean_model_output(
                self._tokenizer.decode(
                    output[0][prompt_length:], skip_special_tokens=True
                )
            )
        except RAGError:
            raise
        except Exception as exc:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": None,
            }
            raise RAGError(
                "qwen_generation_failed",
                f"Astro Qwen local generation failed: {exc}",
                502,
            ) from exc

        if not answer:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": "no_visible_output",
                "usage": None,
            }
            raise RAGError(
                "qwen_empty_output",
                "Astro Qwen returned no visible answer. Try a shorter question or check available memory.",
                502,
            )

        self.last_diagnostics = {
            "status": "completed",
            "incomplete_reason": None,
            "usage": None,
        }
        return answer


    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ):
        """Stream tokens live from the local Qwen LoRA model using TextIteratorStreamer."""
        import threading

        self._ensure_loaded()
        try:
            import torch
            from transformers import TextIteratorStreamer
        except ImportError:
            yield self.generate(system_prompt, user_prompt, max_tokens=max_tokens, model=model)
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_device = self._model.get_input_embeddings().weight.device
        inputs = {
            key: value.to(input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        eos_ids = []
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_ids.append(eos_token_id)
        convert_tokens_to_ids = getattr(
            self._tokenizer, "convert_tokens_to_ids", None
        )
        if callable(convert_tokens_to_ids):
            im_end_id = convert_tokens_to_ids("<|im_end|>")
            unk_token_id = getattr(self._tokenizer, "unk_token_id", None)
            if (
                isinstance(im_end_id, int)
                and im_end_id != unk_token_id
                and im_end_id not in eos_ids
            ):
                eos_ids.append(im_end_id)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_options: dict[str, Any] = {
            "max_new_tokens": min(
                max_tokens or settings.generation_max_tokens, 700
            ),
            "do_sample": False,
            "repetition_penalty": 1.15,
            "streamer": streamer,
        }
        if eos_ids:
            generation_options["eos_token_id"] = eos_ids
        if isinstance(eos_token_id, int):
            generation_options["pad_token_id"] = eos_token_id

        errors: list[Exception] = []

        def _worker() -> None:
            try:
                with self._generation_lock, torch.inference_mode():
                    self._model.generate(**inputs, **generation_options)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        raw_chunks: list[str] = []
        emitted_visible = ""
        for chunk in streamer:
            raw_chunks.append(chunk)
            current_raw = "".join(raw_chunks)
            if "<think" in current_raw and "</think>" not in current_raw:
                continue
            cleaned_so_far = _clean_model_output(current_raw)
            if len(cleaned_so_far) > len(emitted_visible):
                delta = cleaned_so_far[len(emitted_visible):]
                emitted_visible = cleaned_so_far
                yield delta

        thread.join(timeout=1.0)
        if errors:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": None,
            }
            raise RAGError(
                "qwen_generation_failed",
                f"Astro Qwen local generation failed: {errors[0]}",
                502,
            ) from errors[0]

        final_answer = _clean_model_output("".join(raw_chunks))
        if len(final_answer) > len(emitted_visible):
            yield final_answer[len(emitted_visible):]
        if not final_answer:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": "no_visible_output",
                "usage": None,
            }
            raise RAGError(
                "qwen_empty_output",
                "Astro Qwen returned no visible answer.",
                502,
            )
        self.last_diagnostics = {
            "status": "completed",
            "incomplete_reason": None,
            "usage": None,
        }

_client: QwenLocalClient | None = None
_client_lock = Lock()


def get_qwen_client() -> QwenLocalClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = QwenLocalClient()
    return _client