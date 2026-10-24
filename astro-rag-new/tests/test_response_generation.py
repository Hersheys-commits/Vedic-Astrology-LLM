from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest
from fastapi.testclient import TestClient

from app.api import ask as ask_api
from app.facts import intent
from app.generation import response_generator
from app.generation.llm_client import LLMClient
from app.generation.openvino_npu_client import (
    OpenVINONPUClient,
    _allocate_passage_token_limits,
)
from app.generation.qwen_client import (
    ADAPTER_MODEL,
    QwenLocalClient,
    _clean_model_output,
    _load_adapter_tokenizer,
    _repair_passage_citations,
    _remove_unsupported_divisional_claims,
    _stream_visible_text,
    verify_and_repair_chart_claims,
)
from app.main import app
from app.models.requests import AskRequest
from app.utils.errors import RAGError


class FakeGenerator:
    def __init__(self, model, mode):
        self.model = model
        self.mode = mode
        self.last_diagnostics = {
            "status": "completed",
            "incomplete_reason": None,
            "usage": None,
        }

    def unload(self):
        return None

    def generate(self, system_prompt, user_prompt):
        return f"answer from {self.model}"


def test_qwen_adapter_model_is_v2():
    assert ADAPTER_MODEL == "XeroxFlynn/astro-rag-lora-v2"


def test_response_generation_routes_only_to_the_selected_answer_provider(monkeypatch):
    openai = FakeGenerator("gpt-5-mini", "openai")
    qwen = FakeGenerator(ADAPTER_MODEL, "transformers")
    npu = FakeGenerator(ADAPTER_MODEL, "openvino_npu")
    monkeypatch.setattr(response_generator, "get_llm_client", lambda: openai)
    monkeypatch.setattr(response_generator, "get_qwen_client", lambda: qwen)
    monkeypatch.setattr(response_generator, "get_openvino_npu_client", lambda: npu)

    openai_answer, openai_provider = response_generator.generate_response(
        "gpt-5-mini", "system", "user"
    )
    qwen_answer, qwen_provider = response_generator.generate_response(
        "astro-qwen-lora", "system", "user"
    )
    npu_answer, npu_provider = response_generator.generate_response(
        "astro-qwen-npu", "system", "user"
    )

    assert (openai_answer, openai_provider) == ("answer from gpt-5-mini", openai)
    assert (qwen_answer, qwen_provider) == (
        f"answer from {ADAPTER_MODEL}", qwen,
    )
    assert (npu_answer, npu_provider) == (
        f"answer from {ADAPTER_MODEL}", npu,
    )


def test_qwen_failure_does_not_fall_back_to_openai(monkeypatch):
    def fail_qwen():
        raise RAGError("qwen_dependencies_missing", "Install local model dependencies.", 503)

    monkeypatch.setattr(response_generator, "get_qwen_client", fail_qwen)
    monkeypatch.setattr(
        response_generator,
        "get_llm_client",
        lambda: pytest.fail("Qwen failure must not select GPT automatically"),
    )

    with pytest.raises(RAGError, match="Install local model dependencies"):
        response_generator.generate_response("astro-qwen-lora", "system", "user")


def test_intent_extraction_remains_on_the_configured_gpt_model(monkeypatch):
    monkeypatch.setattr(intent.settings, "openai_model", "gpt-5-mini")
    monkeypatch.setattr(intent.settings, "intent_model", "")
    monkeypatch.setattr(intent, "build_intent_prompt", lambda *_: ("system", "user"))
    seen = {}

    class IntentGenerator:
        mode = "openai"

        def generate(self, system, user, max_tokens=None, model=None):
            seen["configured_model"] = model or intent.settings.openai_model
            return '{"facts": []}'

    assert intent.extract_facts_llm("Question", {}, IntentGenerator()) is None
    assert seen["configured_model"] == "gpt-5-mini"


def test_ask_uses_response_selection_and_logs_processing_stages(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger=ask_api.logger.name)
    captured = {}
    provider = FakeGenerator(ADAPTER_MODEL, "transformers")

    def resolve(question, chart, mode=None):
        captured["intent_mode"] = mode
        return [], "rules"

    def generate(model, system_prompt, user_prompt):
        captured["response_model"] = model
        return provider.generate(system_prompt, user_prompt), provider

    monkeypatch.setattr(ask_api, "resolve_facts", resolve)
    monkeypatch.setattr(ask_api, "get_store", lambda: object())
    monkeypatch.setattr(ask_api, "retrieve_for_facts", lambda **_: [])
    monkeypatch.setattr(ask_api, "generate_response", generate)
    monkeypatch.setattr(ask_api.cr, "chart_summary", lambda _: {})

    response = TestClient(app).post(
        "/ask",
        json={
            "chart_json": {},
            "question": "What does my chart suggest?",
            "intent_mode": "rules",
            "response_model": "astro-qwen-lora",
        },
    )

    assert response.status_code == 200
    assert "Ask request started" in caplog.text
    assert "chart ready; resolving question facts" in caplog.text
    assert "answer generation started" in caplog.text
    assert "Ask request completed" in caplog.text
    assert captured == {"intent_mode": "rules", "response_model": "astro-qwen-lora"}
    assert response.json()["model"] == ADAPTER_MODEL
    assert response.json()["generation_mode"] == "transformers"


@pytest.mark.parametrize(
    ("response_model", "requested_mode", "expected_mode"),
    [
        ("astro-qwen-lora", None, "rules"),
        ("astro-qwen-npu", None, "rules"),
        ("astro-qwen-npu", "auto", "auto"),
        ("gpt-5-mini", None, None),
    ],
)
def test_ask_uses_deterministic_intent_for_local_models_by_default(
    response_model, requested_mode, expected_mode, monkeypatch
):
    captured = {}
    provider = FakeGenerator(response_model, "mock")

    def resolve(question, chart, mode=None):
        captured["mode"] = mode
        return [], "rules"

    monkeypatch.setattr(ask_api, "resolve_facts", resolve)
    monkeypatch.setattr(ask_api, "get_store", lambda: object())
    monkeypatch.setattr(ask_api, "retrieve_for_facts", lambda **_: [])
    monkeypatch.setattr(ask_api, "build_prompt", lambda *_: ("system", "user"))
    monkeypatch.setattr(
        ask_api,
        "generate_response",
        lambda *_: ("answer", provider),
    )
    monkeypatch.setattr(ask_api.cr, "chart_summary", lambda _: {})
    request_data = {
        "chart_json": {},
        "question": "What does my chart suggest?",
        "response_model": response_model,
    }
    if requested_mode is not None:
        request_data["intent_mode"] = requested_mode

    response = TestClient(app).post("/ask", json=request_data)

    assert response.status_code == 200
    assert captured["mode"] == expected_mode


@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
def test_ask_endpoints_run_chart_claim_verifier(
    endpoint, monkeypatch
):
    chart = {
        "planets": [{"body": "Mercury", "placement": {"house": 4}}],
        "houses": [{"number": 4, "lord": "Mercury"}],
    }
    passages = [{
        "chunk_id": "source",
        "text": "Source passage.",
        "metadata": {},
        "score": 1.0,
        "match_type": "relation+semantic",
    }]
    calls = []
    provider = FakeGenerator("model", "mock")
    monkeypatch.setattr(
        ask_api,
        "_prepare_ask_context",
        lambda _: (chart, [], "rules", passages, "system", "user"),
    )
    monkeypatch.setattr(ask_api.cr, "chart_summary", lambda _: {})
    monkeypatch.setattr(
        ask_api,
        "verify_and_repair_chart_claims",
        lambda answer, actual_chart, actual_passages: (
            calls.append((actual_chart, actual_passages)) or "verified answer"
        ),
    )
    monkeypatch.setattr(
        ask_api,
        "generate_response",
        lambda *_: ("unverified answer", provider),
    )
    monkeypatch.setattr(
        ask_api,
        "stream_response",
        lambda *_: (iter(["unverified ", "answer"]), provider),
    )

    response = TestClient(app).post(
        endpoint,
        json={"chart_json": {}, "question": "Question"},
    )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0] == (chart, passages)
    if endpoint == "/ask":
        assert response.json()["answer"] == "verified answer"
        assert response.json()["generation_usage"]["retrieval_signature"] == [{
            "chunk_id": "source",
            "score": 1.0,
            "match_type": "relation+semantic",
        }]
    else:
        replace = next(
            frame for frame in response.text.split("\n\n")
            if frame.startswith("event: replace")
        )
        replacement = __import__("json").loads(replace.split("data: ", 1)[1])
        done = next(
            frame for frame in response.text.split("\n\n")
            if frame.startswith("event: done")
        )
        payload = __import__("json").loads(done.split("data: ", 1)[1])
        assert payload["answer"] == "verified answer"
        assert replacement["answer"] == "verified answer"
        assert response.text.index("event: replace") < response.text.index("event: done")
        assert payload["generation_usage"]["retrieval_signature"] == [{
            "chunk_id": "source",
            "score": 1.0,
            "match_type": "relation+semantic",
        }]


def test_ask_request_rejects_unknown_response_model():
    with pytest.raises(ValueError):
        AskRequest.model_validate({"chart_json": {}, "question": "Question", "response_model": "other"})


def test_ask_request_accepts_intel_npu_response_model():
    request = AskRequest.model_validate({
        "chart_json": {},
        "question": "Question",
        "response_model": "astro-qwen-npu",
    })
    assert request.response_model == "astro-qwen-npu"


def test_stream_response_dispatches_to_provider_stream_method(monkeypatch):
    provider = FakeGenerator("test-model", "openvino_npu")
    qwen = FakeGenerator("local", "transformers")
    qwen_unloaded = []
    qwen.unload = lambda: qwen_unloaded.append(True)
    provider.generate_stream = lambda *_: iter(["first", " second"])
    monkeypatch.setattr(response_generator, "get_openvino_npu_client", lambda: provider)
    monkeypatch.setattr(response_generator, "get_qwen_client", lambda: qwen)

    deltas, selected = response_generator.stream_response(
        "astro-qwen-npu", "system", "user"
    )

    assert selected is provider
    assert qwen_unloaded == []
    assert list(deltas) == ["first", " second"]
    assert qwen_unloaded == [True]


def test_openai_client_streams_text_deltas_and_records_final_diagnostics():
    class Responses:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return iter([
                SimpleNamespace(type="response.output_text.delta", delta="Hello"),
                SimpleNamespace(type="response.output_text.delta", delta=" world"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        status="completed",
                        incomplete_details=None,
                        usage=None,
                    ),
                ),
            ])

    client = LLMClient()
    client._client = SimpleNamespace(responses=Responses())

    assert list(client.generate_stream("system", "user")) == ["Hello", " world"]
    assert client.last_diagnostics["status"] == "completed"


def test_ask_stream_emits_metadata_sanitized_deltas_and_final_response(monkeypatch):
    provider = FakeGenerator(ADAPTER_MODEL, "transformers")

    def stream_response(model, system_prompt, user_prompt):
        assert model == "astro-qwen-lora"
        return iter([
            "Grounded ",
            "<think>private reasoning</think>",
            "<tool_call>{\"secret\":true}</tool_call>",
            "answer.",
        ]), provider

    monkeypatch.setattr(
        ask_api,
        "_prepare_ask_context",
        lambda req: ({}, [], "rules", [], "system", "user"),
    )
    monkeypatch.setattr(ask_api.cr, "chart_summary", lambda _: {"ascendant_sign": "Aries"})
    monkeypatch.setattr(ask_api, "stream_response", stream_response)

    response = TestClient(app).post(
        "/ask/stream",
        json={
            "chart_json": {},
            "question": "What does my chart suggest?",
            "response_model": "astro-qwen-lora",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    frames = [
        frame for frame in response.text.split("\n\n")
        if frame.strip()
    ]
    events = {
        frame.splitlines()[0].removeprefix("event: "):
            __import__("json").loads(frame.split("data: ", 1)[1])
        for frame in frames
    }
    assert events["metadata"]["chart_snapshot"] == {"ascendant_sign": "Aries"}
    assert events["metadata"]["model"] == ADAPTER_MODEL
    assert events["metadata"]["generation_mode"] == "transformers"
    assert response.text.index("event: progress") < response.text.index("event: metadata")
    assert response.text.count("event: progress") == 2
    assert response.text.index('"stage": "preparing"') < response.text.index('"stage": "generating"')
    assert response.text.index("event: metadata") < response.text.index("event: delta")
    assert response.text.index("event: delta") < response.text.index("event: replace")
    assert response.text.index("event: replace") < response.text.index("event: done")
    assert "private reasoning" not in response.text
    assert "secret" not in response.text
    assert events["replace"]["answer"] == "Grounded answer."
    assert events["done"]["answer"] == "Grounded answer."
    assert events["done"]["generation_status"] == "completed"


def test_openvino_stream_stops_generation_on_repeated_answer_header(monkeypatch):
    recorded = {"statuses": []}

    class Pipeline:
        def get_generation_config(self):
            return SimpleNamespace(apply_chat_template=True)

        def generate(self, prompts, config, streamer):
            recorded["apply_chat_template"] = config.apply_chat_template
            recorded["statuses"].append(streamer("**Answer:** First answer."))
            recorded["statuses"].append(streamer("\n**Answer:** Repeated answer."))
            return SimpleNamespace(
                texts=["**Answer:** First answer.\n**Answer:** Repeated answer."],
                perf_metrics=SimpleNamespace(
                    get_num_generated_tokens=lambda: 40
                ),
            )

    client = OpenVINONPUClient()
    client._pipeline = Pipeline()
    client._tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1, 2])
    client._fit_prompt_to_npu_budget = lambda *_: "rendered prompt"
    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        SimpleNamespace(
            StreamingStatus=SimpleNamespace(STOP="stop", RUNNING="running"),
            GenerationConfig=lambda: SimpleNamespace(apply_chat_template=True),
        ),
    )
    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_new_tokens",
        100,
    )

    output = list(client.generate_stream("system", "user"))

    assert "".join(output) == "**Answer:** First answer."
    assert recorded["statuses"] == ["running", "stop"]
    assert recorded["apply_chat_template"] is False
    assert client.last_diagnostics["status"] == "completed"
    assert client.last_diagnostics["usage"]["streaming_callback_used"] is True


def test_openvino_stream_stops_on_fragmented_reasoning_heading_without_leaking_it(monkeypatch):
    recorded = {"statuses": []}

    class Pipeline:
        def get_generation_config(self):
            return SimpleNamespace(apply_chat_template=True)

        def generate(self, prompts, config, streamer):
            for chunk in (
                "The chart suggests a mixed result.",
                "\n\n**Reason",
                "ing",
                ":** private analysis",
            ):
                status = streamer(chunk)
                recorded["statuses"].append(status)
                if status == "stop":
                    break
            return SimpleNamespace(
                texts=["The chart suggests a mixed result.\n\n**Reasoning:** private analysis"],
                perf_metrics=SimpleNamespace(get_num_generated_tokens=lambda: 20),
            )

    client = OpenVINONPUClient()
    client._pipeline = Pipeline()
    client._tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1, 2])
    client._fit_prompt_to_npu_budget = lambda *_: "rendered prompt"
    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        SimpleNamespace(
            StreamingStatus=SimpleNamespace(STOP="stop", RUNNING="running"),
            GenerationConfig=lambda: SimpleNamespace(apply_chat_template=True),
        ),
    )
    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_new_tokens",
        100,
    )

    output = list(client.generate_stream("system", "user"))

    assert "".join(output) == "The chart suggests a mixed result."
    assert recorded["statuses"] == ["running", "running", "running", "stop"]


def test_qwen_local_stream_hides_thought_and_tool_call_content(monkeypatch):
    class Tensor:
        def to(self, device):
            return self

    class Streamer:
        def __init__(self, *_args, **_kwargs):
            pass

        def __iter__(self):
            return iter([
                "Visible ",
                "<think>private reasoning</think>",
                "<tool_call>{\"secret\":true}</tool_call>",
                "answer.",
            ])

    class Tokenizer:
        eos_token_id = 1
        unk_token_id = 0

        def apply_chat_template(self, *_args, **_kwargs):
            return {"input_ids": Tensor()}

        def convert_tokens_to_ids(self, token):
            return 2 if token == "<|im_end|>" else 0

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

        def generate(self, **kwargs):
            assert kwargs["streamer"] is not None

    client = QwenLocalClient()
    client._tokenizer = Tokenizer()
    client._model = Model()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=nullcontext),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TextIteratorStreamer=Streamer),
    )

    output = list(client.generate_stream("system", "user"))

    assert "".join(output) == "Visible answer."
    assert client.last_diagnostics["status"] == "completed"


def test_qwen_client_reports_missing_optional_dependencies(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch intentionally unavailable in this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    client = QwenLocalClient()

    with pytest.raises(RAGError, match="requirements-local-model.txt") as exc_info:
        client._ensure_loaded()

    assert exc_info.value.status_code == 503
    assert client.last_diagnostics["status"] == "error"


def test_qwen_generation_uses_chat_template_and_returns_only_new_tokens(monkeypatch):
    recorded = {}

    class InputIds(list):
        shape = (1, 3)

        def to(self, device):
            recorded["input_device"] = device
            return self

    class Tokenizer:
        eos_token_id = 7
        unk_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            recorded["messages"] = messages
            recorded["template_kwargs"] = kwargs
            return {"input_ids": InputIds([1, 2, 3])}

        def decode(self, tokens, **kwargs):
            recorded["decoded_tokens"] = tokens
            return "Grounded Qwen answer"

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 9

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

        def generate(self, **kwargs):
            recorded["generation_kwargs"] = kwargs
            return [[1, 2, 3, 4, 5]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    client = QwenLocalClient()
    client._model = Model()
    client._tokenizer = Tokenizer()

    answer = client.generate("System instructions", "Chart and evidence prompt")

    assert answer == "Grounded Qwen answer"
    assert recorded["messages"] == [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "Chart and evidence prompt"},
    ]
    assert recorded["template_kwargs"]["add_generation_prompt"] is True
    assert recorded["decoded_tokens"] == [4, 5]
    assert recorded["generation_kwargs"]["max_new_tokens"] == 700
    assert recorded["generation_kwargs"]["do_sample"] is False
    assert recorded["generation_kwargs"]["repetition_penalty"] == 1.15
    assert recorded["generation_kwargs"]["eos_token_id"] == [7, 9]
    assert recorded["generation_kwargs"]["pad_token_id"] == 7
    assert client.last_diagnostics["status"] == "completed"


def test_qwen_output_cleanup_removes_internal_tags_and_echoed_passage_blocks():
    answer = _clean_model_output(
        '<tool_call>{"name":"lookup","arguments":{}}</tool_call>\n'
        "<tool_call>\n\nSupported answer.\n<think>private reasoning</think>\n"
        "More answer.\n[Passage 2 — source: book]\nraw retrieval text"
    )

    assert answer == "Supported answer.\n\nMore answer."


def test_qwen_output_cleanup_strips_reasoning_leakage_after_answer():
    answer = _clean_model_output(
        "**Answer:** Mercury occupies H4 Leo.\n"
        "**Reasoning:** This response incorrectly attributes the 4th lordship."
    )

    assert answer == "**Answer:** Mercury occupies H4 Leo."


def test_stream_filter_hides_a_reasoning_heading_split_across_chunks():
    chunks = [
        "The chart suggests a mixed result.",
        "\n\n**Reason",
        "ing",
        ":** internal text",
    ]
    raw = ""
    visible = ""
    for chunk in chunks:
        raw += chunk
        visible, _ = _stream_visible_text(raw)

    assert visible == "The chart suggests a mixed result."
    assert _stream_visible_text("The reasoning behind this is historical.") == (
        "The reasoning behind this is historical.",
        False,
    )


def test_stream_filter_catches_complete_heading_and_preserves_normal_answer():
    answer = "A direct answer.\n\n### Chart Factors Examined: private notes"

    assert _stream_visible_text(answer) == ("A direct answer.", True)
    assert _stream_visible_text("A direct answer with no hidden section.") == (
        "A direct answer with no hidden section.",
        False,
    )


def test_passage_citation_validation_corrects_source_and_removes_unknown_passage():
    passages = [
        {"metadata": {"source": {"book": "Brihat Jataka"}}},
        {"metadata": {"source": {"book": "Phaladeepika"}}},
    ]

    repaired = _repair_passage_citations(
        "Passage 1 — Invented Book supports this. Passage 8 — Unknown Source.",
        passages,
    )

    assert repaired == "Passage 1 — Brihat Jataka supports this."


def test_unsupported_divisional_claim_is_limited_without_rewriting_general_text():
    chart = {"planets": [], "houses": []}
    answer = (
        "Your D-9 shows a strong placement. "
        "D-9 is traditionally used for marriage."
    )

    repaired = _remove_unsupported_divisional_claims(answer, chart)

    assert "does not include D-9" in repaired
    assert "D-9 is traditionally used for marriage." in repaired


def test_supplied_divisional_data_preserves_chart_claim():
    chart = {"divisional_charts": {"D9": {"planets": []}}}
    answer = "Your D-9 shows a strong placement."

    assert _remove_unsupported_divisional_claims(answer, chart) == answer


def test_qwen_output_cleanup_removes_unverified_yoga_names_and_internal_notes():
    answer = _clean_model_output(
        "This is Dwipaka Yoga and Drishti Yoga. "
        "[Chart Verification Note: Mercury is in H4 (NOT H5).]"
    )

    assert answer == "This is classical combination and classical combination."


def test_chart_claim_verifier_repairs_planet_placement_house_lordship_and_yoga():
    chart = {
        "planets": [
            {
                "body": "Mercury",
                "position": {"zodiac_sign": "Leo"},
                "placement": {"house": 4},
            }
        ],
        "houses": [
            {
                "number": number,
                "sign": sign,
                "lord": lord,
            }
            for number, sign, lord in [
                (1, "Aries", "Mars"),
                (2, "Gemini", "Mercury"),
                (3, "Cancer", "Moon"),
                (4, "Leo", "Sun"),
                (5, "Virgo", "Mercury"),
                (6, "Libra", "Venus"),
                (7, "Scorpio", "Mars"),
                (8, "Sagittarius", "Jupiter"),
                (9, "Capricorn", "Saturn"),
                (10, "Aquarius", "Saturn"),
                (11, "Pisces", "Jupiter"),
                (12, "Taurus", "Venus"),
            ]
        ],
    }
    answer = (
        "Your 5th lord Mercury is in the 5th house (H5). "
        "Mercury rules both the 5th (Virgo) and 11th houses (Pisces). "
        "This is Purva Bhanga Yoga, unlike Raja Yoga."
    )

    repaired = verify_and_repair_chart_claims(answer, chart, [])

    assert "Mercury is in the 4th house (Leo/H4)" in repaired
    assert "Mercury rules the 2nd (Gemini) and 5th (Virgo) houses" in repaired
    assert "Purva Bhanga Yoga" not in repaired
    assert "classical combination" in repaired
    assert "Raja Yoga" in repaired


def test_chart_claim_verifier_repairs_wrong_lord_and_h_number_but_not_hypotheticals():
    chart = {
        "planets": [
            {"body": "Mars", "position": {"zodiac_sign": "Leo"}, "placement": {"house": 4}},
            {"body": "Rahu", "position": {"zodiac_sign": "Aries"}, "placement": {"house": 12}},
        ],
        "houses": [
            {"number": 4, "sign": "Leo", "lord": "Sun"},
            {"number": 12, "sign": "Aries", "lord": "Mars"},
        ],
    }
    actual_chart_claim = (
        "In your chart, your 12th lord **Rahu** occupies `H12` (Aries), "
        "which is the 4th house (`H4/Leo`)."
    )
    repaired = verify_and_repair_chart_claims(actual_chart_claim, chart, [])
    assert "12th lord **Mars**" in repaired
    assert "4th house (Leo/H4)" in repaired

    hypothetical = (
        "According to BPHS, if the 12th lord Rahu occupies the 12th house, "
        "the native may travel abroad."
    )
    assert verify_and_repair_chart_claims(hypothetical, chart, []) == hypothetical


def test_chart_claim_verifier_repairs_incorrect_dignity_but_preserves_hypotheticals():
    chart = {
        "planets": [
            {"body": "Mercury", "position": {"zodiac_sign": "Leo"}, "placement": {"house": 4}},
            {"body": "Venus", "position": {"zodiac_sign": "Virgo"}, "placement": {"house": 5}},
        ],
    }
    answer = (
        "In your chart, Mercury is exalted in Leo. "
        "Your chart shows Venus is in its own sign."
    )

    repaired = verify_and_repair_chart_claims(answer, chart, [])

    assert "Mercury is in an ordinary sign" in repaired
    assert "Venus is debilitated" in repaired
    assert "Mercury is exalted" not in repaired
    assert "Venus is in its own sign" not in repaired

    hypothetical = (
        "According to a classical rule, if Mercury is exalted in Virgo, "
        "the result may differ."
    )
    assert verify_and_repair_chart_claims(hypothetical, chart, []) == hypothetical


def test_chart_claim_verifier_does_not_infer_unprovided_house_lordships():
    chart = {
        "planets": [],
        "houses": [
            {"number": 5, "sign": "Virgo", "lord": "Mercury"},
        ],
    }

    answer = verify_and_repair_chart_claims(
        "Mercury rules both the 5th and 11th houses.",
        chart,
        [],
    )

    assert answer == "Mercury rules both the 5th and 11th houses."


def test_qwen_output_cleanup_repairs_headers_titles_and_trailing_rubric():
    answer = _clean_model_output(
        "Answer:\n\n**Response:** A grounded answer citing Brihat Parasara Bhashya.\n"
        "**Final Answer:** repeated answer\nPer Rule 5, add another rubric."
    )

    assert answer == (
        "**Answer:** A grounded answer citing Brihat Parasara Hora Sastra."
    )


def test_qwen_output_cleanup_removes_jammed_headers_and_repeated_sections():
    answer = _clean_model_output(
        ": **Answer:** **Reasoning:** Mercury/Pluto is mentioned.\n"
        "**Reasoning:** First explanation.\n"
        "**Answer:** Repeated answer."
    )

    assert answer == "**Answer:** Mercury is mentioned."


def test_qwen_output_cleanup_normalizes_known_corrupted_titles():
    answer = _clean_model_output(
        "Sources: Brihat Parasara Hora Sastry; Brihat Parashar Samhita; "
        "Brihat Parasara Hora Samhasta; Brihadaranyananda; (**BPBH**; (**BP**)**; "
        "BPSh; BPP."
    )

    assert answer.count("Brihat Parasara Hora Sastra") == 5
    assert answer.endswith("BPHS; BPHS.")


def test_qwen_output_cleanup_normalizes_corrupted_book_names_and_labels():
    answer = _clean_model_output(
        "Brihat Parashar Samh Sastra, BPBHParasara, Phala Deepika, "
        "Brihat Jatakam, Jataka Parijatha, Slipper 36, Passge 3."
    )

    assert answer == (
        "Brihat Parasara Hora Sastra, Brihat Parasara Hora Sastra, Phaladeepika, "
        "Brihat Jataka, Jataka Parijata, Sloka 36, Passage 3."
    )


def test_chart_facts_explicitly_label_ordinary_sign_dignity():
    from app.generation.prompt import _format_chart_facts

    facts = _format_chart_facts(
        [{"type": "planet_house", "planet": "Mercury", "house": 4, "sign": "Leo"}]
    )

    assert "- Mercury: house=H4; sign=Leo" in facts
    assert "dignity=" not in facts


def test_adapter_tokenizer_uses_embedded_template_when_jinja_file_is_absent(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    recorded = {}

    class Tokenizer:
        chat_template = "template from tokenizer config"

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            recorded["source"] = source
            recorded["kwargs"] = kwargs
            return Tokenizer()

    tokenizer = _load_adapter_tokenizer(
        TokenizerLoader,
        tmp_path,
        "cached-token",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )

    assert tokenizer.chat_template == "template from tokenizer config"
    assert recorded["source"] == str(tmp_path)
    assert recorded["kwargs"]["token"] == "cached-token"


def test_adapter_tokenizer_prefers_standalone_template_file(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("adapter template", encoding="utf-8")
    recorded = {}

    class Tokenizer:
        chat_template = "base template"

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            recorded["source"] = source
            return Tokenizer()

    tokenizer = _load_adapter_tokenizer(
        TokenizerLoader,
        tmp_path,
        None,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )

    assert recorded["source"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert tokenizer.chat_template == "adapter template"


def test_openvino_npu_client_uses_npu_and_reports_runtime(monkeypatch, tmp_path):
    import types

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "openvino_model.xml").write_text("", encoding="utf-8")
    recorded = {}

    class GenerationConfig:
        def __init__(self):
            self.apply_chat_template = True
            self.max_new_tokens = None
            self.do_sample = None

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            recorded["prompt"] = kwargs
            return "rendered prompt"

        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, tokens, **kwargs):
            return "".join(tokens)

    class Pipeline:
        def get_generation_config(self):
            recorded["generation_config_loaded_from_pipeline"] = True
            return GenerationConfig()

        def generate(self, prompt, config):
            recorded["device"] = "NPU"
            recorded["generation"] = (prompt, config.max_new_tokens)
            recorded["do_sample"] = config.do_sample
            recorded["apply_chat_template"] = config.apply_chat_template
            recorded["repetition_penalty"] = config.repetition_penalty
            recorded["stop_strings"] = config.stop_strings
            recorded["include_stop_str_in_output"] = config.include_stop_str_in_output
            return types.SimpleNamespace(
                texts=["<tool_call>\n\n<think>internal</think>NPU answer"]
            )

    class Core:
        available_devices = ["NPU"]

    def create_pipeline(path, device, config):
        recorded["pipeline_config"] = config
        return Pipeline()

    openvino_module = types.SimpleNamespace(Core=Core)
    genai_module = types.SimpleNamespace(
        GenerationConfig=GenerationConfig,
        LLMPipeline=create_pipeline,
    )
    transformers_module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda path, **kwargs: Tokenizer()
        )
    )
    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_model_dir",
        model_dir,
    )
    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        1024,
    )
    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_new_tokens",
        600,
    )
    monkeypatch.setitem(sys.modules, "openvino", openvino_module)
    monkeypatch.setitem(sys.modules, "openvino_genai", genai_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    client = OpenVINONPUClient()
    assert client.generate("System", "User") == "NPU answer"
    assert recorded["device"] == "NPU"
    assert recorded["pipeline_config"] == {
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 600,
        "CACHE_DIR": str(model_dir / ".npucache"),
    }
    assert recorded["generation"][0] == ["System\n\nUser"]
    assert recorded["do_sample"] is False
    assert recorded["apply_chat_template"] is False
    assert recorded["repetition_penalty"] == 1.15
    assert recorded["stop_strings"] == {
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
    assert recorded["include_stop_str_in_output"] is False
    assert recorded["generation_config_loaded_from_pipeline"] is True
    assert recorded["generation"][1] == 600
    usage = client.last_diagnostics["usage"]
    assert usage["device"] == "NPU"
    assert usage["runtime"] == "OpenVINO GenAI"
    assert usage["max_prompt_len"] == 1024
    assert usage["prompt_tokens"] == len("System\n\nUser")
    assert usage["max_new_tokens"] == 600
    assert usage["generated_tokens"] is None
    assert usage["returned_text_count"] == 1
    assert usage["prompt_sha256"]
    assert usage["model_fingerprint"]
    assert usage["do_sample"] is False
    assert usage["repetition_penalty"] == 1.15
    assert usage["streaming_callback_used"] is False
    assert usage["generation_config"]["max_new_tokens"] == 600
    assert usage["generation_config"]["do_sample"] is False
    assert len(usage["prompt_sha256"]) == 16
    assert len(usage["model_fingerprint"]) == 16


def test_openvino_npu_empty_result_logs_openvino_result_metadata(monkeypatch, caplog):
    import logging
    import types

    class GenerationConfig:
        max_new_tokens = None
        do_sample = None
        apply_chat_template = True

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    class PerfMetrics:
        def get_num_generated_tokens(self):
            return 0

    class Pipeline:
        def get_generation_config(self):
            return GenerationConfig()

        def generate(self, prompts, config):
            assert prompts == ["System\n\nUser"]
            return types.SimpleNamespace(
                texts=[""],
                perf_metrics=PerfMetrics(),
            )

    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        types.SimpleNamespace(GenerationConfig=GenerationConfig),
    )
    client = OpenVINONPUClient()
    client._pipeline = Pipeline()
    client._tokenizer = Tokenizer()

    with caplog.at_level(logging.WARNING, logger="app.generation.openvino_npu_client"):
        with pytest.raises(RAGError, match="returned no visible answer"):
            client.generate("System", "User")

    assert (
        "[NPU] Returned no visible output" in caplog.text
        and "result_type=SimpleNamespace" in caplog.text
        and "returned_text_count=1" in caplog.text
        and "generated_tokens=0" in caplog.text
    )
    assert client.last_diagnostics["usage"] == {
        "device": "NPU",
        "prompt_tokens": len("System\n\nUser"),
        "raw_output_chars": 0,
        "result_type": "SimpleNamespace",
        "returned_text_count": 1,
        "generated_tokens": 0,
    }


def test_openvino_npu_marks_output_incomplete_when_token_limit_is_reached(monkeypatch):
    import types

    class GenerationConfig:
        max_new_tokens = None
        do_sample = None
        apply_chat_template = True

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    class PerfMetrics:
        def get_num_generated_tokens(self):
            return 600

    class Pipeline:
        def get_generation_config(self):
            return GenerationConfig()

        def generate(self, prompts, config):
            self.config = config
            return types.SimpleNamespace(
                texts=["A visible NPU answer."],
                perf_metrics=PerfMetrics(),
            )

    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        types.SimpleNamespace(GenerationConfig=GenerationConfig),
    )
    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_new_tokens",
        600,
    )
    client = OpenVINONPUClient()
    client._pipeline = Pipeline()
    client._tokenizer = Tokenizer()

    assert client.generate("System", "User") == "A visible NPU answer."
    assert client._pipeline.config.max_new_tokens == 600
    assert client.last_diagnostics["status"] == "incomplete"
    assert client.last_diagnostics["incomplete_reason"] == "max_new_tokens"


def test_openvino_npu_prompt_compaction_trims_only_passages(monkeypatch):
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return (
                f"SYSTEM\n{messages[0]['content']}\nUSER\n"
                f"{messages[1]['content']}\nASSISTANT"
            )

        def encode(self, text, **kwargs):
            return [text[index:index + 4] for index in range(0, len(text), 4)]

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        1500,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    user_prompt = (
        "CHART SNAPSHOT\nAscendant: Aries\n\n"
        "Houses: H1:Aries(Sun) H2:Taurus(Venus)\n\n"
        "RELEVANT CHART FACTS FOR THIS QUESTION\nCHART FACTS MUST STAY\n\n"
        "RETRIEVED REFERENCE PASSAGES\n"
        + "[Passage 1 — source: provenance]\n"
        + "- --14\nafin |\nATA\nARSE\nSioku D.--If the Sun occupy H4.\n"
        + "FIRST CORE RULE.\n"
        + ("English translation. raw OCR हिन्दी text. " * 200)
        + "\n\nQUESTION\nQUESTION MUST STAY"
    )

    fitted = client._fit_prompt_to_npu_budget("short system", user_prompt)

    assert len(client._tokenizer.encode(fitted)) <= 1372
    assert "CHART FACTS MUST STAY" in fitted
    assert "QUESTION MUST STAY" in fitted
    assert "provenance" in fitted
    assert "Houses: H1 sign=Aries lord=Sun occupants=none" in fitted
    assert "FIRST CORE RULE." in fitted
    assert "If the Sun occupy H4." in fitted
    assert "Sioku D." not in fitted
    assert "ATA" not in fitted
    assert "ARSE" not in fitted
    assert "हिन्दी" not in fitted
    assert " ..." in fitted


def test_openvino_npu_prompt_budget_preserves_every_passage_header(monkeypatch):
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return (
                f"SYSTEM\n{messages[0]['content']}\nUSER\n"
                f"{messages[1]['content']}\nASSISTANT"
            )

        def encode(self, text, **kwargs):
            return [text[index:index + 4] for index in range(0, len(text), 4)]

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        1024,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    user_prompt = (
        "CHART SNAPSHOT\nAries ascendant\n\n"
        "RELEVANT CHART FACTS FOR THIS QUESTION\nMars in house 3\n\n"
        "RETRIEVED REFERENCE PASSAGES\n"
        "[Passage 1 — source: Book One]\nFIRST CORE RULE. " + ("Long note. " * 30) + "\n\n"
        "[Passage 2 — source: Book Two]\nSECOND CORE RULE. " + ("Long note. " * 30) + "\n\n"
        "[Passage 3 — source: Book Three]\nTHIRD CORE RULE. " + ("Long note. " * 30)
        + "\nQUESTION\nKeep this exact question"
    )

    fitted = client._fit_prompt_to_npu_budget("x" * 3000, user_prompt)

    assert len(client._tokenizer.encode(fitted)) <= 896
    assert "careful Vedic astrology" in fitted
    assert "Occupancy is not lordship" in fitted
    assert "DIGNITY: Exalted" in fitted
    assert "[Passage 1 — source: Book One]" in fitted
    assert "[Passage 2 — source: Book Two]" in fitted
    assert "[Passage 3 — source: Book Three]" in fitted
    assert "FIRST CORE RULE." in fitted
    assert "SECOND CORE RULE." in fitted
    assert "THIRD CORE RULE." in fitted
    assert "Keep this exact question" in fitted


def test_npu_passage_token_allocation_prioritizes_higher_ranked_passages():
    entries = [
        (f"[Passage {index}]", list(range(500)), "")
        for index in range(1, 4)
    ]

    assert _allocate_passage_token_limits(entries, 110) == [50, 40, 20]
    assert _allocate_passage_token_limits(entries, 1) == [1, 0, 0]
    assert _allocate_passage_token_limits(entries, 2000) == [500, 500, 500]


def test_npu_fixed_context_compaction_preserves_chart_relations_and_citations(monkeypatch):
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"{messages[0]['content']}\n{messages[1]['content']}"

        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        750,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    user_prompt = (
        "CHART SNAPSHOT\n"
        "Ascendant: Gemini | Moon sign: Cancer | Sun sign: Virgo\n"
        "Houses: H1:Gemini(Mercury/Moon) H2:Cancer(Moon) "
        "H3:Leo(Sun/Mars) H4:Virgo(Mercury)\n"
        "Planets: Mercury:H4/Virgo[lord of H1,H4]\n\n"
        "RELEVANT CHART FACTS FOR THIS QUESTION\n"
        "- Mercury: house=H4; sign=Virgo; role=lord_of_H10\n"
        "- H4: sign=Virgo; lord=Mercury; occupants=Mercury\n\n"
        "RETRIEVED REFERENCE PASSAGES\n"
        "[Passage 1 — source: Brihat Jataka | Chapter 4 — Long title | PDF p. 11, match: direct]\n"
        "A concise retrieved rule. " + ("Additional supporting detail. " * 30) + "\n\n"
        "QUESTION\nHow is Mercury relevant?"
    )

    fitted = client._fit_prompt_to_npu_budget("System", user_prompt)

    assert len(client._tokenizer.encode(fitted)) <= 622
    assert "H1 sign=Gemini lord=Mercury occupants=Moon" in fitted
    assert "Mercury sign=Virgo house=H4 lord_of=H1,H4" in fitted
    assert "Mercury: house=H4; sign=Virgo; role=lord_of_H10" in fitted
    assert "H4: sign=Virgo; lord=Mercury; occupants=Mercury" in fitted
    assert "[Passage 1 — Brihat Jataka]" in fitted
    assert "A concise retrieved rule." in fitted
    assert "How is Mercury relevant?" in fitted
    assert "Long title" not in fitted


def test_npu_header_budget_drops_low_ranked_passages_and_keeps_notes(monkeypatch):
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"{messages[0]['content']}\n{messages[1]['content']}"

        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        610,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    user_prompt = (
        "CHART SNAPSHOT\nChart summary\n\n"
        "RELEVANT CHART FACTS FOR THIS QUESTION\nRelevant chart fact\n\n"
        "RETRIEVED REFERENCE PASSAGES\n"
        "[Passage 1 — source: First Book]\nRule one.\n"
        "[Chart Binding: " + ("First passage chart binding. " * 4) + "]\n\n"
        "[Passage 2 — source: Second Book]\nRule two.\n"
        "[Chart Verification Note: " + ("Second passage unmet condition. " * 3) + "]\n\n"
        "[Passage 3 — source: Third Book]\nRule three.\n"
        "[Chart Verification Note: " + ("Third passage unmet condition. " * 4) + "]\n\n"
        "QUESTION\nWhat does this indicate?"
    )

    fitted = client._fit_prompt_to_npu_budget("System", user_prompt)

    assert len(client._tokenizer.encode(fitted)) <= 482
    assert "[Passage 1" in fitted
    assert "[Passage 2" in fitted
    assert "[Passage 3" not in fitted
    assert "First passage chart binding." in fitted
    assert "Second passage unmet condition." in fitted
    assert "Third passage unmet condition." not in fitted
    assert "What does this indicate?" in fitted


def test_openvino_npu_prompt_uses_chat_template_with_thinking_disabled():
    recorded = {}

    class ChatTokenizer:
        chat_template = "Qwen chat template"

        def apply_chat_template(self, messages, **kwargs):
            recorded["messages"] = messages
            recorded["kwargs"] = kwargs
            return "chat-rendered prompt"

        def encode(self, text, **kwargs):
            return list(text)

    client = OpenVINONPUClient()
    client._tokenizer = ChatTokenizer()

    fitted = client._fit_prompt_to_npu_budget("system prompt", "user prompt")

    assert fitted == "chat-rendered prompt"
    assert recorded["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert recorded["kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_openvino_prompt_closes_unsupported_qwen_thinking_template():
    class QwenTemplateTokenizer:
        chat_template = "Qwen template that ignores enable_thinking"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "<|im_start|>system\nSystem<|im_end|>\n<|im_start|>user\nUser<|im_end|>\n<|im_start|>assistant\n"

        def encode(self, text, **kwargs):
            return list(text)

    client = OpenVINONPUClient()
    client._tokenizer = QwenTemplateTokenizer()

    fitted = client._fit_prompt_to_npu_budget("System", "User")

    assert fitted.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_openvino_npu_prompt_keeps_notes_for_short_verse_entries():
    class CharacterTokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    user_prompt = (
        "CHART SNAPSHOT\nChart\n\n"
        "RETRIEVED REFERENCE PASSAGES\n"
        "[Passage 1 — source: Brihat Jataka]\n"
        "Short Sanskrit verse.\nNotes: The English commentary explains the rule.\n\n"
        "QUESTION\nWhat does it mean?"
    )

    fitted = client._fit_prompt_to_npu_budget("System", user_prompt)

    assert "Short Sanskrit verse." in fitted
    assert "Note: The English commentary explains the rule." in fitted


def test_openvino_npu_prompt_preserves_verification_note_during_compaction(monkeypatch):
    class CharacterTokenizer:
        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        900,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    note = (
        "[Chart Verification Note: Mercury is in H4 Virgo (NOT H5). "
        "Do not claim this condition applies.]"
    )
    user_prompt = (
        "CHART SNAPSHOT\nChart\n\n"
        "RETRIEVED REFERENCE PASSAGES\n"
        "[Passage 1 — source: Brihat Jataka]\n"
        + "Long rule body. " * 150
        + "\n"
        + note
        + "\n\nQUESTION\nWhat does it mean?"
    )

    fitted = client._fit_prompt_to_npu_budget("System", user_prompt)

    assert note in fitted
    assert "Long rule body." in fitted


def test_openvino_npu_prompt_preserves_chart_binding_during_compaction(monkeypatch):
    class CharacterTokenizer:
        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        900,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    note = (
        "[Chart Binding: H11 lord Jupiter is placed in H5 Virgo; "
        "distinguish this from H5 lord Mercury, placed in H4 Leo.]"
    )
    user_prompt = (
        "CHART SNAPSHOT\nChart\n\n"
        "RETRIEVED REFERENCE PASSAGES\n"
        "[Passage 1 — source: Brihat Jataka]\n"
        + "Long rule body. " * 150
        + "\n"
        + note
        + "\n\nQUESTION\nWhat does it mean?"
    )

    fitted = client._fit_prompt_to_npu_budget("System", user_prompt)

    assert note in fitted


def test_openvino_npu_prompt_budget_has_no_fixed_850_token_cap(monkeypatch):
    class CharacterTokenizer:
        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        1400,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    user_prompt = (
        "CHART SNAPSHOT\nAries ascendant\n\n"
        "RELEVANT CHART FACTS FOR THIS QUESTION\nAll facts stay\n\n"
        "RETRIEVED REFERENCE PASSAGES\n"
        "[Passage 1 — source: Book One | Chapter 1 — Long chapter title | PDF p. 11, match: direct]\n"
        + "Rule text. " * 300
        + "\nQUESTION\nQuestion stays"
    )

    fitted = client._fit_prompt_to_npu_budget("System", user_prompt)

    assert len(fitted) <= 1272
    assert "All facts stay" in fitted
    assert "Question stays" in fitted
    assert "Chapter 1 — Long chapter title" not in fitted
    assert "PDF p. 11" not in fitted


def test_openvino_npu_prompt_preserves_all_retrieved_passages(monkeypatch):
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"{messages[0]['content']}\n{messages[1]['content']}"

        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        2048,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    passages = "\n\n".join(
        f"[Passage {i} — source: Book {i}]\nRule {i}."
        for i in range(1, 6)
    )
    user_prompt = (
        "CHART SNAPSHOT\nAries ascendant\n\n"
        "RELEVANT CHART FACTS FOR THIS QUESTION\nAll facts stay\n\n"
        f"RETRIEVED REFERENCE PASSAGES\n{passages}\n\n"
        "QUESTION\nQuestion stays"
    )

    fitted = client._fit_prompt_to_npu_budget("System", user_prompt)

    for i in range(1, 6):
        assert f"[Passage {i} — source: Book {i}]" in fitted


def test_openvino_npu_prompt_budget_errors_when_fixed_context_is_too_long(monkeypatch):
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"{messages[0]['content']}\n{messages[1]['content']}"

        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_max_prompt_len",
        100,
    )
    client = OpenVINONPUClient()
    client._tokenizer = CharacterTokenizer()
    user_prompt = (
        "CHART SNAPSHOT\nLong fixed chart details\n\n"
        "RETRIEVED REFERENCE PASSAGES\nPassage text\n\n"
        "QUESTION\nA question"
    )

    with pytest.raises(RAGError, match="exceed the configured NPU token budget"):
        client._fit_prompt_to_npu_budget("system details " * 10, user_prompt)


def test_openvino_npu_client_rejects_missing_model_without_fallback(monkeypatch, tmp_path):
    from app.generation.openvino_npu_client import OpenVINONPUClient

    monkeypatch.setattr(
        "app.generation.openvino_npu_client.settings.openvino_npu_model_dir",
        tmp_path / "missing",
    )
    client = OpenVINONPUClient()

    with pytest.raises(RAGError, match="model is not prepared"):
        client._ensure_loaded()

    assert client.last_diagnostics["status"] == "error"
