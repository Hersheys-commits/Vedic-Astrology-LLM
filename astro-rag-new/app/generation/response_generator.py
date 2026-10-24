"""Dispatch only final answer generation to the selected provider."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.generation.llm_client import get_llm_client
from app.generation.openvino_npu_client import get_openvino_npu_client
from app.generation.qwen_client import get_qwen_client


def get_response_provider(response_model: str) -> Any:
    """Select a provider once so streaming and non-streaming use the same routing."""
    if response_model == "astro-qwen-lora":
        provider = get_qwen_client()
    elif response_model == "astro-qwen-npu":
        get_qwen_client().unload()
        provider = get_openvino_npu_client()
    else:
        provider = get_llm_client()
    return provider


def generate_response(
    response_model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, Any]:
    """Return the generated answer and the provider that produced it."""
    provider = get_response_provider(response_model)
    return provider.generate(system_prompt, user_prompt), provider


def stream_response(
    response_model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[Iterator[str], Any]:
    """Return generated text deltas and the provider that produced them."""
    if response_model == "astro-qwen-npu":
        provider = get_openvino_npu_client()

        def npu_stream() -> Iterator[str]:
            get_qwen_client().unload()
            stream_method = getattr(provider, "generate_stream", None)
            if callable(stream_method):
                yield from stream_method(system_prompt, user_prompt)
            else:
                yield provider.generate(system_prompt, user_prompt)

        return npu_stream(), provider

    provider = get_response_provider(response_model)
    stream_method = getattr(provider, "generate_stream", None)
    if callable(stream_method):
        return iter(stream_method(system_prompt, user_prompt)), provider

    def single_chunk() -> Iterator[str]:
        yield provider.generate(system_prompt, user_prompt)

    return single_chunk(), provider
