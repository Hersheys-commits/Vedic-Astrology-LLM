"""LLM generation for chart facts + retrieved reference passages.

Uses the OpenAI API when ``OPENAI_API_KEY`` is configured. Falls back to a
deterministic mock mode so the chart -> facts -> retrieval path can be tested
without network access or API cost.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.config.settings import settings
from app.utils.errors import RAGError


def _mock_generate(system_prompt: str, user_prompt: str) -> str:
    """Return a deterministic preview grounded in the retrieved passages."""
    marker = "RETRIEVED REFERENCE PASSAGES"
    question_marker = "QUESTION"
    passages_block = ""

    if marker in user_prompt and question_marker in user_prompt:
        passages_block = (
            user_prompt.split(marker, 1)[1]
            .split(question_marker, 1)[0]
            .strip()
        )

    snippet = passages_block[:600] + ("..." if len(passages_block) > 600 else "")
    return (
        "[MOCK MODE — set OPENAI_API_KEY in .env for a real generated answer]\n\n"
        "Based on the retrieved passages below, here is a preview of "
        "what the LLM would be grounded in:\n\n"
        f"{snippet or '(no passages retrieved)'}"
    )


class LLMClient:
    # keep your existing __init__ and mode property

    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self._client = None

        self.last_diagnostics: dict[str, Any] = {
            "status": "not_called",
            "incomplete_reason": None,
            "usage": None,
        }

        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                self._client = None

    @property
    def mode(self) -> str:
        return "openai" if self._client else "mock"

    def _record_diagnostics(self, response: Any) -> None:
        incomplete_reason = None

        details = getattr(response, "incomplete_details", None)
        if details is not None:
            incomplete_reason = getattr(details, "reason", None)

            if incomplete_reason is None and isinstance(details, dict):
                incomplete_reason = details.get("reason")

        usage = getattr(response, "usage", None)
        usage_dict = None

        if usage is not None:
            try:
                usage_dict = usage.model_dump()
            except Exception:
                try:
                    usage_dict = dict(usage)
                except Exception:
                    usage_dict = None

        self.last_diagnostics = {
            "status": getattr(response, "status", None),
            "incomplete_reason": incomplete_reason,
            "usage": usage_dict,
        }

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        m = (model or "").lower()
        return (
            m.startswith("gpt-5")
            or m.startswith("o1")
            or m.startswith("o3")
            or m.startswith("o4")
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:

        # Mock mode
        if not self._client:
            answer = _mock_generate(system_prompt, user_prompt)

            self.last_diagnostics = {
                "status": "completed",
                "incomplete_reason": None,
                "usage": None,
            }

            return answer

        chosen_model = model or self.model
        output_budget = max_tokens or settings.generation_max_tokens

        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": output_budget,
        }

        # Reasoning configuration
        if self._is_reasoning_model(chosen_model):
            kwargs["reasoning"] = {
                "effort": settings.openai_reasoning_effort
            }

        try:
            response = self._client.responses.create(**kwargs)

        except Exception as exc:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": None,
            }

            raise RAGError(
                "llm_generation_failed",
                f"OpenAI generation failed: {exc}",
                502,
            ) from exc

        self._record_diagnostics(response)

        status = getattr(response, "status", None)
        answer = (getattr(response, "output_text", None) or "").strip()

        # Retry if reasoning consumed the output budget
        if (
            status == "incomplete"
            and answer == ""
            and self.last_diagnostics.get("incomplete_reason")
            == "max_output_tokens"
        ):
            retry_budget = max(output_budget * 2, 5000)

            retry_kwargs = dict(kwargs)
            retry_kwargs["max_output_tokens"] = retry_budget

            try:
                retry_response = self._client.responses.create(
                    **retry_kwargs
                )

            except Exception as exc:
                raise RAGError(
                    "llm_generation_failed",
                    f"OpenAI retry failed: {exc}",
                    502,
                ) from exc

            self._record_diagnostics(retry_response)

            answer = (
                getattr(retry_response, "output_text", None) or ""
            ).strip()

            status = getattr(retry_response, "status", None)

        # Never silently return ""
        if not answer:
            reason = (
                self.last_diagnostics.get("incomplete_reason")
                or "no_visible_output"
            )

            raise RAGError(
                "llm_empty_output",
                (
                    "The OpenAI response contained no visible answer "
                    f"(status={status}, reason={reason}). "
                    "Check generation limits and the server log."
                ),
                502,
            )

        return answer

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Yield OpenAI Responses API deltas; mock mode emits one preview chunk."""
        if not self._client:
            yield self.generate(
                system_prompt, user_prompt, max_tokens=max_tokens, model=model
            )
            return

        chosen_model = model or self.model
        output_budget = max_tokens or settings.generation_max_tokens
        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": output_budget,
            "stream": True,
        }
        if self._is_reasoning_model(chosen_model):
            kwargs["reasoning"] = {"effort": settings.openai_reasoning_effort}

        try:
            stream = self._client.responses.create(**kwargs)
            answer_chunks: list[str] = []
            final_response = None
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        answer_chunks.append(delta)
                        yield delta
                elif event_type in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    final_response = getattr(event, "response", None)

            if final_response is not None:
                self._record_diagnostics(final_response)
                if getattr(final_response, "status", None) == "failed":
                    raise RAGError(
                        "llm_generation_failed",
                        "OpenAI streaming generation failed. Check the provider "
                        "response and server logs.",
                        502,
                    )
            else:
                self.last_diagnostics = {
                    "status": "completed",
                    "incomplete_reason": None,
                    "usage": None,
                }
            if not "".join(answer_chunks).strip():
                answer = self.generate(
                    system_prompt, user_prompt, max_tokens=max_tokens, model=model
                )
                yield answer
        except RAGError:
            raise
        except Exception as exc:
            self.last_diagnostics = {
                "status": "error",
                "incomplete_reason": None,
                "usage": None,
            }
            raise RAGError(
                "llm_generation_failed",
                f"OpenAI streaming generation failed: {exc}",
                502,
            ) from exc


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
