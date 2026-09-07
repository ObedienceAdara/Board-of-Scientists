"""Reliability wrappers for the historical agent runtime.

These wrappers are installed once by ``agents.__init__`` so the legacy agent
functions receive bounded retries and contextual failures without keeping a
second implementation of every agent.
"""

from __future__ import annotations

import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def _retryable(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "429", "rate limit", "too many requests", "temporarily unavailable",
        "service unavailable", "gateway timeout", "timeout", "timed out",
        "connection reset", "connection aborted", "502", "503", "504",
    )
    return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(marker in text for marker in markers)


def install(runtime_module) -> None:
    """Replace fragile runtime LLM helpers with bounded, observable wrappers."""
    parser = StrOutputParser()

    def run_chain(prompt_template: str, inputs: dict, model: str) -> str:
        last_exc: BaseException | None = None
        for attempt in range(2):
            try:
                llm = runtime_module.make_llm(model)
                prompt = ChatPromptTemplate.from_template(prompt_template)
                return (prompt | llm | parser).invoke(inputs)
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and _retryable(exc):
                    time.sleep(1.0)
                    continue
                raise RuntimeError(
                    f"LLM narrative call failed for model={model!r} "
                    f"on attempt {attempt + 1}: {type(exc).__name__}: {exc}"
                ) from exc
        raise RuntimeError(f"LLM narrative call failed: {last_exc}") from last_exc

    def run_structured_chain(prompt_template: str, inputs: dict, model: str, schema, temperature: float = 0.1):
        llm = runtime_module.make_llm(model, temperature)
        structured_llm = llm.with_structured_output(schema)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | structured_llm
        last_exc: BaseException | None = None
        for attempt in range(2):
            try:
                return chain.invoke(inputs)
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and _retryable(exc):
                    time.sleep(1.0)
                    continue
                # Structured-output parse/validation failures are worth one
                # immediate retry; deterministic configuration/programming
                # failures are surfaced immediately with context.
                text = str(exc).lower()
                if attempt == 0 and any(k in text for k in ("validation", "structured", "parse", "json", "tool_calls")):
                    continue
                raise runtime_module.StructuredOutputError(
                    f"Structured output failed for {schema.__name__}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        raise runtime_module.StructuredOutputError(
            f"Structured output failed twice for {schema.__name__}: "
            f"{type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    runtime_module.run_chain = run_chain
    runtime_module.run_structured_chain = run_structured_chain


__all__ = ["install"]
