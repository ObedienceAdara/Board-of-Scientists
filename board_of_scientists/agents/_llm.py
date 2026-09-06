"""
llm_provider.py — Provider-aware LLM factory.

THE BUG THIS FILE FIXES
------------------------
The old make_llm() hardcoded ChatOpenAI pointed at Groq's endpoint
("https://api.groq.com/openai/v1") and read GROQ_API_KEY, with
default_headers ("HTTP-Referer" / "X-Title") that only mean anything on
OpenRouter. Meanwhile env.example and main.py's own docstring told users to
override models with OpenRouter-style IDs like "anthropic/claude-3.5-sonnet"
or "deepseek/deepseek-r1" — which Groq does not serve. Following the
project's own setup instructions broke it: Groq would reject those model IDs.

There was no real "OpenRouter support" — just leftover comments from an
incomplete migration.

THE FIX
-------
An explicit LLM_PROVIDER env var ("groq" | "openrouter" | "openai") selects
one of three real, first-party LangChain integrations — each with its own
correct default model, correct API key variable, and correct base URL:

  - groq       -> langchain_groq.ChatGroq          (GROQ_API_KEY)
  - openrouter -> langchain_openrouter.ChatOpenRouter (OPENROUTER_API_KEY)
  - openai     -> langchain_openai.ChatOpenAI       (OPENAI_API_KEY)

We deliberately do NOT reuse "ChatOpenAI with a custom base_url" as a generic
shim for Groq/OpenRouter. LangChain's own docs now flag that pattern as a
known source of "broken structured output, missing reasoning content,
unsupported provider-specific features, and incorrect tracing" — which would
have directly undermined the structured-output fix elsewhere in this
codebase. Each provider gets its real, purpose-built integration package.

If the required API key for the selected provider is missing, make_llm()
fails immediately with a clear message naming the exact env var to set,
instead of letting a cryptic 401 surface three network calls deep into a
LangGraph run.
"""

import os
from typing import Callable


class LLMConfigError(RuntimeError):
    """Raised when the selected LLM_PROVIDER is missing required configuration."""


def _get_groq_chat():
    from langchain_groq import ChatGroq
    return ChatGroq


def _get_openrouter_chat():
    from langchain_openrouter import ChatOpenRouter
    return ChatOpenRouter


def _get_openai_chat():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI


# Provider config: base default model + API key env var + how to construct
# the chat model instance. Model IDs below are real, valid IDs for that
# provider (not borrowed from a different provider's catalogue).
_PROVIDER_CONFIG = {
    "groq": {
        "api_key_env":     "GROQ_API_KEY",
        "default_model":   "llama-3.3-70b-versatile",
        "get_chat_cls":    _get_groq_chat,
        "extra_kwargs":    lambda: {},
    },
    "openrouter": {
        "api_key_env":     "OPENROUTER_API_KEY",
        "default_model":   "meta-llama/llama-3.3-70b-instruct",
        "get_chat_cls":    _get_openrouter_chat,
        "extra_kwargs":    lambda: {},
    },
    "openai": {
        "api_key_env":     "OPENAI_API_KEY",
        "default_model":   "gpt-4o-mini",
        "get_chat_cls":    _get_openai_chat,
        "extra_kwargs":    lambda: {},
    },
}


def get_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in _PROVIDER_CONFIG:
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER='{provider}'. Valid options: {sorted(_PROVIDER_CONFIG)}."
        )
    return provider


def default_model_for_provider(provider: str = None) -> str:
    provider = provider or get_provider()
    return _PROVIDER_CONFIG[provider]["default_model"]


def make_llm(model: str = None, temperature: float = 0.1):
    """
    Build a chat model using whichever provider LLM_PROVIDER selects.

    `model` may be None, in which case the correct default model for the
    selected provider is used (never a different provider's model ID).
    """
    provider = get_provider()
    cfg = _PROVIDER_CONFIG[provider]

    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise LLMConfigError(
            f"LLM_PROVIDER='{provider}' requires {cfg['api_key_env']} to be set "
            f"(see env.example). No API key found in the environment."
        )

    chat_cls: Callable = cfg["get_chat_cls"]()
    resolved_model = model or cfg["default_model"]

    return chat_cls(
        model=resolved_model,
        temperature=temperature,
        api_key=api_key,
        **cfg["extra_kwargs"](),
    )
