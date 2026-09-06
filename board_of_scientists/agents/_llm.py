"""Provider-aware LLM factory boundary."""
from board_of_scientists._legacy.llm_provider import LLMConfigError, get_provider, default_model_for_provider, make_llm
__all__ = ["LLMConfigError","get_provider","default_model_for_provider","make_llm"]
