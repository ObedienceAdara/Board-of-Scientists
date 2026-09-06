"""Shared agent utilities kept separate from role-specific logic."""
from board_of_scientists._legacy.agents import (
    make_llm, sanitize_prompt_input, StructuredOutputError, get_feedback,
    format_prior_feedback, post_message, get_messages_for, run_chain,
    run_structured_chain, increment_revision,
)
__all__ = ["make_llm","sanitize_prompt_input","StructuredOutputError","get_feedback","format_prior_feedback","post_message","get_messages_for","run_chain","run_structured_chain","increment_revision"]
