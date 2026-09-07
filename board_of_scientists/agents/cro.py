"""Chief Research Officer agent operations.

The CRO quality gate is implemented here rather than delegated to the legacy
runtime bridge so evaluation failures cannot silently become approvals.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..schemas.agents import EvaluationResult
from ._llm import LLMConfigError, make_llm
from .prompts import CRO_EVALUATE_PROMPT, CRO_FINAL_VERDICT_PROMPT, CRO_IMPLEMENTATION_PLAN_PROMPT, CRO_READING_NOTES_PROMPT, CONTENT_DELIMITER_INSTRUCTION
from .registry import ALL, ENGINEER, normalize_key, post_message if False else ALL

# The runtime compatibility module owns shared message-board semantics for now;
# importing only these helpers avoids using its evaluator implementation.
from ._runtime import increment_revision, post_message, StructuredOutputError


CRO_MODEL = ""


def _model_name() -> str:
    import os
    from ._llm import default_model_for_provider
    return os.getenv("CRO_MODEL") or default_model_for_provider()


def _prior_evaluation(state: dict, agent_key: str) -> dict:
    return state.get("evaluations", {}).get(normalize_key(agent_key), {})


def _run_structured(prompt_template: str, inputs: dict, schema):
    llm = make_llm(_model_name(), 0.1)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm.with_structured_output(schema)
    return chain.invoke(inputs)


def cro_evaluate_agent(state: dict, agent_key: str, output_key: str) -> dict:
    """Evaluate an agent and fail closed if the evaluator itself is unavailable."""
    agent_key = normalize_key(agent_key)
    role_map = {
        "analyst": "Paper Analyst",
        "theorist": "Theorist",
        "architect": "ML Architect",
        "engineer": "Senior ML Engineer",
        "reviewer": "Code Reviewer",
        "experiment": "Experiment Engineer",
        "writer": "Technical Writer",
    }
    role = role_map.get(agent_key, agent_key)
    output = state.get(output_key, "")

    try:
        result = _run_structured(
            CRO_EVALUATE_PROMPT,
            {
                "agent_role": role,
                "paper_title": state.get("paper_title", ""),
                "paper_excerpt": str(state.get("research_report", ""))[:2000],
                "output": str(output)[:4000],
                "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
            },
            EvaluationResult,
        )
        passed = bool(result.passed)
        feedback = result.feedback
        issues = list(result.critical_issues)
    except Exception as exc:
        # Evaluation uncertainty is never approval. Persist a failed gate so
        # the graph's bounded retry/terminal policy can handle it explicitly.
        passed = False
        feedback = (
            "CRO evaluation was unavailable. This deliverable cannot be "
            "approved until the evaluator succeeds."
        )
        issues = [f"Evaluator failure: {type(exc).__name__}: {exc}"]

    rc = dict(state.get("revision_counts", {}))
    count = rc.get(agent_key, 0)
    evaluations = dict(state.get("evaluations", {}))
    evaluations[agent_key] = {
        "passed": passed,
        "feedback": feedback,
        "issues": issues,
        "iteration": count,
    }

    needs_revision = list(state.get("needs_revision", []))
    board = list(state.get("message_board", []))
    if not passed:
        if agent_key not in needs_revision:
            needs_revision.append(agent_key)
        board = post_message(
            {"message_board": board},
            "cro",
            agent_key,
            f"Your output needs revision. Issues: {'; '.join(issues[:3])}. {feedback[:500]}",
            "feedback",
        )
    else:
        if agent_key in needs_revision:
            needs_revision.remove(agent_key)

    return {
        **state,
        "evaluations": evaluations,
        "needs_revision": needs_revision,
        "message_board": board,
    }


def cro_read_paper(state: dict) -> dict:
    """Read the paper before the mathematical and architecture passes."""
    from ._runtime import run_chain
    output = run_chain(
        CRO_READING_NOTES_PROMPT,
        {
            "full_paper_text": str(state.get("full_paper_text", ""))[:10000],
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
        },
        _model_name(),
    )
    board = post_message(
        state, "cro", ALL,
        "I have completed my initial reading of the paper. Theorist and Architect: begin your analyses in parallel. Pay special attention to the mathematical framework and proposed architecture.",
        "directive",
    )
    return {**state, "cro_reading_notes": output, "message_board": board, "revision_counts": increment_revision(state, "cro")}


def cro_create_plan(state: dict) -> dict:
    """Create the master implementation plan without replacing Architect structure."""
    from ._runtime import run_chain
    output = run_chain(
        CRO_IMPLEMENTATION_PLAN_PROMPT,
        {
            "paper_title": state.get("paper_title", ""),
            "cro_reading_notes": str(state.get("cro_reading_notes", ""))[:3000],
            "theoretical_analysis": str(state.get("theoretical_analysis", ""))[:3000],
            "architecture_analysis": str(state.get("architecture_analysis", ""))[:3000],
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
        },
        _model_name(),
    )
    board = post_message(
        state, "cro", ENGINEER,
        "Implementation plan is ready. Begin with Phase A (core data structures). Follow the Architect's file manifest and architecture exactly. Tag me if the paper is ambiguous.",
        "directive",
    )
    return {
        **state,
        "implementation_plan": output,
        "message_board": board,
    }


def cro_final_verdict(state: dict) -> dict:
    """Issue a final CRO verdict using measured validation evidence."""
    from ._runtime import run_chain
    modules = ", ".join(state.get("code_modules", {}).keys())
    output = run_chain(
        CRO_FINAL_VERDICT_PROMPT,
        {
            "paper_title": state.get("paper_title", ""),
            "implementation_summary": str(state.get("implementation_plan", ""))[:2000],
            "validation_report": str(state.get("validation_report", ""))[:2000],
            "code_modules_list": modules,
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
        },
        _model_name(),
    )
    return {**state, "final_verdict": output}


__all__ = ["CRO_MODEL", "cro_read_paper", "cro_create_plan", "cro_evaluate_agent", "cro_final_verdict"]
