"""Conditional routing policy for the LangGraph workflow."""

from ..agents.registry import ANALYST, THEORIST, ARCHITECT, ENGINEER, REVIEWER, EXPERIMENT, WRITER
from ..schemas.state import ResearchState

MAX_AGENT_REVISIONS = 3


def _failed(state: ResearchState, name: str) -> bool:
    evaluations = state["communication"].evaluations
    evaluation = evaluations.get(name)
    return evaluation is not None and not evaluation.get("passed", True)


def _revision_count(state: ResearchState, name: str) -> int:
    return state["communication"].revision_counts.get(name, 0)


def _route(state: ResearchState, name: str, retry_target: str, advance_target: str) -> str:
    """Return retry/advance/terminal route without ever treating a failed gate as success."""
    if _failed(state, name) and _revision_count(state, name) >= MAX_AGENT_REVISIONS:
        return "quality_gate_failed"
    if _failed(state, name):
        return retry_target
    return advance_target


def route_analyst(state: ResearchState):
    return _route(state, ANALYST, ANALYST, "cro_read")


def route_theorist(state: ResearchState):
    return _route(state, THEORIST, THEORIST, "architect")


def route_architect(state: ResearchState):
    return _route(state, ARCHITECT, ARCHITECT, "cro_plan")


def route_engineer(state: ResearchState):
    return _route(state, ENGINEER, ENGINEER, "reviewer")


def route_reviewer(state: ResearchState):
    return _route(state, REVIEWER, ENGINEER, "experiment")


def route_experiment(state: ResearchState):
    return _route(state, EXPERIMENT, ENGINEER, "writer")


def route_writer(state: ResearchState):
    return _route(state, WRITER, WRITER, "cro_verdict")


__all__ = [
    "MAX_AGENT_REVISIONS", "route_analyst", "route_theorist", "route_architect",
    "route_engineer", "route_reviewer", "route_experiment", "route_writer",
]
