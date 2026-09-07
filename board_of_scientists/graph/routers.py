"""Conditional routing policy for the LangGraph workflow."""

from ..agents.registry import (
    ANALYST,
    THEORIST,
    ARCHITECT,
    ENGINEER,
    REVIEWER,
    EXPERIMENT,
    WRITER,
)
from ..schemas.state import ResearchState

MAX_AGENT_REVISIONS = 3


def _failed(state: ResearchState, name: str) -> bool:
    evaluations = state["communication"].evaluations
    return name in evaluations and not evaluations[name].get("passed", True)


def _quality_gate_exhausted(state: ResearchState, name: str) -> bool:
    """Never convert a maxed-out revision loop into an approval."""
    revisions = state["communication"].revision_counts.get(name, 0)
    evaluation = state["communication"].evaluations.get(name)
    return bool(evaluation and revisions >= MAX_AGENT_REVISIONS)


def _route(state: ResearchState, name: str, advance: str):
    if _quality_gate_exhausted(state, name):
        return "quality_gate_failed"
    return name if _failed(state, name) else advance


def route_analyst(state: ResearchState):
    return _route(state, ANALYST, "cro_read")


def route_theorist(state: ResearchState):
    return _route(state, THEORIST, "architect")


def route_architect(state: ResearchState):
    return _route(state, ARCHITECT, "cro_plan")


def route_engineer(state: ResearchState):
    return _route(state, ENGINEER, "reviewer")


def route_reviewer(state: ResearchState):
    return _route(state, REVIEWER, "experiment")


def route_experiment(state: ResearchState):
    return _route(state, EXPERIMENT, "writer")


def route_writer(state: ResearchState):
    return _route(state, WRITER, "cro_verdict")


__all__ = [
    "MAX_AGENT_REVISIONS",
    "route_analyst",
    "route_theorist",
    "route_architect",
    "route_engineer",
    "route_reviewer",
    "route_experiment",
    "route_writer",
]
