"""Conditional routing policy for the LangGraph workflow.

Routing reads only the ``CommunicationState`` domain object. It deliberately
contains no agent implementation imports beyond the canonical identity registry.
"""

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


def _failed(state: ResearchState, name: str) -> bool:
    evaluations = state["communication"].evaluations
    return name in evaluations and not evaluations[name].get("passed", True)


def route_analyst(state: ResearchState):
    return "analyst" if _failed(state, ANALYST) else "cro_read"


def route_theorist(state: ResearchState):
    return "theorist" if _failed(state, THEORIST) else "architect"


def route_architect(state: ResearchState):
    return "architect" if _failed(state, ARCHITECT) else "cro_plan"


def route_engineer(state: ResearchState):
    return "engineer" if _failed(state, ENGINEER) else "reviewer"


def route_reviewer(state: ResearchState):
    return "engineer" if _failed(state, REVIEWER) else "experiment"


def route_experiment(state: ResearchState):
    return "engineer" if _failed(state, EXPERIMENT) else "writer"


def route_writer(state: ResearchState):
    return "writer" if _failed(state, WRITER) else "cro_verdict"


__all__ = [
    "route_analyst",
    "route_theorist",
    "route_architect",
    "route_engineer",
    "route_reviewer",
    "route_experiment",
    "route_writer",
]
