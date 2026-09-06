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


def _failed(state, name):
    evaluations = state.get("evaluations", {})
    return name in evaluations and not evaluations[name].get("passed", True)


def route_analyst(state):
    return "analyst" if _failed(state, ANALYST) else "cro_read"


def route_theorist(state):
    return "theorist" if _failed(state, THEORIST) else "architect"


def route_architect(state):
    return "architect" if _failed(state, ARCHITECT) else "cro_plan"


def route_engineer(state):
    return "engineer" if _failed(state, ENGINEER) else "reviewer"


def route_reviewer(state):
    return "engineer" if _failed(state, REVIEWER) else "experiment"


def route_experiment(state):
    return "engineer" if _failed(state, EXPERIMENT) else "writer"


def route_writer(state):
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
