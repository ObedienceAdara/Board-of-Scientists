import pytest

from board_of_scientists.graph.routers import (
    route_analyst,
    route_architect,
    route_engineer,
    route_experiment,
    route_reviewer,
    route_theorist,
    route_writer,
)
from board_of_scientists.schemas.state import create_initial_state


ROUTES = [
    ("analyst", route_analyst, "analyst", "cro_read"),
    ("theorist", route_theorist, "theorist", "architect"),
    ("architect", route_architect, "architect", "cro_plan"),
    ("engineer", route_engineer, "engineer", "reviewer"),
    ("reviewer", route_reviewer, "engineer", "experiment"),
    ("experiment", route_experiment, "engineer", "writer"),
    ("writer", route_writer, "writer", "cro_verdict"),
]


@pytest.mark.parametrize("agent_key,router,retry_target,advance_target", ROUTES)
def test_failed_evaluation_retries_agent(agent_key, router, retry_target, advance_target):
    state = create_initial_state("paper.pdf")
    state["communication"].evaluations[agent_key] = {"passed": False}
    assert router(state) == retry_target


@pytest.mark.parametrize("agent_key,router,retry_target,advance_target", ROUTES)
def test_passed_or_missing_evaluation_advances(agent_key, router, retry_target, advance_target):
    state = create_initial_state("paper.pdf")
    assert router(state) == advance_target
    state["communication"].evaluations[agent_key] = {"passed": True}
    assert router(state) == advance_target
