import pytest

from board_of_scientists.graph.nodes import node_quality_gate_failed
from board_of_scientists.graph.routers import (
    route_analyst, route_architect, route_engineer, route_experiment,
    route_reviewer, route_theorist, route_writer,
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
def test_failed_evaluation_routes_to_correct_retry(agent_key, router, retry_target, advance_target):
    state = create_initial_state("paper.pdf")
    state["communication"].evaluations[agent_key] = {"passed": False}
    state["communication"].revision_counts[agent_key] = 1
    assert router(state) == retry_target

    state["communication"].evaluations[agent_key] = {"passed": True}
    assert router(state) == advance_target


@pytest.mark.parametrize("agent_key,router,_,__", ROUTES)
def test_failed_evaluation_after_revision_budget_routes_to_terminal(agent_key, router, _, __):
    state = create_initial_state("paper.pdf")
    state["communication"].evaluations[agent_key] = {"passed": False}
    state["communication"].revision_counts[agent_key] = 3
    assert router(state) == "quality_gate_failed"


def test_failed_quality_gate_node_sets_unresolved_verdict():
    state = create_initial_state("paper.pdf")
    state["communication"].evaluations["analyst"] = {"passed": False}
    state["communication"].revision_counts["analyst"] = 3

    result = node_quality_gate_failed(state)
    verdict = result["output"].final_verdict
    assert verdict.startswith("UNRESOLVED — QUALITY GATE EXHAUSTED")
    assert "analyst" in verdict


def test_normal_route_inputs_are_explicitly_initialized():
    state = create_initial_state("paper.pdf")
    assert state["communication"].evaluations == {}
    assert state["communication"].revision_counts == {}
    assert route_analyst(state) == "cro_read"
