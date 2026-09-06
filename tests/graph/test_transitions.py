import pytest

from board_of_scientists.schemas.state import create_initial_state
from board_of_scientists.graph import workflow
from board_of_scientists.graph.workflow import build_research_graph


AGENT_EVAL_NODES = {
    "eval_analyst": "analyst",
    "eval_theorist": "theorist",
    "eval_architect": "architect",
    "eval_engineer": "engineer",
    "eval_reviewer": "reviewer",
    "eval_experiment": "experiment",
    "eval_writer": "writer",
}

STAGE_NODES = [
    "analyst", "cro_read", "theorist", "architect", "cro_plan",
    "engineer", "reviewer", "experiment", "writer", "cro_verdict", "output",
]


@pytest.mark.parametrize(
    "failed_eval,expected_retry,expected_next",
    [
        ("eval_analyst", "analyst", "cro_read"),
        ("eval_reviewer", "engineer", "experiment"),
        ("eval_experiment", "engineer", "writer"),
        ("eval_writer", "writer", "cro_verdict"),
    ],
)
def test_graph_executes_requested_retry_transition(monkeypatch, failed_eval, expected_retry, expected_next):
    trace = []
    attempts = {key: 0 for key in AGENT_EVAL_NODES.values()}

    def make_stage_node(name):
        def node(state):
            trace.append(name)
            return state
        return node

    def make_eval_node(eval_node_name, agent_key):
        def node(state):
            attempts[agent_key] += 1
            trace.append(eval_node_name)
            failed_once = eval_node_name == failed_eval and attempts[agent_key] == 1
            state["communication"].evaluations[agent_key] = {"passed": not failed_once}
            return state
        return node

    for name in STAGE_NODES:
        monkeypatch.setattr(workflow, f"node_{name}", make_stage_node(name))
    for eval_node, agent_key in AGENT_EVAL_NODES.items():
        monkeypatch.setattr(workflow, f"node_{eval_node}", make_eval_node(eval_node, agent_key))

    graph = build_research_graph()
    result = graph.invoke(create_initial_state("paper.pdf"))

    # The run must pass through the failing evaluation, revisit the expected
    # repair node, then continue beyond that evaluation into the normal path.
    first_eval = trace.index(failed_eval)
    assert trace[first_eval + 1] == expected_retry
    second_eval = trace.index(failed_eval, first_eval + 1)
    assert trace[second_eval + 1] == expected_next
    assert result["communication"].evaluations[AGENT_EVAL_NODES[failed_eval]]["passed"] is True


def test_graph_normal_path_reaches_output(monkeypatch):
    trace = []

    def stage(name):
        def node(state):
            trace.append(name)
            return state
        return node

    def evaluation(name):
        agent_key = AGENT_EVAL_NODES[name]

        def node(state):
            trace.append(name)
            state["communication"].evaluations[agent_key] = {"passed": True}
            return state

        return node

    for name in STAGE_NODES:
        monkeypatch.setattr(workflow, f"node_{name}", stage(name))
    for name in AGENT_EVAL_NODES:
        monkeypatch.setattr(workflow, f"node_{name}", evaluation(name))

    build_research_graph().invoke(create_initial_state("paper.pdf"))
    assert trace == [
        "analyst", "eval_analyst", "cro_read", "theorist", "eval_theorist",
        "architect", "eval_architect", "cro_plan", "engineer", "eval_engineer",
        "reviewer", "eval_reviewer", "experiment", "eval_experiment",
        "writer", "eval_writer", "cro_verdict", "output",
    ]
