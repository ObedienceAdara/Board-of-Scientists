from board_of_scientists.graph.nodes import _run_agent
from board_of_scientists.graph.state_adapter import from_runtime_state, to_runtime_state
from board_of_scientists.schemas.evidence import ConsistencyIssue
from board_of_scientists.schemas.state import create_initial_state


def test_agent_output_round_trips_through_domain_state():
    state = create_initial_state("paper.pdf")

    def fake_agent(runtime):
        assert runtime["pdf_path"] == "paper.pdf"
        return {
            **runtime,
            "paper_title": "Agent Test",
            "research_report": "analysis result",
            "code_modules": {
                "model.py": {
                    "filename": "model.py",
                    "language": "python",
                    "code": "print('ok')",
                    "description": "test model",
                    "status": "draft",
                }
            },
            "revision_counts": {"analyst": 1},
        }

    result = _run_agent(fake_agent, state)
    assert result["research_input"].paper_title == "Agent Test"
    assert result["analysis"].research_report == "analysis result"
    assert result["implementation"].code_modules["model.py"]["status"] == "draft"
    assert result["communication"].revision_counts["analyst"] == 1


def test_runtime_projection_copies_nested_collections():
    state = create_initial_state("paper.pdf")
    issue = ConsistencyIssue(source="engine", message="contradiction", related=("claim:c1",))
    state["evidence"].consistency_issues.append(issue)
    state["validation"].measured_validation["accuracy"] = 0.91
    runtime = to_runtime_state(state)
    assert runtime["consistency_issues"][0]["message"] == "contradiction"
    assert runtime["measured_validation"]["accuracy"] == 0.91
    rebuilt = from_runtime_state(runtime)
    assert rebuilt["evidence"].consistency_issues[0] == issue
    assert rebuilt["validation"].measured_validation["accuracy"] == 0.91
