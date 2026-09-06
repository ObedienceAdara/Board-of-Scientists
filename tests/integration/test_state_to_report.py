import json
from pathlib import Path

from board_of_scientists.graph.state_adapter import to_runtime_state
from board_of_scientists.reports.pdf import generate_implementation_report
from board_of_scientists.reports.provenance import save_message_board
from board_of_scientists.schemas.state import create_initial_state


def test_state_payload_can_be_projected_into_report_artifacts(tmp_path: Path):
    state = create_initial_state("paper.pdf")
    state["research_input"].paper_title = "Integration Paper"
    state["research_input"].paper_abstract = "A deterministic abstract."
    state["analysis"].theoretical_analysis = "A theorem-level observation."
    state["validation"].measured_validation = {"accuracy": 0.95}
    state["communication"].message_board = [{"sender": "analyst", "content": "done"}]

    runtime = to_runtime_state(state)
    report_path = tmp_path / "implementation_report.pdf"
    generate_implementation_report(
        {
            "paper_title": runtime["paper_title"],
            "date": "2026-09-06",
            "sections": [
                {"title": "Abstract", "content": runtime["paper_abstract"]},
                {"title": "Theory", "content": runtime["theoretical_analysis"]},
                {"title": "Measurements", "content": json.dumps(runtime["measured_validation"])},
            ],
        },
        str(report_path),
    )

    board_path = save_message_board(str(tmp_path), runtime["message_board"])
    assert report_path.exists() and report_path.stat().st_size > 100
    assert Path(board_path).exists()
