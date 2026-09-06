"""LangGraph node functions.

The node layer owns graph-facing adapters only. Business logic stays in the
specialized agent, execution, ingestion, and reporting modules.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from ..agents.analyst import analyst_agent
from ..agents.theorist import theorist_agent
from ..agents.architect import architect_agent
from ..agents.engineer import engineer_agent
from ..agents.reviewer import reviewer_agent
from ..agents.experiment import experiment_engineer_agent
from ..agents.writer import writer_agent
from ..agents.cro import (
    cro_read_paper,
    cro_create_plan,
    cro_evaluate_agent,
    cro_final_verdict,
)
from ..agents.registry import (
    ANALYST, THEORIST, ARCHITECT, ENGINEER, REVIEWER, EXPERIMENT, WRITER,
)
from ..schemas.state import ResearchState
from ..reports.pdf import generate_implementation_report
from ..reports.provenance import save_all_modules, save_message_board


def node_analyst(state):
    return analyst_agent(state)


def node_cro_read(state):
    return cro_read_paper(state)


def node_theorist(state):
    return theorist_agent(state)


def node_architect(state):
    return architect_agent(state)


def node_cro_plan(state):
    return cro_create_plan(state)


def node_engineer(state):
    return engineer_agent(state)


def node_reviewer(state):
    return reviewer_agent(state)


def node_experiment(state):
    return experiment_engineer_agent(state)


def node_writer(state):
    return writer_agent(state)


def node_cro_verdict(state):
    return cro_final_verdict(state)


def node_eval_analyst(state):
    return cro_evaluate_agent(state, ANALYST, "research_report")


def node_eval_theorist(state):
    return cro_evaluate_agent(state, THEORIST, "theoretical_analysis")


def node_eval_architect(state):
    return cro_evaluate_agent(state, ARCHITECT, "architecture_analysis")


def node_eval_engineer(state):
    return cro_evaluate_agent(state, ENGINEER, "implementation_notes")


def node_eval_reviewer(state):
    return cro_evaluate_agent(state, REVIEWER, "review_summary")


def node_eval_experiment(state):
    return cro_evaluate_agent(state, EXPERIMENT, "validation_report")


def node_eval_writer(state):
    return cro_evaluate_agent(state, WRITER, "readme")


def node_output(state: ResearchState) -> ResearchState:
    """Persist generated artifacts and the final implementation report."""
    paper_slug = (
        state.get("paper_title", "paper")
        .lower()
        .replace(" ", "_")
        .replace("/", "_")[:40]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{paper_slug}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved = save_all_modules(output_dir, state.get("code_modules", {}))
    board_path = save_message_board(output_dir, state.get("message_board", []))

    measured = state.get("measured_validation")
    measured_appendix = (
        json.dumps(measured, indent=2)
        if measured
        else "No measured execution results are available for this run."
    )

    sections = [
        {"title": "Abstract & Overview", "content": state.get("paper_abstract", "")},
        {"title": "CRO Reading Notes", "content": state.get("cro_reading_notes", "")},
        {"title": "Theoretical Analysis", "content": state.get("theoretical_analysis", "")},
        {"title": "Architecture Design", "content": state.get("architecture_analysis", "")},
        {"title": "Implementation Plan", "content": state.get("implementation_plan", "")},
        {"title": "Code Review Findings", "content": json.dumps(state.get("review_feedback", {}), indent=2)},
        {
            "title": "Validation Report (Experiment Engineer's analysis — see appendix for raw measurements)",
            "content": state.get("validation_report", ""),
        },
        {
            "title": "Appendix: Raw Measured Execution Results (ground truth, not LLM-generated)",
            "content": measured_appendix,
        },
        {"title": "CRO Final Verdict", "content": state.get("final_verdict", "")},
        {
            "title": "Team Communications Log",
            "content": json.dumps(state.get("message_board", []), indent=2),
        },
    ]

    pdf_path = os.path.join(output_dir, "implementation_report.pdf")
    generate_implementation_report(
        {
            "paper_title": state.get("paper_title", "Research Paper"),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sections": sections,
        },
        pdf_path,
    )

    return {
        **state,
        "output_dir": output_dir,
        "pdf_report_path": pdf_path,
    }


__all__ = [
    "node_analyst", "node_cro_read", "node_theorist", "node_architect",
    "node_cro_plan", "node_engineer", "node_reviewer", "node_experiment",
    "node_writer", "node_cro_verdict", "node_eval_analyst", "node_eval_theorist",
    "node_eval_architect", "node_eval_engineer", "node_eval_reviewer",
    "node_eval_experiment", "node_eval_writer", "node_output",
]
