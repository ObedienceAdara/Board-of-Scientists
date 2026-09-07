"""LangGraph node adapters.

Graph nodes speak the canonical nested ``ResearchState`` domain model. The
current behavior-preserving agent implementation still uses a flat runtime
mapping, so this module owns the only projection boundary between those two
representations.
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
from .state_adapter import from_runtime_state, to_runtime_state


def _run_agent(agent, state: ResearchState) -> ResearchState:
    """Invoke an agent behind the domain/runtime state boundary."""
    runtime_state = to_runtime_state(state)
    result = agent(runtime_state)
    return from_runtime_state(result)


def node_analyst(state):
    return _run_agent(analyst_agent, state)


def node_cro_read(state):
    return _run_agent(cro_read_paper, state)


def node_theorist(state):
    return _run_agent(theorist_agent, state)


def node_architect(state):
    return _run_agent(architect_agent, state)


def node_cro_plan(state):
    """Run the CRO planning step without destroying the Architect's design."""
    architect_structure = state["architecture"].codebase_structure
    result = _run_agent(cro_create_plan, state)
    if architect_structure:
        result["architecture"].codebase_structure = architect_structure
    return result


def node_engineer(state):
    return _run_agent(engineer_agent, state)


def node_reviewer(state):
    return _run_agent(reviewer_agent, state)


def node_experiment(state):
    return _run_agent(experiment_engineer_agent, state)


def node_writer(state):
    return _run_agent(writer_agent, state)


def node_cro_verdict(state):
    return _run_agent(cro_final_verdict, state)


def _run_eval(state: ResearchState, agent_key: str, output_key: str) -> ResearchState:
    runtime_state = to_runtime_state(state)
    result = cro_evaluate_agent(runtime_state, agent_key, output_key)
    return from_runtime_state(result)


def node_eval_analyst(state):
    return _run_eval(state, ANALYST, "research_report")


def node_eval_theorist(state):
    return _run_eval(state, THEORIST, "theoretical_analysis")


def node_eval_architect(state):
    return _run_eval(state, ARCHITECT, "architecture_analysis")


def node_eval_engineer(state):
    return _run_eval(state, ENGINEER, "implementation_notes")


def node_eval_reviewer(state):
    return _run_eval(state, REVIEWER, "review_summary")


def node_eval_experiment(state):
    return _run_eval(state, EXPERIMENT, "validation_report")


def node_eval_writer(state):
    return _run_eval(state, WRITER, "readme")


def node_output(state: ResearchState) -> ResearchState:
    """Persist generated artifacts and the final implementation report."""
    runtime = to_runtime_state(state)

    paper_slug = (
        runtime.get("paper_title", "paper")
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")[:40]
        or "paper"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(os.getenv("OUTPUT_DIR", ".")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"output_{paper_slug}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_all_modules(str(output_dir), runtime.get("code_modules", {}))
    save_message_board(str(output_dir), runtime.get("message_board", []))

    measured = runtime.get("measured_validation")
    measured_appendix = (
        json.dumps(measured, indent=2)
        if measured
        else "No measured execution results are available for this run."
    )

    sections = [
        {"title": "Abstract & Overview", "content": runtime.get("paper_abstract", "")},
        {"title": "CRO Reading Notes", "content": runtime.get("cro_reading_notes", "")},
        {"title": "Theoretical Analysis", "content": runtime.get("theoretical_analysis", "")},
        {"title": "Architecture Design", "content": runtime.get("architecture_analysis", "")},
        {"title": "Implementation Plan", "content": runtime.get("implementation_plan", "")},
        {"title": "Code Review Findings", "content": json.dumps(runtime.get("review_feedback", {}), indent=2)},
        {
            "title": "Validation Report (Experiment Engineer's analysis — see appendix for raw measurements)",
            "content": runtime.get("validation_report", ""),
        },
        {
            "title": "Appendix: Raw Measured Execution Results (ground truth, not LLM-generated)",
            "content": measured_appendix,
        },
        {"title": "CRO Final Verdict", "content": runtime.get("final_verdict", "")},
        {
            "title": "Team Communications Log",
            "content": json.dumps(runtime.get("message_board", []), indent=2),
        },
    ]

    pdf_path = output_dir / "implementation_report.pdf"
    generate_implementation_report(
        {
            "paper_title": runtime.get("paper_title", "Research Paper"),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sections": sections,
        },
        str(pdf_path),
    )

    runtime["output_dir"] = str(output_dir)
    runtime["pdf_report_path"] = str(pdf_path)
    return from_runtime_state(runtime)


__all__ = [
    "node_analyst", "node_cro_read", "node_theorist", "node_architect",
    "node_cro_plan", "node_engineer", "node_reviewer", "node_experiment",
    "node_writer", "node_cro_verdict", "node_eval_analyst", "node_eval_theorist",
    "node_eval_architect", "node_eval_engineer", "node_eval_reviewer",
    "node_eval_experiment", "node_eval_writer", "node_output",
]
