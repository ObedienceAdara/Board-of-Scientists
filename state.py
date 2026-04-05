"""
state.py — Shared state across the AI Research Implementation Team.
Every agent reads from and writes to this single object throughout the session.
"""

from typing import TypedDict


class PaperSection(TypedDict):
    page:     int
    heading:  str
    text:     str
    figures:  list   # list of { caption, description }
    tables:   list   # list of { caption, data }
    equations: list  # list of { label, latex, description }


class CodeModule(TypedDict):
    filename:    str
    language:    str
    code:        str
    description: str
    status:      str  # "draft" / "reviewed" / "validated"


class AgentMessage(TypedDict):
    sender:    str
    recipient: str   # agent name or "ALL"
    content:   str
    message_type: str  # "question" / "insight" / "concern" / "answer"


class ResearchState(TypedDict):
    # ── Input ───────────────────────────────────────────────
    pdf_path:             str
    paper_title:          str
    paper_abstract:       str

    # ── Phase 0: Ingestion ──────────────────────────────────
    raw_pages:            list   # list of PaperSection
    full_paper_text:      str    # concatenated full paper
    figures_summary:      str    # all figures described
    tables_summary:       str    # all tables described
    equations_summary:    str    # all equations extracted

    # ── Phase 1: Deep Analysis ──────────────────────────────
    theoretical_analysis: str    # Theorist's deep mathematical breakdown
    architecture_analysis: str   # Architect's structural breakdown

    # ── Phase 2: CRO Synthesis ──────────────────────────────
    cro_reading_notes:    str    # CRO's own reading notes on the paper
    implementation_plan:  str    # Master implementation blueprint
    codebase_structure:   str    # File/module structure the Architect designed

    # ── Phase 3: Implementation ─────────────────────────────
    code_modules:         dict   # { filename: CodeModule }
    review_feedback:      dict   # { filename: review_notes }
    implementation_notes: str    # Engineer's notes during implementation

    # ── Phase 4: Validation ─────────────────────────────────
    execution_results:    str    # Experiment Engineer's run results
    validation_report:    str    # Comparison against paper's reported results
    discrepancies:        str    # What doesn't match and why

    # ── Phase 5: Documentation ──────────────────────────────
    readme:               str
    implementation_paper: str    # Technical doc explaining implementation choices

    # ── Inter-agent Communication ───────────────────────────
    message_board:        list   # list of AgentMessage — agents leave notes for each other

    # ── CRO Control ─────────────────────────────────────────
    cro_directives:       dict   # { agent_name: directive_string }
    evaluations:          dict   # { agent_name: { passed, feedback, iteration } }
    revision_counts:      dict   # { agent_name: int }
    needs_revision:       list

    # ── Outputs ─────────────────────────────────────────────
    output_dir:           str    # directory where all code files are saved
    pdf_report_path:      str    # final PDF documentation
    final_verdict:        str    # CRO's verdict on implementation completeness
