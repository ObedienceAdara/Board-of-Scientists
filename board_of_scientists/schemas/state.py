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
    # sender/recipient are CANONICAL AGENT KEYS (see agent_registry.py), e.g.
    # "engineer", "reviewer", "all" — never human display names. Routing
    # (get_messages_for) matches on these keys. Display names are looked up
    # from the registry only when rendering something for a human to read.
    sender:         str
    recipient:      str
    sender_name:    str   # human-readable, for logs/reports only
    recipient_name: str   # human-readable, for logs/reports only
    content:        str
    message_type:   str  # "insight" / "concern" / "question" / "answer" / "directive" / "feedback"


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
    page_notes:           str    # human-readable joined per-page analyst notes (debug/log use)
    page_notes_list:      list   # list[str] — the individual per-batch notes, kept
                                  # separately (not just joined) so both a CRO-triggered
                                  # revision AND the hierarchical reduce step can re-run
                                  # without re-reading the whole PDF. This is also what
                                  # let the old hard 40-page cap be removed: an arbitrarily
                                  # long list of notes gets recursively merged down to a
                                  # handful of consolidated summaries (map-reduce) instead
                                  # of being joined into one giant string and truncated.

    # ── Phase 1: Deep Analysis ──────────────────────────────
    theoretical_analysis: str    # Theorist's deep mathematical breakdown
    architecture_analysis: str   # Architect's structural breakdown
    file_manifest:         list  # list[FileSpec-as-dict] — the Architect's file-by-file
                                  # implementation plan, in dependency order. The Engineer
                                  # iterates this one file (or small group) at a time
                                  # instead of writing the whole codebase in one call.

    # ── Phase 2: CRO Synthesis ──────────────────────────────
    cro_reading_notes:    str    # CRO's own reading notes on the paper
    implementation_plan:  str    # Master implementation blueprint
    codebase_structure:   str    # File/module structure the Architect designed

    # ── Phase 3: Implementation ─────────────────────────────
    code_modules:         dict   # { filename: CodeModule }
    review_feedback:      dict   # { filename: review_notes }
    review_summary:       str    # latest full review text — this is what the CRO
                                  # actually evaluates (previously the CRO evaluated
                                  # a key, "review_feedback_str", that was never set,
                                  # i.e. it always evaluated an empty string)
    implementation_notes: str    # Engineer's notes during implementation

    # ── Phase 4: Validation ─────────────────────────────────
    execution_results:    str    # human-readable rendering of the REAL, measured
                                  # sandbox execution results (syntax/import/
                                  # instantiation checks) — never LLM-invented
    measured_validation:  dict   # the raw structured measurement dict itself
    validation_report:    str    # Experiment Engineer's write-up. Anything in here
                                  # not sourced from measured_validation must be
                                  # tagged [LLM-INFERRED] by the agent.
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
