"""Adapter between canonical domain state and the current agent runtime shape."""

from __future__ import annotations

from typing import Any

from ..schemas.state import ResearchState

RuntimeResearchState = dict[str, Any]


def _dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def to_runtime_state(state: ResearchState) -> RuntimeResearchState:
    """Project canonical domain state into the flat agent-runtime mapping."""
    ri = state["research_input"]
    pc = state["paper_corpus"]
    an = state["analysis"]
    ar = state["architecture"]
    im = state["implementation"]
    va = state["validation"]
    ev = state["evidence"]
    co = state["communication"]
    out = state["output"]
    return {
        "pdf_path": ri.pdf_path,
        "paper_title": ri.paper_title,
        "paper_abstract": ri.paper_abstract,
        "raw_pages": pc.raw_pages,
        "full_paper_text": pc.full_paper_text,
        "figures_summary": pc.figures_summary,
        "tables_summary": pc.tables_summary,
        "equations_summary": pc.equations_summary,
        "page_notes": pc.page_notes,
        "page_notes_list": pc.page_notes_list,
        "research_report": an.research_report,
        "cro_reading_notes": an.cro_reading_notes,
        "theoretical_analysis": an.theoretical_analysis,
        "architecture_analysis": ar.architecture_analysis,
        "implementation_plan": ar.implementation_plan,
        "codebase_structure": ar.codebase_structure,
        "file_manifest": ar.file_manifest,
        "code_modules": im.code_modules,
        "review_feedback": im.review_feedback,
        "review_summary": im.review_summary,
        "implementation_notes": im.implementation_notes,
        "execution_results": va.execution_results,
        "measured_validation": va.measured_validation,
        "validation_report": va.validation_report,
        "discrepancies": va.discrepancies,
        "claims": [_dump(x) for x in ev.claims],
        "equations": [_dump(x) for x in ev.equations],
        "traceability_edges": [_dump(x) for x in ev.traceability_edges],
        "consistency_issues": [_dump(x) for x in ev.consistency_issues],
        "message_board": co.message_board,
        "cro_directives": co.cro_directives,
        "evaluations": co.evaluations,
        "revision_counts": co.revision_counts,
        "needs_revision": co.needs_revision,
        "readme": out.readme,
        "implementation_paper": out.implementation_paper,
        "output_dir": out.output_dir,
        "pdf_report_path": out.pdf_report_path,
        "final_verdict": out.final_verdict,
    }


def from_runtime_state(runtime: RuntimeResearchState) -> ResearchState:
    """Rebuild canonical domain state from an agent-runtime result."""
    from ..schemas.state import (
        AnalysisState, ArchitectureState, CommunicationState, EvidenceState,
        ImplementationState, OutputState, PaperCorpus, ResearchInput, ValidationState,
    )
    return {
        "research_input": ResearchInput(
            pdf_path=runtime.get("pdf_path", ""), paper_title=runtime.get("paper_title", ""),
            paper_abstract=runtime.get("paper_abstract", ""),
        ),
        "paper_corpus": PaperCorpus(
            raw_pages=runtime.get("raw_pages", []), full_paper_text=runtime.get("full_paper_text", ""),
            figures_summary=runtime.get("figures_summary", ""), tables_summary=runtime.get("tables_summary", ""),
            equations_summary=runtime.get("equations_summary", ""), page_notes=runtime.get("page_notes", ""),
            page_notes_list=runtime.get("page_notes_list", []),
        ),
        "analysis": AnalysisState(
            research_report=runtime.get("research_report", ""), cro_reading_notes=runtime.get("cro_reading_notes", ""),
            theoretical_analysis=runtime.get("theoretical_analysis", ""),
        ),
        "architecture": ArchitectureState(
            architecture_analysis=runtime.get("architecture_analysis", ""),
            implementation_plan=runtime.get("implementation_plan", ""),
            codebase_structure=runtime.get("codebase_structure", ""),
            file_manifest=runtime.get("file_manifest", []),
        ),
        "implementation": ImplementationState(
            code_modules=runtime.get("code_modules", {}), review_feedback=runtime.get("review_feedback", {}),
            review_summary=runtime.get("review_summary", ""), implementation_notes=runtime.get("implementation_notes", ""),
        ),
        "validation": ValidationState(
            execution_results=runtime.get("execution_results", ""), measured_validation=runtime.get("measured_validation") or {},
            validation_report=runtime.get("validation_report", ""), discrepancies=runtime.get("discrepancies", ""),
        ),
        "evidence": EvidenceState(
            claims=runtime.get("claims", []), equations=runtime.get("equations", []),
            traceability_edges=runtime.get("traceability_edges", []), consistency_issues=runtime.get("consistency_issues", []),
        ),
        "communication": CommunicationState(
            message_board=runtime.get("message_board", []), cro_directives=runtime.get("cro_directives", {}),
            evaluations=runtime.get("evaluations", {}), revision_counts=runtime.get("revision_counts", {}),
            needs_revision=runtime.get("needs_revision", []),
        ),
        "output": OutputState(
            readme=runtime.get("readme", ""), implementation_paper=runtime.get("implementation_paper", ""),
            output_dir=runtime.get("output_dir", ""), pdf_report_path=runtime.get("pdf_report_path", ""),
            final_verdict=runtime.get("final_verdict", ""),
        ),
    }


__all__ = ["RuntimeResearchState", "from_runtime_state", "to_runtime_state"]
