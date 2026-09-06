import pytest
from pydantic import ValidationError

from board_of_scientists.schemas.state import (
    AnalysisState,
    ArchitectureState,
    CommunicationState,
    EvidenceState,
    ImplementationState,
    OutputState,
    PaperCorpus,
    ResearchInput,
    ValidationState,
    create_initial_state,
)


def test_initial_state_contains_all_bounded_contexts():
    state = create_initial_state("paper.pdf")
    assert state["research_input"].pdf_path == "paper.pdf"
    assert isinstance(state["paper_corpus"], PaperCorpus)
    assert isinstance(state["analysis"], AnalysisState)
    assert isinstance(state["architecture"], ArchitectureState)
    assert isinstance(state["implementation"], ImplementationState)
    assert isinstance(state["validation"], ValidationState)
    assert isinstance(state["evidence"], EvidenceState)
    assert isinstance(state["communication"], CommunicationState)
    assert isinstance(state["output"], OutputState)


def test_state_defaults_are_independent_instances():
    first = create_initial_state("a.pdf")
    second = create_initial_state("b.pdf")
    first["communication"].message_board.append({"content": "only first"})
    first["implementation"].code_modules["a.py"] = {
        "filename": "a.py", "language": "python", "code": "", "description": "", "status": "draft"
    }
    assert second["communication"].message_board == []
    assert second["implementation"].code_modules == {}


def test_domain_models_validate_core_constraints():
    with pytest.raises(ValidationError):
        from board_of_scientists.schemas.agents import EvaluationResult

        EvaluationResult(passed=True, accuracy=object())

    with pytest.raises(ValidationError):
        from board_of_scientists.schemas.agents import Claim  # type: ignore[attr-defined]

        Claim(text="bad")


def test_research_input_accepts_minimal_valid_input():
    model = ResearchInput(pdf_path="paper.pdf")
    assert model.model_dump() == {
        "pdf_path": "paper.pdf",
        "paper_title": "",
        "paper_abstract": "",
    }
