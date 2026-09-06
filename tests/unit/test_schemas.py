import pytest
from pydantic import ValidationError

from board_of_scientists.schemas.agents import (
    ArchitectOutput,
    CodeFile,
    EvaluationResult,
    ExperimentOutput,
    FileSpec,
    ReviewerOutput,
    TheoristOutput,
    WriterOutput,
)


def test_evaluation_result_rejects_missing_required_pass_flag():
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate({"accuracy": "PASS"})


def test_code_file_requires_filename_and_code():
    with pytest.raises(ValidationError):
        CodeFile.model_validate({"filename": "model.py"})


def test_file_spec_defaults_are_safe():
    spec = FileSpec(filename="models/model.py", description="model")
    assert spec.depends_on == []
    assert spec.group == ""


def test_agent_output_models_round_trip():
    evaluator = EvaluationResult(passed=True)
    assert evaluator.passed is True
    architect = ArchitectOutput(analysis="design", file_manifest=[FileSpec(filename="model.py", description="model")])
    assert architect.file_manifest[0].filename == "model.py"
    assert TheoristOutput(analysis="analysis").analysis == "analysis"
    assert ReviewerOutput(review="review").verdict == "REVISE"
    assert ExperimentOutput(analysis="validation").analysis == "validation"
    assert WriterOutput(readme_md="# README", implementation_notes_md="notes").readme_md.startswith("#")
