"""
schemas.py — Structured output contracts for every agent that used to rely on
regex or marker-text parsing.

THE BUG THIS FILE FIXES
------------------------
Previously, several agents wrote free-text output that downstream code tried
to parse back out with regex:
  - Engineer:  code was extracted with `#\\s*filename:\\s*(\\S+\\.py)\\s*\\n(.*?)`
               which does not account for the markdown code fences the prompt
               itself told the model to use — the trailing ``` almost always
               ends up captured as part of the "code" and gets written
               straight into the .py file, corrupting it with a syntax error.
  - Writer:    README vs IMPLEMENTATION_NOTES was split by searching for a
               "DOCUMENT 2" marker string in the raw text — fragile, and
               silently wrong if the model phrases the heading differently.
  - CRO eval:  JSON was extracted by stripping ```json fences with .replace()
               and passed to json.loads() inside a bare try/except.
  - Everyone:  inter-agent notes were extracted with a `[MESSAGE TO X]: ...`
               regex that only works if the model reproduces that exact
               literal tag.

THE FIX
-------
Every one of these is now a real, validated Pydantic schema passed to
`llm.with_structured_output(schema)` (tool-calling under the hood). The model
either returns something that satisfies the schema, or LangChain raises a
validation error we can catch and retry — there is no more "hope the regex
matches the exact text format we asked for."
"""

from typing import List
from pydantic import BaseModel, Field


class CodeFile(BaseModel):
    """A single implementation file."""
    filename: str = Field(
        description=(
            "Relative file path only, e.g. 'models/backbone.py' or "
            "'training/trainer.py'. Never an absolute path, never contains "
            "'..'. Use forward slashes for subdirectories."
        )
    )
    language: str = Field(default="python", description="Language of this file, e.g. 'python', 'markdown', 'yaml'.")
    description: str = Field(default="", description="One-sentence description of what this file implements.")
    code: str = Field(
        description=(
            "Complete, runnable file contents. Plain source code only — do "
            "NOT wrap this in markdown code fences (```), do not include the "
            "filename as a comment line, just the raw file contents."
        )
    )


class EngineerOutput(BaseModel):
    """Everything the Senior ML Engineer produces in one implementation pass."""
    files: List[CodeFile] = Field(description="Every code file produced or updated in this pass. At least one file.")
    implementation_notes: str = Field(
        default="", description="Summary of what was implemented, key design decisions, and any [TODO: ...] flags left for the CRO."
    )
    message_to_reviewer: str = Field(default="", description="Specific things to review and why. Empty string if nothing specific.")
    message_to_cro: str = Field(default="", description="Concerns, ambiguities, or plan deviations to flag to the CRO. Empty string if none.")


class WriterOutput(BaseModel):
    """The two documentation deliverables from the Technical Writer."""
    readme_md: str = Field(description="Full contents of README.md, in markdown.")
    implementation_notes_md: str = Field(description="Full contents of IMPLEMENTATION_NOTES.md, in markdown.")


class EvaluationResult(BaseModel):
    """The CRO's pass/fail evaluation of another agent's deliverable."""
    passed: bool = Field(description="True only if the deliverable meets a NeurIPS-review bar on all five criteria.")
    accuracy: str = Field(default="PASS", description="'PASS' or 'FAIL'")
    completeness: str = Field(default="PASS", description="'PASS' or 'FAIL'")
    depth: str = Field(default="PASS", description="'PASS' or 'FAIL'")
    correctness: str = Field(default="PASS", description="'PASS' or 'FAIL'")
    alignment: str = Field(default="PASS", description="'PASS' or 'FAIL'")
    critical_issues: List[str] = Field(default_factory=list, description="Specific issues that must be fixed. Empty list if passed.")
    feedback: str = Field(default="", description="Precise, technical, actionable feedback with section/line references. Empty string if passed.")


class TheoristOutput(BaseModel):
    """The Theorist's mathematical analysis."""
    analysis: str = Field(description="The full theoretical analysis, in markdown, covering every section requested in the prompt.")
    message_to_architect: str = Field(
        default="", description="The top mathematical constraints the ML Architect must honor in the codebase design."
    )


class ArchitectOutput(BaseModel):
    """The ML Architect's system design."""
    analysis: str = Field(description="The full architecture design, in markdown, covering every section requested in the prompt.")
    message_to_engineer: str = Field(
        default="", description="A clear briefing to the Engineer on the most critical implementation constraints and where to start."
    )


class ReviewerOutput(BaseModel):
    """The Code Reviewer's assessment of the implementation."""
    review: str = Field(description="The full code review, in markdown, covering every section requested in the prompt.")
    verdict: str = Field(default="REVISE", description="One of: 'APPROVED', 'REVISE', 'MAJOR REVISION REQUIRED'.")
    message_to_engineer: str = Field(default="", description="Specific, actionable feedback for the Engineer.")
    message_to_cro: str = Field(default="", description="Critical paper misalignments that need a CRO decision. Empty string if none.")


class ExperimentOutput(BaseModel):
    """The Experiment Engineer's validation write-up."""
    analysis: str = Field(
        description=(
            "Full validation write-up, in markdown. Any number or claim that "
            "is not copied directly from the MEASURED RESULTS block must be "
            "explicitly prefixed with '[LLM-INFERRED]' — never present an "
            "inferred or estimated number as if it were measured."
        )
    )
    message_to_cro: str = Field(default="", description="Summary of validation results and critical discrepancies.")
    message_to_engineer: str = Field(default="", description="Specific bugs to fix with reproduction steps. Empty string if none found.")
