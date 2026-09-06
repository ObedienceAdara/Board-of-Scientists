# Testing strategy

Phase 2 establishes a deterministic test pyramid for the research pipeline. The suite is intentionally split by failure domain so cheap tests catch local regressions before graph-level or multi-module tests run.

## Test tree

```text
tests/
├── unit/
│   ├── test_claims.py
│   ├── test_equations.py
│   ├── test_llm_mock.py
│   ├── test_manifest.py
│   ├── test_path_sanitization.py
│   ├── test_report.py
│   ├── test_routing.py
│   ├── test_schemas.py
│   └── test_state.py
├── integration/
│   ├── test_agent_state.py
│   ├── test_generated_code_sandbox.py
│   ├── test_pdf_ingestion.py
│   └── test_state_to_report.py
├── graph/
│   └── test_transitions.py
├── security/
│   └── test_sandbox.py
└── fixtures/
    ├── __init__.py
    └── pdf_factory.py
```

## Deterministic boundary

The tests never require Groq, OpenRouter, OpenAI, Tavily, or LangSmith credentials. `tests/unit/test_llm_mock.py` replaces the LLM factory with a small runnable fake and still exercises the structured-output path used by the agents.

The graph tests go one level higher: they compile the actual LangGraph topology and replace node implementations with deterministic state transitions. This validates retry and advance edges without invoking an LLM.

## Covered behaviors

Unit tests cover path normalization and artifact persistence, routing decisions, domain-state initialization, Pydantic validation, claim normalization and evidence bounds, equation extraction, report generation, sandbox rejection policy, safe sandbox execution, manifest sanitization/batching, and the structured-output LLM boundary.

Integration tests cover PDF ingestion, agent/runtime-state projection, generated code execution through the sandbox, and conversion of domain state into persisted report artifacts.

Graph tests verify the normal path and the important repair loops: Analyst failure retries Analyst; Reviewer and Experiment failure route back to Engineer; Writer failure retries Writer.

## Running locally

```bash
pytest -q
```

Useful focused runs:

```bash
pytest -q tests/unit
pytest -q tests/integration
pytest -q tests/graph
pytest -q tests/security
```

The CI workflow runs the complete suite on every push to `main` and every pull request with provider API-key environment variables empty by design.

## Scope boundary

This phase does not attempt expensive live-provider acceptance tests, real multi-page LLM analysis, or full-paper end-to-end runs. Those belong to a higher-cost validation tier and should be added only after deterministic contracts are stable.
