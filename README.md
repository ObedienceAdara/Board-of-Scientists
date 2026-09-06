# Board of Scientists — AI Research Implementation Team

Board of Scientists is a multi-agent LangGraph system that takes a machine-learning research paper as input and drives it through paper analysis, theoretical interpretation, architecture design, iterative implementation, code review, execution-based validation, documentation, and a final research verdict.

The application is organized as a Python package under `board_of_scientists/` with explicit package boundaries and a composed domain state model.

## Current architecture

```text
board_of_scientists/
│
├── graph/
│   ├── workflow.py        # LangGraph topology + runner + REST API
│   ├── routers.py         # Conditional routing policy
│   └── nodes.py           # Domain-state graph adapters
│
├── agents/
│   ├── analyst.py         # Paper analysis
│   ├── theorist.py        # Mathematical/theoretical analysis
│   ├── architect.py       # System design + file manifest
│   ├── engineer.py        # Iterative per-file implementation
│   ├── reviewer.py        # Code/paper review
│   ├── experiment.py      # Execution-based validation
│   ├── writer.py          # README + implementation documentation
│   └── cro.py             # Oversight, evaluation, planning, final verdict
│
├── evidence/
│   ├── claims.py          # Claim/evidence primitives
│   ├── equations.py       # Equation evidence primitives
│   ├── traceability.py    # Source-to-implementation trace edges
│   └── consistency.py     # Deterministic consistency checks / global engine seam
│
├── execution/
│   ├── sandbox.py         # Sandboxed generated-code execution
│   ├── environment.py     # Execution environment helpers
│   └── experiments.py     # Validation/experiment execution boundary
│
├── ingestion/
│   ├── pdf.py             # PDF/page extraction
│   ├── figures.py         # Figure discovery signals
│   ├── equations.py       # Equation extraction
│   └── tables.py          # Table discovery signals
│
├── schemas/
│   ├── agents.py          # Agent output contracts
│   ├── evidence.py        # Evidence contracts
│   └── state.py           # Domain models + LangGraph state envelope
│
├── reports/
│   ├── pdf.py             # PDF report generation
│   └── provenance.py      # Artifact/message persistence
│
└── tests/                 # Unit, integration, graph, security, ingestion and evidence tests
```

A root `main.py` remains a thin compatibility launcher so the established CLI commands continue to work.

## Architectural migration status

Phase 0 removed the archived implementation dependency. Phase 1 now defines and tests the package dependency DAG, and LangGraph carries a composed `ResearchState` rather than a flat bag of unrelated fields.

Phase 2 establishes the automated testing foundation: deterministic unit tests, integration tests across subsystem seams, compiled-graph transition tests, security tests for the sandbox, ingestion/evidence coverage, reusable fixtures, and credential-free CI.

The allowed application dependency direction is:

```text
graph    → agents, schemas, reports
agents   → schemas, evidence, ingestion, execution
reports  → schemas, evidence
evidence → schemas
ingestion → schemas
execution → schemas
schemas  → application packages
```

Forbidden reverse or lateral dependencies are enforced by `tests/test_architecture.py`.

## Domain state

The graph state is composed from explicit bounded contexts:

```text
ResearchState
├── ResearchInput
├── PaperCorpus
├── AnalysisState
├── ArchitectureState
├── ImplementationState
├── ValidationState
├── EvidenceState
├── CommunicationState
└── OutputState
```

`graph/nodes.py` owns the only projection between this canonical domain model and the current behavior-preserving flat agent runtime. This keeps the LangGraph contract stable while individual agents can be migrated to domain-native inputs without another global state rewrite.

## Testing foundation

The test pyramid lives under `tests/`:

```text
tests/
├── unit/         # fast local contracts and pure functions
├── integration/  # ingestion/state/sandbox/report subsystem seams
├── graph/        # compiled LangGraph transitions and retry behavior
├── security/     # sandbox policy and defensive controls
├── ingestion/    # real PDF fixture parsing
├── evidence/     # evidence-domain regression coverage
└── fixtures/     # deterministic test-data factories
```

The suite does not require an API key. The structured-output path has a fake LLM test, and graph execution tests replace node behavior with deterministic state transitions while compiling the real workflow topology. GitHub Actions runs `pytest -q` on pushes to `main` and on pull requests with provider credentials empty by design.

See `docs/TESTING.md` for the detailed testing strategy and scope boundaries.

## Current limitations

The system still does not perform a full training-based reproduction of a paper's reported metrics. Figure understanding is currently limited to extracted metadata/signals rather than full visual reasoning. Global cross-report scientific consistency is represented by the evidence subsystem but is not yet the complete contradiction/traceability engine. The sandbox is designed for semi-trusted generated code and is not a hard OS security boundary for adversarial payloads.

## Development

Run the complete deterministic test suite with:

```bash
pytest -q
```

Targeted suites can be run independently, for example `pytest -q tests/unit`, `pytest -q tests/integration`, `pytest -q tests/graph`, or `pytest -q tests/security`.

The architecture suite checks the package import graph, verifies the intended directory structure, validates the composed domain-state shape, and tests round-trip conversion between canonical domain state and the current agent runtime representation.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.