# Board of Scientists architecture

## Dependency policy

The application is divided into seven package-level bounded contexts. The dependency graph is intentionally one-directional:

```text
graph
 ├── agents
 ├── schemas
 └── reports

agents
 ├── schemas
 ├── evidence
 ├── ingestion
 └── execution

evidence
 └── schemas

ingestion
 └── schemas

execution
 └── schemas

reports
 ├── schemas
 └── evidence

schemas
 └── nothing in the application layer
```

A package may import its own submodules. It must not import an application package outside its allow-list. In particular:

- `schemas` must never import `agents`, `graph`, `reports`, `execution`, `ingestion`, or `evidence`.
- `evidence` must never import `agents` or `graph`.
- `ingestion` must never import `graph`, `agents`, `execution`, or `reports`.
- `execution` must never import `graph`, `agents`, `ingestion`, or `reports`.
- `reports` must never import `graph`, `agents`, or `ingestion`.

`tests/test_architecture.py` parses the Python import graph and fails when these boundaries are violated. This is deliberately a source-level check rather than a runtime-only test so that architectural regressions are caught before optional dependencies or LLM configuration are involved.

## Domain state

LangGraph still carries one state envelope, but the envelope is composed of explicit domain objects:

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

The models live in `board_of_scientists.schemas.state` and own the conceptual data boundaries. The current agent runtime still accepts a flat dictionary for behavior compatibility; `graph.nodes` is the only place that projects `ResearchState` into that runtime shape and reconstructs the domain objects afterward.

This projection is intentionally temporary and localized. Future agent migrations can consume the domain models directly without changing the graph contract.

## Design rule

Graph orchestration owns control flow. Agents own research behavior. Evidence owns scientific provenance. Ingestion owns source parsing. Execution owns code execution. Reports own persistence and presentation. Schemas own shared data contracts.
