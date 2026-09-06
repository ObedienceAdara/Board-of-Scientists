# Board of Scientists — AI Research Implementation Team

Board of Scientists is a multi-agent LangGraph system that takes a machine-learning research paper as input and drives it through paper analysis, theoretical interpretation, architecture design, iterative implementation, code review, execution-based validation, documentation, and a final research verdict.

The application is organized as a Python package under `board_of_scientists/`. The implementation has been moved into the active package modules; the `_legacy` dependency boundary has been eliminated.

## Current architecture

```text
board_of_scientists/
│
├── graph/
│   ├── workflow.py        # LangGraph topology, runner, REST API
│   ├── routers.py         # Conditional routing policy
│   └── nodes.py           # Graph-facing node adapters
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
│   ├── equations.py       # Equation extraction boundary
│   └── tables.py          # Table discovery signals
│
├── schemas/
│   ├── agents.py          # Agent output contracts
│   ├── evidence.py        # Evidence contracts
│   └── state.py           # Shared LangGraph state contract
│
├── reports/
│   ├── pdf.py             # PDF report generation
│   └── provenance.py      # Artifact/message persistence
│
└── tests/                 # Architecture and regression tests
```

A root `main.py` remains a thin compatibility launcher so the established CLI commands continue to work.

## Architectural migration status

The Phase 0 `_legacy` migration is complete. Active package modules no longer import from `board_of_scientists._legacy`, and the archived implementation has been removed. Historical absolute-import compatibility is handled at the package boundary by mapping those names to active package modules.

## Current limitations

The system still does not perform a full training-based reproduction of a paper's reported metrics. Figure understanding is currently limited to extracted metadata/signals rather than full visual reasoning. Global cross-report scientific consistency is represented by the evidence subsystem but is not yet the complete contradiction/traceability engine. The sandbox is designed for semi-trusted generated code and is not a hard OS security boundary for adversarial payloads.

## Development

Run the architecture and regression tests with:

```bash
pytest
```

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.
