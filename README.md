# Board of Scientists — AI Research Implementation Team

Board of Scientists is a multi-agent LangGraph system that takes a machine-learning research paper as input and drives it through paper analysis, theoretical interpretation, architecture design, iterative implementation, code review, execution-based validation, documentation, and a final research verdict.

The application is organized as a Python package under `board_of_scientists/`. The previous monolithic implementation has been moved into the active package modules; there is no archived implementation dependency.

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
│   └── tables.py           # Table discovery signals
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

## Workflow

```text
PDF
 │
 ▼
Paper Analyst
 │
 ▼
CRO independent reading
 │
 ▼
Theorist
 │
 ▼
Architect
 │
 ▼
CRO implementation plan
 │
 ▼
Engineer ───────────────┐
 │                      │
 ▼                      │
Reviewer ── fail ───────┘
 │
 ▼
Experiment Engineer ─ fail ──► Engineer
 │
 ▼
Writer ── fail ───────────────► Writer
 │
 ▼
CRO final verdict
 │
 ▼
Artifacts + implementation report
```

The Engineer consumes the Architect's structured `file_manifest`, implementing one file or small tightly coupled group per LLM pass. Dependencies are provided as context so generated modules can remain interface-compatible across passes. The Analyst processes long papers page-by-page and uses hierarchical reduction rather than the old hard 40-page cutoff. The Experiment Engineer executes generated code for real syntax, import, and best-effort model smoke checks; LLM commentary is explicitly distinguished from measured execution results.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/ObedienceAdara/Board-of-Scientists.git
cd Board-of-Scientists
python -m venv .venv
# activate the virtual environment
pip install -r requirements.txt
cp env.example .env
```

Configure `LLM_PROVIDER` and its corresponding API key in `.env`. Supported providers are `groq`, `openrouter`, and `openai`.

## Usage

CLI:

```bash
python main.py path/to/paper.pdf
```

REST API:

```bash
python main.py serve
```

Place a PDF in `UPLOADS_DIR` (default `./uploads`) and call `POST /implement-paper` with the filename.

## Configuration

The main controls remain:

| Variable | Default | Purpose |
|---|---:|---|
| `LLM_PROVIDER` | `groq` | Select LLM provider |
| `ANALYST_MAX_PAGES` | `300` | Safety valve for pathological PDFs |
| `ANALYST_REDUCE_FANOUT` | `6` | Notes merged per reduction call |
| `ENGINEER_MAX_BATCH_SIZE` | `4` | Maximum files in one Engineer pass |
| `TAVILY_API_KEY` | — | Web-search integration |
| `LANGCHAIN_TRACING_V2` | — | Optional LangSmith tracing |
| `API_AUTH_TOKEN` | — | Optional REST authentication |
| `UPLOADS_DIR` | `./uploads` | REST upload directory |

Per-agent model overrides are available through `CRO_MODEL`, `ANALYST_MODEL`, `THEORIST_MODEL`, `ARCHITECT_MODEL`, `ENGINEER_MODEL`, `REVIEWER_MODEL`, `EXPERIMENT_MODEL`, and `WRITER_MODEL`.

## Architectural migration status

The structural migration is complete for the `_legacy` dependency boundary. Agent, ingestion, execution, schema, and reporting responsibilities now resolve through active package modules. Temporary compatibility aliases exist only for historical absolute imports and point directly to those active modules.

## Current limitations

The system still does not perform a full training-based reproduction of a paper's reported metrics. Figure understanding is currently limited to extracted metadata/signals rather than full visual reasoning. Global cross-report scientific consistency is represented by the evidence subsystem but is not yet the complete contradiction/traceability engine. The sandbox is designed for semi-trusted generated code and is not a hard OS security boundary for adversarial payloads.

## Development

Run the architecture and regression tests with:

```bash
pytest
```

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.
