# Board of Scientists — AI Research Implementation Team

A multi-agent LangGraph pipeline that reads a machine learning research paper
(PDF) and produces a working code implementation, a real (not simulated)
validation pass, documentation, and a PDF report — driven by eight
specialized LLM agents that critique and revise each other's work under a
Chief Research Officer's evaluation loop.

```
python main.py path/to/paper.pdf
```

produces a directory of generated code, a `team_communications.json` log of
every inter-agent message, and an `implementation_report.pdf` covering the
theoretical analysis, architecture design, code review, and measured
validation results.

---

## How it works

```
 PDF ──▶ Analyst ──▶ CRO (reads independently)
                        │
                        ▼
                    Theorist ──▶ Architect ──▶ CRO (implementation plan)
                                                    │
                                                    ▼
                                     Engineer ◀──────────────┐
                                        │                    │
                                        ▼                    │
                                    Reviewer ─────(fail)──────┘
                                        │
                                      (pass)
                                        ▼
                              Experiment Engineer ───(fail)──▶ back to Engineer
                                        │
                                      (pass)
                                        ▼
                                     Writer ──▶ CRO final verdict ──▶ output
```

Every agent's output is evaluated by the CRO against a five-criteria rubric
(accuracy, completeness, depth, correctness, alignment). A failed evaluation
routes back to the same agent with the CRO's specific feedback attached, up
to 3 revision attempts before the pipeline force-accepts and moves on. See
[Architecture](#architecture) for what each phase actually does.

### The team

| Agent | Role | Model config |
|---|---|---|
| Dr. Aria Chen | Chief Research Officer — reads the paper independently, writes the master implementation plan, evaluates every other agent's output, issues the final verdict | `CRO_MODEL` |
| Dr. Marcus Webb | Paper Analyst — reads the PDF page-by-page, produces a structured synthesis | `ANALYST_MODEL` |
| Prof. Elena Vasquez | Theorist — deep mathematical/theoretical breakdown | `THEORIST_MODEL` |
| Dr. James Okafor | ML Architect — designs the codebase structure and file manifest | `ARCHITECT_MODEL` |
| Dr. Kai Nakamura | Senior ML Engineer — implements the code, one file (or small group) at a time | `ENGINEER_MODEL` |
| Dr. Priya Sharma | Code Reviewer — reviews the implementation against the paper | `REVIEWER_MODEL` |
| Dr. Santiago Reyes | Experiment Engineer — actually executes the generated code in a sandbox and reports measured results | `EXPERIMENT_MODEL` |
| Dr. Amara Osei | Technical Writer — produces README.md and IMPLEMENTATION_NOTES.md for the generated project | `WRITER_MODEL` |

---

## Architecture

### 1. Ingestion & analysis
The Analyst extracts text page-by-page via PyMuPDF and produces per-page
notes with no length cap — for long papers, notes are recursively
consolidated via hierarchical map-reduce (`ANALYST_REDUCE_FANOUT`) rather
than truncated, so per-call context size stays constant regardless of paper
length. The CRO separately reads the full paper and forms its own
understanding, used later as an independent reference point when evaluating
everyone else.

### 2. Theory & design
The Theorist produces a full mathematical breakdown; the Architect designs
the codebase structure and, critically, a **structured file manifest**
(`file_manifest`) — every file to be implemented, in dependency order, with
a specific description and its dependencies on other files in the project.

### 3. Implementation
The Engineer iterates the file manifest **one file (or small, tightly-coupled
group) per LLM call**, in dependency order — not one call for the entire
codebase. Each pass receives the full code of whatever it depends on (so
signatures actually match) and a lightweight summary of the rest of the
project. The Code Reviewer then reviews the aggregate result; a failing
review sends it back to the Engineer with specific feedback attached.

### 4. Validation
The Experiment Engineer **actually executes the generated code** in a
sandboxed subprocess: real syntax checks, real import attempts with real
tracebacks on failure, and a best-effort instantiation/forward-pass smoke
test for any `torch.nn.Module` subclasses found. Everything the LLM reports
beyond these measured results (e.g. estimated comparison to the paper's
reported numbers, since no training run happens) must be explicitly tagged
`[LLM-INFERRED]` in its write-up — the two are never presented as the same
kind of claim.

### 5. Documentation & verdict
The Writer produces `README.md` and `IMPLEMENTATION_NOTES.md` for the
generated project. The CRO issues a final verdict (COMPLETE /
SUBSTANTIALLY COMPLETE / INCOMPLETE), and everything is written to disk
alongside a PDF report.

### Inter-agent communication
Agents post to and read from a shared message board, keyed by a stable
canonical identifier per agent (not a display name), so a message addressed
to the Engineer reliably reaches the Engineer regardless of which agent sent
it. The full log is saved as `team_communications.json` in the output
directory.

---

## Project layout

| File | Purpose |
|---|---|
| `main.py` | LangGraph graph definition, CLI entry point, REST API (FastAPI) |
| `agents.py` | All 8 agent implementations, message board, LLM chain helpers |
| `agent_registry.py` | Canonical agent identity keys ↔ display names |
| `schemas.py` | Pydantic schemas for structured LLM output (no regex parsing of model output) |
| `prompts.py` | Every agent's prompt template |
| `state.py` | `ResearchState` — the shared LangGraph state schema |
| `tools.py` | PDF extraction, sandboxed code execution/validation, file output, PDF report generation, web search |
| `llm_provider.py` | Provider-aware LLM factory (Groq / OpenRouter / OpenAI) |
| `requirements.txt` | Pinned dependencies |
| `env.example` | All configuration variables, documented |
| `FIXES.md` | Detailed record of bug fixes made to this codebase |
| `FEATURES.md` | Detailed record of feature additions made to this codebase |

---

## Installation

Requires Python 3.11+.

```bash
git clone <this-repo>
cd Board-of-Scientists
pip install -r requirements.txt
cp env.example .env
```

Edit `.env`: set `LLM_PROVIDER` to one of `groq`, `openrouter`, or `openai`,
and set the matching API key. See [Configuration](#configuration) below for
everything else.

---

## Usage

### CLI

```bash
python main.py path/to/paper.pdf
```

Produces `output_<paper_title>_<timestamp>/` containing:
- every generated code file (subdirectory structure preserved, e.g. `models/backbone.py`)
- `README.md` and `IMPLEMENTATION_NOTES.md`
- `team_communications.json` — the full inter-agent message log
- `implementation_report.pdf` — theoretical analysis, architecture design,
  code review, validation report (with a dedicated appendix of the raw
  measured execution results, separate from the LLM's commentary), and the
  CRO's final verdict

### REST API

```bash
python main.py serve
```

Starts a FastAPI server on `:8000`. Place a PDF in the configured
`UPLOADS_DIR` (default `./uploads`) and:

```bash
curl -X POST http://localhost:8000/implement-paper \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_AUTH_TOKEN" \
  -d '{"pdf_filename": "paper.pdf"}'
```

`pdf_filename` is resolved **inside** `UPLOADS_DIR` only — never treated as
an arbitrary filesystem path. `X-API-Key` is required only if
`API_AUTH_TOKEN` is set; the server logs a warning on startup if it's left
unset (fine for local use, not for anything reachable beyond localhost).
Interactive API docs at `/docs`.

---

## Configuration

All variables below go in `.env` (see `env.example` for the authoritative,
fully-commented copy).

**LLM provider** — pick one:

| Variable | Used when |
|---|---|
| `LLM_PROVIDER` | `groq` (default) \| `openrouter` \| `openai` |
| `GROQ_API_KEY` | `LLM_PROVIDER=groq` |
| `OPENROUTER_API_KEY` | `LLM_PROVIDER=openrouter` |
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` |

**Per-agent model overrides** (optional — defaults to the selected
provider's default model): `CRO_MODEL`, `ANALYST_MODEL`, `THEORIST_MODEL`,
`ARCHITECT_MODEL`, `ENGINEER_MODEL`, `REVIEWER_MODEL`, `EXPERIMENT_MODEL`,
`WRITER_MODEL`.

**Analyst / Engineer tuning:**

| Variable | Default | Meaning |
|---|---|---|
| `ANALYST_MAX_PAGES` | 300 | Safety valve against a pathologically long PDF — not a quality cap; loudly logged if hit |
| `ANALYST_REDUCE_FANOUT` | 6 | Notes merged per hierarchical-reduce call |
| `ENGINEER_MAX_BATCH_SIZE` | 4 | Max files per Engineer implementation pass |

**Web search:** `TAVILY_API_KEY`

**LangSmith tracing (optional):** `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`

**REST API:** `API_AUTH_TOKEN` (shared-secret header; unauthenticated if unset), `UPLOADS_DIR`

---

## Security notes

The sandbox used for code validation (`tools.py`) layers several real
defenses — regex-based static pre-filtering, a guarded `__import__` that
blocks dangerous modules (`os`, `subprocess`, `socket`, `ctypes`, `pickle`,
etc.) at the interpreter level rather than by text pattern-matching, OS
resource limits (`resource.setrlimit`) capping memory/CPU/file descriptors,
subprocess timeouts, and an ephemeral, isolated temp directory. This is
**not a hard security boundary against a fully adversarial payload** — no
pure-Python sandbox is, and no attempt is made at OS-level isolation
(containers, gVisor, Firecracker). It's appropriate for validating code this
same pipeline generated. If you plan to run this against untrusted,
adversarial input at scale, put a real container/VM boundary around it
first.

The REST API endpoint restricts file access to a configured uploads
directory and supports an optional shared-secret header, but has no
rate-limiting or user-level auth — treat it as a single-tenant tool, not a
multi-tenant service, unless you add that yourself.

---

## Known limitations

- No automated test suite, CI, or LICENSE file exists for this project
  itself.
- Engineer revisions currently redo the entire file manifest, not just the
  file(s) the CRO flagged, since feedback is attributed to the aggregate
  implementation rather than a specific file.
- The Experiment Engineer's validation is real but limited to import/
  instantiate/forward-pass checks — it does not run an actual training loop,
  so it cannot confirm the paper's reported metrics are reproduced.
- Figures and diagrams in the source PDF are not currently fed to the
  agents — only extracted text.

See `FIXES.md` and `FEATURES.md` for the detailed history of what's been
addressed so far.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Fill in the
copyright holder in `LICENSE` and `NOTICE` before distributing.
