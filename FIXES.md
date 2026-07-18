# FIXES.md — What changed and why

This document records a deep pass on five specific bugs, plus a small number
of directly-motivated fixes discovered and made along the way (called out
separately below, never silently). Every fix was verified — either by
executing the real code (the sandbox rewrite, path sanitization, provider
abstraction) or by running the full multi-agent graph end-to-end against a
mocked LLM/graph harness that exercises the actual control flow, message
routing, and structured-output attribute access. See "How this was tested"
at the end.

---

## 1. Message routing now uses stable agent keys, not display names

**Files:** `agent_registry.py` (new), `agents.py`, `state.py`

**The bug:** `post_message()` addressed recipients by human display name
(`"Senior ML Engineer"`), but `get_messages_for()` looked up the inbox by a
different string (`"engineer"`). `"ENGINEER" != "SENIOR ML ENGINEER"`, so the
lookup always returned nothing. The Architect's directives, the Reviewer's
follow-ups, and the Experiment Engineer's bug reports never reached the
Engineer — the entire "team communication" concept was broken by this one
mismatch, silently, with no error.

**The fix:** `agent_registry.py` defines one canonical, lowercase key per
agent (`CRO`, `ANALYST`, `THEORIST`, `ARCHITECT`, `ENGINEER`, `REVIEWER`,
`EXPERIMENT`, `WRITER`, `ALL`) plus a `normalize_key()` that also recognizes
legacy display names defensively. `post_message()` and `get_messages_for()`
both route exclusively through these keys now. `AgentMessage` in `state.py`
stores both the canonical key and a display name (for human-readable
logs/PDF output), so routing and prettiness can never drift apart again.

**Verified:** a standalone test posts messages from Architect, Reviewer, and
Experiment Engineer to the Engineer, then confirms `get_messages_for(state,
"engineer")` returns all three — previously this call always returned
`"No messages from team yet."`

---

## 2. CRO evaluation feedback now actually reaches the agent being asked to revise

**Files:** `agents.py`, `prompts.py`

**The bug:** `get_feedback(state, agent_name)` was defined but **never
called anywhere in the codebase**. When the CRO rejected an agent's output
and routed it back for revision, the agent re-ran with the *exact same
inputs* — it had no way to see what was wrong. The "up to 3 revision
rounds" safety net was really just re-rolling the same prompt on a low
temperature, not a correction loop.

**The fix:** `format_prior_feedback(state, agent_key)` renders
`state["evaluations"][agent_key]` (critical issues + detailed feedback) into
prompt-ready text, and every agent's prompt (`ANALYST_SYNTHESIS_PROMPT`,
`THEORIST_PROMPT`, `ARCHITECT_PROMPT`, `ENGINEER_PROMPT`, `REVIEWER_PROMPT`,
`EXPERIMENT_ENGINEER_PROMPT`, `WRITER_PROMPT`) now has a `{prior_feedback}`
slot wired to it in `agents.py`.

**Also fixed along the way:** the Reviewer's evaluation was reading
`state["review_feedback_str"]` — a key nothing ever set (the Reviewer wrote
to `"review_feedback"`, a dict, and to nothing called
`"review_feedback_str"`). The CRO was therefore *always* evaluating an empty
string for the Reviewer, regardless of what was actually reviewed.
`reviewer_agent` now also writes `state["review_summary"]` (the real review
text), and `main.py`'s `node_eval_reviewer` evaluates that field instead.

**Also fixed:** the Writer was never evaluated at all — a `route_writer()`
function existed in `main.py` but was never wired into the graph, unlike
every other agent. There's now a real `eval_writer` node with the same
revision loop as everyone else.

**Efficiency bonus:** on an Analyst revision, the original code re-ran the
entire ~9-LLM-call page-by-page reading process from scratch. The paper
hasn't changed between revisions — only the synthesis needs to improve — so
`analyst_agent` now persists `page_notes` in state and, on a detected
revision, skips straight to redoing the synthesis with the CRO's feedback
attached.

**Verified:** a standalone test renders feedback for a failed Engineer
evaluation and confirms the specific issues and detailed feedback text
appear in the output; a second test confirms a revision reuses persisted
page notes without calling `extract_pdf_pages()` again.

---

## 3. Validation now actually executes the generated code

**Files:** `tools.py` (`run_codebase_validation`, `format_measured_results`),
`agents.py`, `prompts.py`, `main.py`

**The bug:** `experiment_engineer_agent` ran a fixed, generic smoke test
(checked only whether `torch`/`numpy` import) and never touched the actual
code the Engineer wrote. Every "forward pass result," "discrepancy
percentage," and "gradient check" in the Experiment Engineer's report was
the LLM inventing plausible text based on a prompt — not a measurement —
even though it ended up in a polished PDF under a "FINAL VERDICT."

**The fix:** `tools.run_codebase_validation()` writes every generated file
into a real, importable project directory (auto-creating `__init__.py` so
intra-project imports like `from models.backbone import Encoder` resolve
correctly) and runs a harness **inside the hardened sandbox** that performs,
per `.py` file:
- a real `ast.parse` syntax check,
- a real `importlib.import_module` attempt with the actual traceback on
  failure,
- a best-effort instantiation + forward-pass smoke test for any
  `torch.nn.Module` subclasses found (only if torch is actually installed).

The result is a structured dict — `measured_validation` in state — that is
never touched by an LLM. `EXPERIMENT_ENGINEER_PROMPT` now requires the model
to explicitly prefix anything it didn't measure (results comparisons,
ablations, performance profiling — none of which can be produced without an
actual training run) with `[LLM-INFERRED]`, and the final PDF report gets a
dedicated "Appendix: Raw Measured Execution Results" section with the raw
JSON, separate from the Experiment Engineer's commentary.

**Verified directly** (no LLM needed for this one): a multi-file project
with an intra-project import, a real syntax error, and a file that tries
`os.system`/`subprocess.run` — confirmed the harness correctly reports which
files have valid syntax, which import successfully (including the
intra-project import), and that the dangerous-module file is genuinely
blocked (not just pattern-matched) with a real `ImportError`.

### A closely-related security fix, made because fix #3 required touching this code anyway

The sandbox's own docstring claimed 5 defense-in-depth layers, including
"restricted builtins" and Unix "resource limits via the `resource` module."
**Neither was real**: `_build_safe_builtins()` was defined but never called,
and `resource` was never imported anywhere. The actual protection was a
regex/blocklist scan of source text on an otherwise completely unrestricted
subprocess — a well-known-bypassable pattern (string concatenation, getattr
indirection, module aliasing all defeat it).

Since `run_codebase_validation()` needed a hardened execution path anyway,
`tools.py` now has:
- **Real resource limits** (POSIX): `resource.setrlimit()` via `preexec_fn`
  caps virtual memory, CPU time, and file descriptors — actually enforced by
  the kernel, not just claimed.
- **A real guarded `__import__`**: dangerous modules
  (`os`, `sys`, `subprocess`, `socket`, `ctypes`, `pickle`, etc.) are blocked
  by the interpreter's actual import mechanism, not by pattern-matching the
  source text beforehand — this catches the dynamic/obfuscated-import case
  the old regex approach couldn't (verified with a test that builds the
  module name via string concatenation specifically to bypass the regex,
  and confirms the guarded import still blocks it).
- An honest module docstring: this is explicitly documented as raising the
  bar against unsophisticated and accidental issues, backed by real OS-level
  limits — **not** a hard boundary against a fully adversarial payload (no
  pure-Python sandbox is). A production deployment accepting arbitrary
  untrusted PDFs should still sit this behind real OS-level isolation
  (containers with dropped capabilities, gVisor, Firecracker, nsjail).

Also fixed in the same area: `save_code_file()` used
`os.path.basename(filename)`, which silently discarded every subdirectory
the Architect designed (`models/backbone.py` was saved as flat
`backbone.py`). The shared `sanitize_relative_path()` now preserves safe
subdirectories while still rejecting traversal (`../`), absolute paths, and
null bytes — verified with both path-traversal and legitimate-subdirectory
test cases.

---

## 4. Structured output instead of regex/text parsing

**Files:** `schemas.py` (new), `agents.py`, `prompts.py`

**The bug:** several agents wrote free text that downstream code tried to
parse back out with regex, and none of the regexes actually matched what the
prompts told the model to produce:
- The Engineer's code was extracted with a regex expecting
  `# filename: x.py` immediately followed by code, but the prompt told the
  model to wrap it in a ` ```python ` fence — the trailing fence text almost
  always ended up captured as part of the "code" and written straight into
  the `.py` file, which is a guaranteed syntax error on the next import.
- The Writer's README vs. IMPLEMENTATION_NOTES split searched for a
  `"DOCUMENT 2"` marker string — silently wrong the moment the model phrased
  the heading differently.
- The CRO's evaluation JSON was extracted via `.replace("```json", "")`
  string surgery into `json.loads()`, wrapped in a bare `try/except` that
  defaulted to `passed: True` on any parse failure.
- Every inter-agent note relied on a literal `[MESSAGE TO X]:` tag the model
  had to reproduce exactly.

**The fix:** `schemas.py` defines a Pydantic model per structured deliverable
(`EngineerOutput`, `WriterOutput`, `EvaluationResult`, `TheoristOutput`,
`ArchitectOutput`, `ReviewerOutput`, `ExperimentOutput`), and
`agents.run_structured_chain()` uses `llm.with_structured_output(schema)`
(tool-calling under the hood) instead of parsing prose. If structured output
fails validation, it retries once, then raises `StructuredOutputError`,
which every call site catches and degrades gracefully (a clearly-labeled
fallback that the CRO's evaluation loop — now that it actually delivers
feedback, see fix #2 — will catch and ask to be redone) rather than crashing
the whole graph run. `extract_inter_agent_message()` and `clean_json()` —
the two regex/string-surgery helpers this replaces — are removed as dead
code.

Per-provider note: `ChatOpenAI` with a manually-set `base_url` (the previous
approach for Groq) is specifically documented upstream as having "broken
structured output" among other issues — this is part of why fix #5 moves to
dedicated per-provider packages (`langchain-groq`, `langchain-openrouter`)
rather than the base-url workaround.

**Verified:** the full pipeline run (see "How this was tested") exercises
`with_structured_output(...)` end-to-end, including a nested
`List[CodeFile]` field, and confirms attribute access (`f.filename`,
`f.code`, `result.message_to_reviewer`, etc.) works with no regex involved.
A static cross-check (via Python's `ast` module) also confirms every
`{placeholder}` declared in every prompt template is actually supplied by
its corresponding call site's input dict, and vice versa — this would have
caught the missing/renamed fields introduced by this rewrite.

---

## 5. Dependencies pinned; Groq/OpenRouter mismatch reconciled

**Files:** `llm_provider.py` (new), `agents.py`, `requirements.txt`,
`env.example`, `main.py`

**The bug:** `make_llm()` hardcoded `ChatOpenAI` pointed at Groq's endpoint
and `GROQ_API_KEY`, with `default_headers` (`HTTP-Referer`/`X-Title`) that
only mean anything on OpenRouter. Meanwhile `env.example` and `main.py`'s own
docstring told users to override models with OpenRouter-style IDs like
`anthropic/claude-3.5-sonnet` — which Groq's API would reject outright.
Following the project's own setup instructions broke it. `requirements.txt`
also had zero version pins and listed `langchain`, `langchain-community`,
and `langserve` — none of which are actually imported anywhere — while
missing `langchain-groq`, `langchain-openrouter`, and `pydantic`, which are.

**The fix:** `llm_provider.py` adds an explicit `LLM_PROVIDER` env var
(`groq` | `openrouter` | `openai`) selecting one of three real, first-party
LangChain integrations, each with its own correct default model and API key
variable:
- `groq` → `langchain_groq.ChatGroq` / `GROQ_API_KEY` /
  `llama-3.3-70b-versatile`
- `openrouter` → `langchain_openrouter.ChatOpenRouter` /
  `OPENROUTER_API_KEY` / `meta-llama/llama-3.3-70b-instruct`
- `openai` → `langchain_openai.ChatOpenAI` / `OPENAI_API_KEY` / `gpt-4o-mini`

If the selected provider's API key is missing, `make_llm()` fails
immediately with a message naming the exact env var to set, instead of a
cryptic 401 three network calls deep into a graph run. `requirements.txt` is
now pinned to versions verified against PyPI, with the unused packages
removed and the actually-imported ones added. `env.example` documents all
three providers with real, valid model ID examples for each.

**Also fixed in the same area:** `main.py`'s REST API used LangServe's
`add_routes()`, which is (a) deprecated upstream in favor of LangGraph
Platform, (b) had a published vulnerability in its playground endpoint
allowing arbitrary file reads on the server (versions 0.0.13–0.0.15), and
(c) accepted a raw `pdf_path` string with no validation at all — anyone who
could reach the endpoint could ask the server to open any file it could
read. Since `langserve` had to be dropped from `requirements.txt` as part of
reconciling the dependency list, the endpoint is now a plain FastAPI route
that resolves `pdf_filename` inside a configured `UPLOADS_DIR` only (never a
raw path), with an optional shared-secret `X-API-Key` header
(`API_AUTH_TOKEN`) — open by default for local dev, with a startup warning
if left unset.

**Verified:** a standalone test confirms Groq without `GROQ_API_KEY` fails
fast with a clear message; setting the key selects a real Groq model ID;
switching `LLM_PROVIDER=openrouter` (with its own key) selects a real,
namespaced OpenRouter model ID instead — confirming the two providers can no
longer get their model catalogues crossed.

---

## How this was tested

Real dependencies (LangChain, LangGraph, Groq/OpenRouter clients, FastAPI,
Pydantic) aren't installable in the environment this fix was written in (no
network access). Rather than skip verification, three layers were used:

1. **Directly executable code** — the sandbox rewrite (`tools.py`), path
   sanitization, and the provider abstraction (`llm_provider.py`) have no
   external dependencies beyond the standard library, so these were tested
   by actually running them: real syntax errors, real blocked imports, real
   intra-project imports, real timeouts, real path-traversal attempts.

2. **A mocked LangChain/LangGraph/FastAPI harness** that structurally
   mirrors the real APIs (`ChatPromptTemplate.from_template(...).invoke()`
   raises on a missing template variable exactly like the real one;
   `StateGraph` actually walks nodes/edges/conditional-routing instead of
   being a no-op; `with_structured_output(schema)` returns a populated
   instance of the real Pydantic schema, including nested `List[Model]`
   fields) — used to run the **entire 8-agent graph end-to-end**, including
   every conditional edge, the newly-added `eval_writer` node, and PDF
   report generation, with no crashes.

3. **Targeted unit tests** against the real `agents.py`/`tools.py` for the
   specific bugs described above (message routing, feedback rendering,
   revision-skip-reread, provider selection).

This gives strong confidence in the control flow, data wiring, and
structural correctness. It does **not** replace testing against the real
Groq/OpenRouter/OpenAI APIs and a real paper — please run
`python main.py path/to/paper.pdf` with a real API key as a first check
after installing, and watch for any `StructuredOutputError` messages in the
console, which indicate the model struggled with structured output on that
particular call (open-weight models are less reliable at this than
GPT/Claude; if this happens often, consider `LLM_PROVIDER=openrouter` or
`openai` for the affected agent).

## Known limitations not addressed in this pass

These were flagged in the original audit but are out of scope for the five
fixes above — noted here for transparency, not fixed:

- The Engineer still generates the entire codebase in a single completion
  rather than iterating file-by-file; for a large/complex paper this may
  still hit the model's output length before finishing everything the
  Architect specified.
- The Analyst still hard-caps analysis at the first 40 pages of a paper.
- No automated test suite, CI, or LICENSE file exists for this project
  itself.
