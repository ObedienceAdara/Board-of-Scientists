# FEATURES.md — Iterative code generation + unbounded-length paper analysis

Two follow-up improvements, both aimed at the biggest remaining correctness
gap identified after the first round of fixes (see `FIXES.md`): the Engineer
writing an entire codebase in one completion, and the Analyst silently
refusing to read past page 40.

---

## 1. Iterative per-file code generation

**Files:** `schemas.py` (`FileSpec`, `ArchitectOutput.file_manifest`),
`agents.py` (`architect_agent`, `_sanitize_manifest`, `engineer_agent`,
`_build_batches`, `_build_dependency_context`, `_build_manifest_context`),
`prompts.py` (`ARCHITECT_PROMPT`, `ENGINEER_PROMPT`), `state.py`
(`file_manifest`)

**The problem:** the Engineer was asked to write the entire codebase — every
file the Architect designed — in a single LLM completion
(`"current_file": "ALL FILES"`). For any paper with a nontrivial
implementation (8-10 files is normal), this reliably runs out of output
length before finishing, which is exactly why the old code needed so much
defensive regex-parsing machinery to cope with partial/malformed results
(now removed — see `FIXES.md` fix #4).

**The fix, in three parts:**

1. **The Architect now produces a real file manifest**, not just prose. A
   new `FileSpec` schema (`filename`, `description`, `depends_on`, `group`)
   is added to `ArchitectOutput.file_manifest`, populated in dependency
   order (files with no dependencies first). `_sanitize_manifest()` then
   deduplicates, runs every filename through the same path-sanitizer used
   for on-disk output and sandbox validation (so a manifest entry can never
   request a path-traversal write), and drops `depends_on` references to
   filenames that aren't actually in the manifest rather than trusting them
   blindly.

2. **The Engineer iterates the manifest one file (or small group) at a
   time**, in dependency order, instead of one call for everything.
   `_build_batches()` groups consecutive manifest entries that share the
   same Architect-assigned `group` label (intended only for small,
   tightly-coupled files like a config + constants module) into one pass,
   capped at `ENGINEER_MAX_BATCH_SIZE` (default 4) files even within a
   group — a `group` label can't silently recreate the "write everything at
   once" problem. Everything else is implemented on its own, one file per
   call, with that file's full output budget instead of it being split
   across the whole project.

3. **Each pass sees what it actually needs to stay consistent.**
   `_build_dependency_context()` includes the *full code* of every
   already-implemented file this batch's entries declared a dependency on
   (so the Engineer matches exact class/function signatures instead of
   guessing), while `_build_manifest_context()` gives brief
   `filename: description [done/PENDING]` situational awareness for
   everything else in the project — full code only where it's actually
   needed, not the whole growing codebase pasted into every prompt.

If the Architect's structured output fails (or a manifest is empty for any
other reason), `engineer_agent` falls back to the original single-pass
"ALL FILES" behavior rather than producing nothing — this preserves the
system's prior (if limited) behavior as a safety net instead of introducing
a new failure mode.

**On revision:** if the CRO rejects the Engineer's aggregate output, the
whole manifest is re-implemented (all passes run again) with the new CRO
feedback attached to each. This costs more LLM calls per revision than the
old single-call design, but a real per-file pass is far more likely to
produce complete, correctly-sized files in the first place, and per-file
partial-revision (only re-doing the flagged file) is a reasonable future
optimization noted but not implemented here.

**Verified:**
- `_sanitize_manifest`: a hand-crafted manifest with a path-traversal
  attempt, a duplicate filename, and a `depends_on` reference to a
  nonexistent file — confirms all three are handled correctly (dropped,
  deduplicated, and filtered, respectively).
- `_build_batches`: a mixed manifest (one `group`-tagged pair + three
  standalone files) batches exactly as expected; a synthetic 10-file group
  is correctly capped into `ENGINEER_MAX_BATCH_SIZE`-sized batches.
- `_build_dependency_context` / `_build_manifest_context`: confirmed real
  code is included for declared dependencies (with an honest
  "NOT YET IMPLEMENTED" flag for a dependency that isn't ready — shouldn't
  happen with a correctly-ordered manifest, but handled rather than
  crashing), and the manifest context correctly marks files
  done/PENDING while excluding the current batch's own files.
- A full two-file engineer run (`models/backbone.py` →
  `training/trainer.py`, the latter depending on the former) confirms the
  *actual* dependency code is what gets sent to the second pass's prompt —
  the core value of the feature — by inspecting the real inputs passed to
  `run_structured_chain` for each pass.
- The no-manifest fallback path still produces output.

---

## 2. Removed the 40-page Analyst cap via hierarchical map-reduce

**Files:** `agents.py` (`_hierarchical_reduce`, `analyst_agent`),
`prompts.py` (`ANALYST_REDUCE_PROMPT`, renamed `{all_page_notes}` →
`{consolidated_notes}` in `ANALYST_SYNTHESIS_PROMPT`), `state.py`
(`page_notes_list`)

**The problem:** the per-page analysis loop was hard-capped with
`min(len(pages), 40)` — anything past page 40 was silently never read by
the Analyst at all, with no warning. Even within that cap, all page-batch
notes were joined into one string and passed to a single synthesis call
relying on `sanitize_prompt_input`'s hard truncation (12000 chars) to keep
it prompt-sized, which would have silently dropped content anyway for a
paper with heavy notes even under 40 pages.

**The fix:** the per-page mapping loop no longer has a length-based cap — it
processes every page. A new `_hierarchical_reduce()` function then merges
the resulting list of page-batch notes down to a small, fixed number
(`ANALYST_REDUCE_FANOUT`, default 6) via recursive LLM-driven consolidation
(`ANALYST_REDUCE_PROMPT`, which explicitly instructs "compression of
redundancy, not compression of content" — preserve every equation, figure/
table reference, and implementation-critical detail; only remove genuine
duplication): each level merges chunks of `ANALYST_REDUCE_FANOUT` notes into
one consolidated note, and recurses if there are still too many results,
so the per-call context size stays constant regardless of paper length —
only the number of reduce calls (and cost) grows with it. The final,
already-dense set of <= 6 summaries is what reaches
`ANALYST_SYNTHESIS_PROMPT`, not a truncated blob of raw notes.

A generous, loudly-logged safety valve (`ANALYST_MAX_PAGES`, default 300)
replaces the old silent cap — this is a sanity backstop against a truly
pathological input (a several-hundred-page PDF pointed at this by mistake),
not a quality-limiting truncation that fires on any real paper; if it does
trigger, it prints a clear warning naming the env var to raise, instead of
silently reading less than the whole document.

**On revision:** individual per-page notes are now persisted as a real list
(`page_notes_list`, not just a joined string) specifically so a CRO-triggered
revision can re-run the (comparatively cheap) reduce + synthesis steps with
the new feedback attached, without re-paying for the full page-by-page
reading pass — the paper hasn't changed, so there's no reason to re-read it.

**Verified:**
- A synthetic 60-page paper: confirmed all 12 page-batches are processed
  (previously would have silently stopped at batch 8 / page 40), the
  hierarchical reduce phase correctly triggers (12 notes > fanout of 6) and
  produces 2 consolidated summaries, and the full pipeline completes.
- Direct tests of `_hierarchical_reduce` with 23 and 50 synthetic notes,
  counting actual LLM calls: confirms the exact expected number of reduce
  calls at each recursion level (4 calls for 23 notes in one level; 9+2=11
  calls for 50 notes across two levels) — the tree depth and call count
  scale as designed, with no hard content cap.
- A revision test with 15 persisted `page_notes_list` entries confirms the
  PDF is not re-read (`extract_pdf_pages` call count stays at 0) while the
  reduce + synthesis steps still re-run.

---

## Configuration

Both features are tunable via environment variables, documented in
`env.example`:

| Variable                  | Default | Meaning                                                          |
|----------------------------|---------|-------------------------------------------------------------------|
| `ANALYST_MAX_PAGES`        | 300     | Safety valve, not a quality cap — loudly logged if hit.           |
| `ANALYST_REDUCE_FANOUT`    | 6       | How many notes get merged per hierarchical-reduce call.           |
| `ENGINEER_MAX_BATCH_SIZE`  | 4       | Max files per Engineer pass, even within one Architect `group`.  |

## Known follow-on work not done here

- Engineer revisions currently redo the *entire* manifest, not just the
  file(s) the CRO flagged. Per-file evaluation/revision would cut cost
  further but needs a way for the CRO to attribute feedback to a specific
  file rather than the aggregate `implementation_notes` text.
- The hierarchical reduce prompt is not itself evaluated by the CRO (only
  the final synthesis is) — a bad early reduce could still lose fidelity
  before the CRO ever sees the result. Given the reduce prompt is
  instructed to preserve rather than compress content, this is a reasonable
  trade-off, but a spot-check step is a plausible future addition.
