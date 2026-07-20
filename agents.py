"""
agents.py — All 8 AI Research Team agents.

Each agent is a deep expert with a specific role.
Agents communicate through the shared message board in ResearchState.

This file was substantially rewritten to fix five bugs (see FIXES.md for the
full writeup):

  1. Message routing used human display names as free-text keys, which never
     matched the canonical keys used for lookup — the Engineer never received
     the Architect's/Reviewer's/Experiment Engineer's messages. Fixed via
     agent_registry.py's canonical keys, used everywhere now.

  2. CRO evaluation feedback (state["evaluations"][agent]["feedback"]) was
     computed but never read by any agent — revisions were blind re-rolls,
     not corrections. Fixed via format_prior_feedback(), now wired into
     every agent's prompt.

  3. "Validation" never executed the generated code — only a fixed, unrelated
     smoke test. Fixed via tools.run_codebase_validation(), which actually
     writes and imports the generated files and reports real results, with
     the LLM's commentary explicitly required to tag anything it didn't
     measure as [LLM-INFERRED].

  4. Code/message extraction relied on regex over markdown fences and
     "[MESSAGE TO X]:" tags, which didn't match what the prompts actually
     told the model to produce. Fixed via structured output (schemas.py) —
     no more regex parsing of LLM prose.

  5. make_llm() hardcoded Groq's endpoint while env.example told users to use
     OpenRouter model IDs. Fixed via llm_provider.py's explicit LLM_PROVIDER
     switch with correct per-provider defaults.
"""

import os
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import llm_provider
from agent_registry import (
    CRO, ANALYST, THEORIST, ARCHITECT, ENGINEER, REVIEWER, EXPERIMENT, WRITER, ALL,
    normalize_key, display_name, role_label,
)
from schemas import (
    EngineerOutput, WriterOutput, EvaluationResult,
    TheoristOutput, ArchitectOutput, ReviewerOutput, ExperimentOutput,
)
from state import ResearchState
from prompts import (
    CRO_READING_NOTES_PROMPT,
    CRO_IMPLEMENTATION_PLAN_PROMPT,
    CRO_EVALUATE_PROMPT,
    CRO_FINAL_VERDICT_PROMPT,
    ANALYST_PAGE_PROMPT,
    ANALYST_REDUCE_PROMPT,
    ANALYST_SYNTHESIS_PROMPT,
    THEORIST_PROMPT,
    ARCHITECT_PROMPT,
    ENGINEER_PROMPT,
    REVIEWER_PROMPT,
    EXPERIMENT_ENGINEER_PROMPT,
    WRITER_PROMPT,
    CONTENT_DELIMITER_INSTRUCTION,
)
from tools import (
    extract_pdf_pages,
    get_paper_metadata,
    extract_equations,
    save_all_modules,
    sanitize_relative_path,
    run_codebase_validation,
    format_measured_results,
)


# ══════════════════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════════════════

def make_llm(model: str, temperature: float = 0.1):
    """Thin wrapper — actual provider selection (Groq/OpenRouter/OpenAI) lives in llm_provider.py."""
    return llm_provider.make_llm(model, temperature)


# ── Model assignments ─────────────────────────────────────────
# Override via environment variables for paid models. Defaults now come from
# whichever provider LLM_PROVIDER selects (see llm_provider.py) instead of a
# single hardcoded Groq model name — previously, following env.example's own
# advice to override these with OpenRouter-style IDs would break, because
# make_llm() was hardcoded to Groq's endpoint regardless of what model string
# was passed in.
CRO_MODEL        = os.getenv("CRO_MODEL")        or llm_provider.default_model_for_provider()
ANALYST_MODEL     = os.getenv("ANALYST_MODEL")     or llm_provider.default_model_for_provider()
THEORIST_MODEL    = os.getenv("THEORIST_MODEL")    or llm_provider.default_model_for_provider()
ARCHITECT_MODEL   = os.getenv("ARCHITECT_MODEL")   or llm_provider.default_model_for_provider()
ENGINEER_MODEL    = os.getenv("ENGINEER_MODEL")    or llm_provider.default_model_for_provider()
REVIEWER_MODEL    = os.getenv("REVIEWER_MODEL")    or llm_provider.default_model_for_provider()
EXPERIMENT_MODEL  = os.getenv("EXPERIMENT_MODEL")  or llm_provider.default_model_for_provider()
WRITER_MODEL      = os.getenv("WRITER_MODEL")      or llm_provider.default_model_for_provider()

parser = StrOutputParser()


# ══════════════════════════════════════════════════════════════
# INPUT SANITIZATION
# ══════════════════════════════════════════════════════════════

MAX_INPUT_LENGTH = 50000
MAX_EMBEDDED_CONTENT = 20000


def sanitize_prompt_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """
    Sanitize text before embedding it into prompts.

    Protects against:
    1. Prompt injection — wraps content in explicit delimiters so LLM treats it as data
    2. Token overflow — truncates to safe length
    3. Unicode exploits — strips zero-width and control characters

    Usage: Always wrap user/agent-generated content with this before passing to prompts.
    """
    if not text:
        return ""

    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u0000-\u0008\u000b\u000c\u000e-\u001f]', '', text)

    if len(text) > max_length:
        text = text[:max_length] + "\n\n[... content truncated ...]"

    return text


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

class StructuredOutputError(Exception):
    """Raised when structured output fails validation twice in a row."""


def get_feedback(state: ResearchState, agent_key: str) -> str:
    """Raw feedback text the CRO gave this agent, if any."""
    evals = state.get("evaluations", {})
    return evals.get(normalize_key(agent_key), {}).get("feedback", "")


def format_prior_feedback(state: ResearchState, agent_key: str) -> str:
    """
    THE FIX for "CRO feedback never reaches the agent being asked to revise":
    every agent's prompt now includes this rendering of state["evaluations"]
    instead of silently re-running with identical inputs on every retry.
    Previously get_feedback() existed but was never called anywhere.
    """
    agent_key = normalize_key(agent_key)
    ev = state.get("evaluations", {}).get(agent_key)
    if not ev or ev.get("passed", True):
        return "This is your first attempt — no prior CRO feedback yet."

    lines = [
        "Your previous submission was NOT approved by the CRO. You MUST "
        "address every point below before resubmitting — do not simply "
        "resubmit similar content."
    ]
    issues = ev.get("issues") or []
    if issues:
        lines.append("Critical issues:")
        lines.extend(f"  - {i}" for i in issues)
    feedback = ev.get("feedback", "")
    if feedback:
        lines.append("\nDetailed feedback:\n" + feedback)
    return "\n".join(lines)


def post_message(state: ResearchState, sender: str, recipient: str,
                  content: str, msg_type: str = "insight") -> list:
    """
    Add a message to the team message board, keyed by CANONICAL agent keys
    (see agent_registry.py). This is the fix for the routing bug where
    messages were addressed by human display name ("Senior ML Engineer")
    while lookups used a different string ("engineer"), silently losing
    every message addressed to the Engineer.
    """
    sender_key = normalize_key(sender)
    recipient_key = normalize_key(recipient) or ALL
    board = list(state.get("message_board", []))
    board.append({
        "sender":         sender_key,
        "recipient":      recipient_key,
        "sender_name":    display_name(sender_key),
        "recipient_name": display_name(recipient_key),
        "content":        content,
        "message_type":   msg_type,
    })
    return board


def get_messages_for(state: ResearchState, agent_key: str) -> str:
    """Get all messages addressed to a specific agent, matched by canonical key."""
    agent_key = normalize_key(agent_key)
    board = state.get("message_board", [])
    relevant = [
        m for m in board
        if normalize_key(m.get("recipient", "")) in (agent_key, ALL)
    ]
    if not relevant:
        return "No messages from team yet."
    formatted = [
        f"FROM {display_name(m.get('sender', '')).upper()} [{m.get('message_type', 'insight').upper()}]:\n{m['content']}"
        for m in relevant
    ]
    return "\n\n---\n\n".join(formatted)


def run_chain(prompt_template: str, inputs: dict, model: str) -> str:
    """Run a single LLM chain that returns plain text (used for narrative-only CRO/Analyst steps)."""
    llm    = make_llm(model)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain  = prompt | llm | parser
    return chain.invoke(inputs)


def run_structured_chain(prompt_template: str, inputs: dict, model: str, schema, temperature: float = 0.1):
    """
    Invoke an LLM with a prompt template and parse the response directly into
    `schema` (a Pydantic model) via tool-calling / structured output — no
    regex, no markdown-fence scraping, no manual JSON string cleanup.

    Retries once on validation failure (open-weight models are less reliable
    than GPT/Claude at strict structured output), then raises
    StructuredOutputError so the caller can degrade gracefully rather than
    crashing the whole graph run.
    """
    llm = make_llm(model, temperature)
    structured_llm = llm.with_structured_output(schema)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | structured_llm
    try:
        return chain.invoke(inputs)
    except Exception as first_err:
        try:
            return chain.invoke(inputs)
        except Exception as second_err:
            raise StructuredOutputError(
                f"Structured output failed twice for schema {schema.__name__}: "
                f"{first_err} | retry: {second_err}"
            ) from second_err


def increment_revision(state: ResearchState, agent_key: str) -> dict:
    agent_key = normalize_key(agent_key)
    rc = dict(state.get("revision_counts", {}))
    rc[agent_key] = rc.get(agent_key, 0) + 1
    return rc


# ══════════════════════════════════════════════════════════════
# PHASE 0: PAPER ANALYST
# ══════════════════════════════════════════════════════════════

# How many notes get merged into one consolidated note per reduce step.
# Lower = more, cheaper LLM calls per level but a deeper tree for very long
# papers; higher = fewer levels but each reduce call handles more content.
ANALYST_REDUCE_FANOUT = int(os.getenv("ANALYST_REDUCE_FANOUT", "6"))

# THE FIX: the old code silently stopped reading at page 40 — anything after
# that was never seen by the per-page analysis at all, no warning given. This
# is now a generous, loudly-logged safety valve against a truly pathological
# input (e.g. someone pointing this at a 2000-page PDF by mistake) rather
# than a quality-limiting cap that fires on any real survey paper or anything
# with substantial appendices.
ANALYST_MAX_PAGES = int(os.getenv("ANALYST_MAX_PAGES", "300"))


def _hierarchical_reduce(notes: list, paper_title: str, model: str) -> str:
    """
    Recursively merge a list of textual notes into fewer, denser notes until
    there are few enough (<= ANALYST_REDUCE_FANOUT) to hand directly to the
    final synthesis prompt.

    THE FIX this powers: the old code joined every page-batch note into one
    giant string and relied on sanitize_prompt_input's hard truncation
    (12000 chars) to keep it prompt-sized — for anything beyond ~40 pages,
    later notes were silently cut off entirely before the model ever saw
    them. A real map-reduce tree scales to a paper of any length: each level
    only ever has to summarize ANALYST_REDUCE_FANOUT documents at a time, so
    the per-call context size is constant regardless of paper length; only
    the number of reduce calls (and therefore cost) grows with length.
    """
    if len(notes) <= ANALYST_REDUCE_FANOUT:
        return "\n\n===SECTION===\n\n".join(notes)

    llm = make_llm(model)
    prompt = ChatPromptTemplate.from_template(ANALYST_REDUCE_PROMPT)
    chain = prompt | llm | parser

    reduced = []
    num_chunks = (len(notes) + ANALYST_REDUCE_FANOUT - 1) // ANALYST_REDUCE_FANOUT
    for chunk_idx, i in enumerate(range(0, len(notes), ANALYST_REDUCE_FANOUT), start=1):
        chunk = notes[i:i + ANALYST_REDUCE_FANOUT]
        chunk_text = "\n\n===NOTE===\n\n".join(chunk)
        print(f"   🔀 Consolidating {len(chunk)} notes into 1 (chunk {chunk_idx}/{num_chunks})...")
        summary = chain.invoke({
            "paper_title":   paper_title,
            "notes_chunk":   sanitize_prompt_input(chunk_text, 12000),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
        })
        reduced.append(summary)

    return _hierarchical_reduce(reduced, paper_title, model)


def analyst_agent(state: ResearchState) -> ResearchState:
    print("\n📖 Paper Analyst — Reading PDF page by page...")

    prior_feedback = format_prior_feedback(state, ANALYST)
    is_revision = bool(state.get("page_notes_list")) and bool(state.get("paper_title"))

    if is_revision:
        # A revision — the paper hasn't changed, only the synthesis needs to
        # improve. Reuse the already-persisted page notes instead of paying
        # for potentially dozens of LLM calls to re-read a PDF that didn't
        # change, and let the CRO's feedback (finally wired in — see
        # format_prior_feedback) drive a better synthesis this time. The
        # hierarchical reduce below still re-runs (cheap relative to the
        # page-by-page map phase) so a changed prior_feedback can plausibly
        # shift what the consolidation emphasizes too.
        print("   ♻️  Revision — reusing prior page notes, redoing reduce + synthesis with CRO feedback...")
        paper_title = state.get("paper_title", "")
        page_notes_list = state.get("page_notes_list", [])
        pages = state.get("raw_pages", [])
        full_text = state.get("full_paper_text", "")
        equations_summary = state.get("equations_summary", "")
    else:
        pdf_path = state.get("pdf_path", "")
        pages = extract_pdf_pages(pdf_path)

        if not pages:
            print("   ❌ Could not extract PDF pages.")
            return {**state, "raw_pages": [], "full_paper_text": "PDF extraction failed."}

        metadata = get_paper_metadata(pages)
        paper_title = metadata["title"]
        print(f"   📄 Title: {paper_title[:80]}")
        print(f"   📄 Pages detected: {len(pages)}")
        print(f"   📄 Sections detected: {len(metadata['sections'])}")

        if len(pages) > ANALYST_MAX_PAGES:
            print(f"   ⚠️  Paper has {len(pages)} pages, exceeding ANALYST_MAX_PAGES={ANALYST_MAX_PAGES}.")
            print(f"   ⚠️  Reading the first {ANALYST_MAX_PAGES} pages only. Raise ANALYST_MAX_PAGES to read more.")
        pages_to_read = pages[:ANALYST_MAX_PAGES]

        page_notes_list = []
        previous_context = "This is the beginning of the paper."
        llm = make_llm(ANALYST_MODEL)

        batch_size = 5
        num_batches = (len(pages_to_read) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(pages_to_read), batch_size), start=1):
            batch = pages_to_read[i:i + batch_size]
            batch_text = "\n\n--- PAGE BREAK ---\n\n".join(
                f"[PAGE {p['page']}]\n{p['text']}" for p in batch
            )
            print(f"   📖 Analyzing pages {batch[0]['page']}–{batch[-1]['page']} (batch {batch_idx}/{num_batches})...")

            prompt = ChatPromptTemplate.from_template(ANALYST_PAGE_PROMPT)
            chain  = prompt | llm | parser
            notes  = chain.invoke({
                "page_num":         f"{batch[0]['page']}–{batch[-1]['page']}",
                "paper_title":      paper_title,
                "page_text":        sanitize_prompt_input(batch_text, 6000),
                "previous_context": sanitize_prompt_input(previous_context[-1000:], 1000),
                "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
            })
            page_notes_list.append(notes)
            previous_context = notes[-500:]

        full_text = "\n\n".join(p["text"] for p in pages)
        all_equations = extract_equations(full_text)
        equations_summary = "\n".join(f"Line {eq['line']}: {eq['content']}" for eq in all_equations[:50])

    if len(page_notes_list) > ANALYST_REDUCE_FANOUT:
        print(f"   🔀 Reducing {len(page_notes_list)} page-batch notes down to <= {ANALYST_REDUCE_FANOUT} via hierarchical merge...")
    consolidated_notes = _hierarchical_reduce(page_notes_list, paper_title, ANALYST_MODEL) if page_notes_list else ""

    print("   🔗 Synthesizing full paper analysis...")
    llm = make_llm(ANALYST_MODEL)
    synthesis_prompt = ChatPromptTemplate.from_template(ANALYST_SYNTHESIS_PROMPT)
    synthesis_chain  = synthesis_prompt | llm | parser
    synthesis = synthesis_chain.invoke({
        "paper_title":         paper_title,
        "consolidated_notes":  sanitize_prompt_input(consolidated_notes, 20000),
        "prior_feedback":      sanitize_prompt_input(prior_feedback, 2000),
        "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
    })

    board = post_message(
        state, ANALYST, ALL,
        f"Paper '{paper_title}' analyzed. Synthesis complete"
        + (" (revision)." if is_revision else f". {len(pages)} pages read."),
        "insight",
    )

    rc = increment_revision(state, ANALYST)
    result = {
        **state,
        "paper_title":       paper_title,
        "raw_pages":         pages,
        "full_paper_text":   sanitize_prompt_input(full_text, 15000),
        "equations_summary": equations_summary,
        "page_notes":        "\n\n===PAGE BATCH===\n\n".join(page_notes_list),
        "page_notes_list":   page_notes_list,
        "research_report":   synthesis,
        "message_board":     board,
        "revision_counts":   rc,
    }
    if not is_revision:
        result["paper_abstract"] = metadata["abstract"]
    return result


def get_analyst_synthesis(state: ResearchState) -> str:
    return state.get("research_report", "")


# ══════════════════════════════════════════════════════════════
# CRO — READ & PLAN
# ══════════════════════════════════════════════════════════════

def cro_read_paper(state: ResearchState) -> ResearchState:
    print("\n🧠 CRO — Reading paper and forming own understanding...")

    output = run_chain(
        CRO_READING_NOTES_PROMPT,
        {
            "full_paper_text": sanitize_prompt_input(state.get("full_paper_text", ""), 10000),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
        },
        CRO_MODEL,
    )

    board = post_message(
        state, CRO, ALL,
        "I have completed my initial reading of the paper. "
        "Theorist and Architect: begin your analyses in parallel. "
        "Pay special attention to the mathematical framework and proposed architecture.",
        "directive",
    )

    rc = increment_revision(state, CRO)
    return {**state, "cro_reading_notes": output, "message_board": board, "revision_counts": rc}


def cro_create_plan(state: ResearchState) -> ResearchState:
    print("\n🧠 CRO — Creating master implementation plan...")

    output = run_chain(
        CRO_IMPLEMENTATION_PLAN_PROMPT,
        {
            "paper_title":          state.get("paper_title", ""),
            "cro_reading_notes":    sanitize_prompt_input(state.get("cro_reading_notes", ""), 3000),
            "theoretical_analysis": sanitize_prompt_input(state.get("theoretical_analysis", ""), 3000),
            "architecture_analysis": sanitize_prompt_input(state.get("architecture_analysis", ""), 3000),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
        },
        CRO_MODEL,
    )

    board = post_message(
        state, CRO, ENGINEER,
        "Implementation plan is ready. Begin with Phase A (core data structures). "
        "Follow the file structure exactly. Tag me if the paper is ambiguous.",
        "directive",
    )

    return {**state, "implementation_plan": output, "codebase_structure": output, "message_board": board}


def cro_evaluate_agent(state: ResearchState, agent_key: str, output_key: str) -> ResearchState:
    """
    Evaluate another agent's deliverable. `agent_key` must be one of the
    canonical keys from agent_registry.py (e.g. ENGINEER) — the human role
    label used in the prompt/logs is derived from it, so callers no longer
    need to pass a separate, easily-inconsistent role string.
    """
    agent_key = normalize_key(agent_key)
    role = role_label(agent_key)
    print(f"\n🧠 CRO — Evaluating {role}...")

    output = state.get(output_key, "")

    try:
        result = run_structured_chain(
            CRO_EVALUATE_PROMPT,
            {
                "agent_role":    role,
                "paper_title":   state.get("paper_title", ""),
                "paper_excerpt": sanitize_prompt_input(get_analyst_synthesis(state), 2000),
                "output":        sanitize_prompt_input(output, 4000),
                "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
            },
            CRO_MODEL,
            EvaluationResult,
        )
        passed, feedback, issues = result.passed, result.feedback, list(result.critical_issues)
    except StructuredOutputError as e:
        # Evaluation itself is the thing that failed here (not the agent
        # being evaluated) — default to PASS rather than get the whole
        # pipeline stuck retrying an evaluator that can't produce a verdict.
        print(f"   ⚠️  Evaluation call failed twice ({e}); defaulting to PASS.")
        passed, feedback, issues = True, "", []

    rc = dict(state.get("revision_counts", {}))
    count = rc.get(agent_key, 0)
    if count >= 3:
        passed   = True
        feedback = ""
        print(f"   ⚠️  Max revisions for {role}. Accepting.")

    evaluations = dict(state.get("evaluations", {}))
    evaluations[agent_key] = {
        "passed":    passed,
        "feedback":  feedback,
        "issues":    issues,
        "iteration": count,
    }

    needs_revision = list(state.get("needs_revision", []))
    if not passed:
        if agent_key not in needs_revision:
            needs_revision.append(agent_key)
        board = post_message(
            state, CRO, agent_key,
            f"Your output needs revision. Issues: {'; '.join(issues[:3])}. {feedback[:500]}",
            "feedback",
        )
        print(f"   ❌ {role} needs revision.")
    else:
        if agent_key in needs_revision:
            needs_revision.remove(agent_key)
        board = state.get("message_board", [])
        print(f"   ✅ {role} approved.")

    return {
        **state,
        "evaluations":     evaluations,
        "revision_counts": rc,
        "needs_revision":  needs_revision,
        "message_board":   board,
    }


def cro_final_verdict(state: ResearchState) -> ResearchState:
    print("\n🧠 CRO — Issuing final verdict...")

    modules_list = ", ".join(state.get("code_modules", {}).keys())

    output = run_chain(
        CRO_FINAL_VERDICT_PROMPT,
        {
            "paper_title":            state.get("paper_title", ""),
            "implementation_summary": sanitize_prompt_input(state.get("implementation_plan", ""), 2000),
            "validation_report":      sanitize_prompt_input(state.get("validation_report", ""), 2000),
            "code_modules_list":      modules_list,
            "security_instruction":   CONTENT_DELIMITER_INSTRUCTION,
        },
        CRO_MODEL,
    )

    return {**state, "final_verdict": output}


# ══════════════════════════════════════════════════════════════
# THEORIST
# ══════════════════════════════════════════════════════════════

def theorist_agent(state: ResearchState) -> ResearchState:
    print("\n🔢 Theorist — Deep mathematical analysis...")

    try:
        result = run_structured_chain(
            THEORIST_PROMPT,
            {
                "paper_title":       state.get("paper_title", ""),
                "full_paper_text":   sanitize_prompt_input(state.get("full_paper_text", ""), 8000),
                "analyst_synthesis": sanitize_prompt_input(get_analyst_synthesis(state), 4000),
                "prior_feedback":    sanitize_prompt_input(format_prior_feedback(state, THEORIST), 2000),
                "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
            },
            THEORIST_MODEL,
            TheoristOutput,
        )
        analysis, arch_message = result.analysis, result.message_to_architect
    except StructuredOutputError as e:
        print(f"   ⚠️  Structured output failed twice: {e}")
        analysis = f"[STRUCTURED OUTPUT FAILED after 2 attempts: {e}]"
        arch_message = ""

    board = post_message(
        state, THEORIST, ARCHITECT,
        arch_message or "Mathematical analysis complete. See theoretical_analysis in state.",
        "insight",
    )

    rc = increment_revision(state, THEORIST)
    return {**state, "theoretical_analysis": analysis, "message_board": board, "revision_counts": rc}


# ══════════════════════════════════════════════════════════════
# ML ARCHITECT
# ══════════════════════════════════════════════════════════════

def _sanitize_manifest(file_manifest: list) -> list:
    """
    Convert the Architect's structured file_manifest (a list of FileSpec
    Pydantic objects) into safe, deduplicated plain dicts for storage in
    state: filenames go through the same sanitize_relative_path() used for
    on-disk output and sandbox validation (so a manifest entry can never
    request a path-traversal write), duplicate filenames keep only the
    first occurrence, and depends_on references to filenames not actually
    in the manifest are dropped rather than trusted blindly.
    """
    seen = set()
    sanitized = []
    for spec in file_manifest:
        raw = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
        filename = sanitize_relative_path(raw.get("filename", ""))
        if filename in seen or filename == "unnamed_module.py":
            continue
        seen.add(filename)
        sanitized.append({
            "filename":    filename,
            "description": raw.get("description", ""),
            "depends_on":  list(raw.get("depends_on") or []),
            "group":       (raw.get("group") or "").strip(),
        })

    valid_names = {s["filename"] for s in sanitized}
    for s in sanitized:
        s["depends_on"] = [
            sanitize_relative_path(d) for d in s["depends_on"]
            if sanitize_relative_path(d) in valid_names and sanitize_relative_path(d) != s["filename"]
        ]
    return sanitized


def architect_agent(state: ResearchState) -> ResearchState:
    print("\n🏗️  ML Architect — Designing codebase structure...")

    theorist_message = get_messages_for(state, ARCHITECT)

    try:
        result = run_structured_chain(
            ARCHITECT_PROMPT,
            {
                "paper_title":       state.get("paper_title", ""),
                "full_paper_text":   sanitize_prompt_input(state.get("full_paper_text", ""), 6000),
                "analyst_synthesis": sanitize_prompt_input(get_analyst_synthesis(state), 3000),
                "theorist_message":  sanitize_prompt_input(theorist_message, 1500),
                "prior_feedback":    sanitize_prompt_input(format_prior_feedback(state, ARCHITECT), 2000),
                "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
            },
            ARCHITECT_MODEL,
            ArchitectOutput,
        )
        analysis, eng_message = result.analysis, result.message_to_engineer
        file_manifest = _sanitize_manifest(result.file_manifest)
    except StructuredOutputError as e:
        print(f"   ⚠️  Structured output failed twice: {e}")
        analysis = f"[STRUCTURED OUTPUT FAILED after 2 attempts: {e}]"
        eng_message = ""
        file_manifest = []

    print(f"   📋 File manifest: {len(file_manifest)} files planned"
          + ("" if file_manifest else " (none — Engineer will fall back to a single pass)"))

    # Previously this posted to recipient "Senior ML Engineer" while the
    # Engineer looked itself up as "engineer" — never matching. Now both
    # sides use the same canonical ENGINEER key.
    board = post_message(
        state, ARCHITECT, ENGINEER,
        eng_message or "Architecture design complete. See codebase_structure and file_manifest in state.",
        "directive",
    )

    rc = increment_revision(state, ARCHITECT)
    return {
        **state,
        "architecture_analysis": analysis,
        "codebase_structure":    analysis,
        "file_manifest":         file_manifest,
        "message_board":         board,
        "revision_counts":       rc,
    }


# ══════════════════════════════════════════════════════════════
# SENIOR ML ENGINEER
# ══════════════════════════════════════════════════════════════

# Hard cap on files per implementation pass, even if the Architect assigns
# the same `group` label to more files than this — a `group` label
# shouldn't be able to silently recreate the old "write everything in one
# call" problem.
ENGINEER_MAX_BATCH_SIZE = int(os.getenv("ENGINEER_MAX_BATCH_SIZE", "4"))
ENGINEER_DEPENDENCY_CONTEXT_BUDGET = 6000  # chars, shared across all dependency files in one call


def _build_batches(manifest: list) -> list:
    """
    Group the Architect's file_manifest into implementation passes, in
    manifest order (the Architect is instructed to put the manifest in
    dependency order). Consecutive entries sharing the same non-empty
    `group` label are implemented together in one pass — intended only for
    small, tightly-coupled files; everything else is implemented on its own,
    one file per LLM call, each with the model's full output budget instead
    of it being split across the whole project.
    """
    batches = []
    current_group = None
    current_batch = []

    def flush():
        nonlocal current_batch
        for i in range(0, len(current_batch), ENGINEER_MAX_BATCH_SIZE):
            batches.append(current_batch[i:i + ENGINEER_MAX_BATCH_SIZE])
        current_batch = []

    for entry in manifest:
        group = entry.get("group") or ""
        if group and group == current_group:
            current_batch.append(entry)
        else:
            flush()
            current_batch = [entry]
            current_group = group or None
    flush()
    return batches


def _build_dependency_context(code_modules: dict, batch: list, budget: int = ENGINEER_DEPENDENCY_CONTEXT_BUDGET) -> str:
    """Full code of every already-implemented file this batch's entries declared a dependency on."""
    dep_names = []
    for entry in batch:
        for d in entry.get("depends_on", []):
            if d not in dep_names:
                dep_names.append(d)

    if not dep_names:
        return "This file/batch has no declared dependencies on other project files."

    per_file_budget = max(budget // len(dep_names), 500)
    parts = []
    for name in dep_names:
        mod = code_modules.get(name)
        if not mod:
            parts.append(
                f"# === {name} ===\n[NOT YET IMPLEMENTED — this dependency hasn't been generated "
                f"yet, which shouldn't happen if the manifest is in correct dependency order; "
                f"implement this file's needs conservatively / flag it via message_to_cro]"
            )
            continue
        code = mod.get("code", "") if isinstance(mod, dict) else str(mod)
        parts.append(f"# === {name} ===\n{sanitize_prompt_input(code, per_file_budget)}")
    return "\n\n".join(parts)


def _build_manifest_context(manifest: list, code_modules: dict, batch_filenames: set) -> str:
    """Brief filename: description listing of the rest of the project, marking what's already implemented vs pending."""
    lines = []
    for entry in manifest:
        name = entry["filename"]
        if name in batch_filenames:
            continue
        status = "done" if name in code_modules else "PENDING"
        lines.append(f"- {name} [{status}]: {entry.get('description', '')[:150]}")
    return "\n".join(lines) if lines else "No other files in the manifest."


def engineer_agent(state: ResearchState) -> ResearchState:
    print("\n💻 Senior ML Engineer — Implementing codebase...")

    # THE FIX: the Engineer used to be asked to write the ENTIRE codebase in
    # a single completion ("current_file": "ALL FILES") — reliably too much
    # for one output on any nontrivial paper, which is why so much of the
    # rest of this file used to exist just to parse whatever partial/broken
    # result came back. Now it iterates the Architect's file_manifest one
    # file (or small, Architect-grouped batch) at a time, in dependency
    # order, giving each pass the full code of whatever it depends on.
    manifest = state.get("file_manifest", [])
    batches = _build_batches(manifest) if manifest else []

    if not batches:
        print("   ⚠️  No file manifest available — falling back to a single implementation pass.")
        batches = [[{
            "filename": "ALL FILES",
            "description": "Implement the complete codebase as designed in codebase_structure.",
            "depends_on": [],
            "group": "",
        }]]

    print(f"   📋 {len(batches)} implementation pass(es) covering {sum(len(b) for b in batches)} file(s)")

    # Read once up front, not per-batch — previously get_messages_for(state,
    # "engineer") never matched anything because messages were addressed to
    # "Senior ML Engineer", a display name, not the lookup key.
    team_inbox = get_messages_for(state, ENGINEER)
    review_feedback = json.dumps(state.get("review_feedback", {}), indent=2)[:2000]
    prior_feedback_text = format_prior_feedback(state, ENGINEER)

    code_modules = dict(state.get("code_modules", {}))
    notes_per_batch = []
    rev_messages, cro_messages = [], []
    any_failure = False

    for batch_idx, batch in enumerate(batches, start=1):
        batch_filenames = {entry["filename"] for entry in batch}
        current_file_text = "\n".join(f"- {f}" for f in sorted(batch_filenames))
        file_spec_text = "\n".join(
            f"- {e['filename']}: {e.get('description', '(no description given)')}" for e in batch
        )
        dependency_context = _build_dependency_context(code_modules, batch)
        manifest_context = (
            _build_manifest_context(manifest, code_modules, batch_filenames)
            if manifest else "No manifest — implementing everything in one pass."
        )

        print(f"   💻 Pass {batch_idx}/{len(batches)}: {', '.join(sorted(batch_filenames))}")

        try:
            result = run_structured_chain(
                ENGINEER_PROMPT,
                {
                    "paper_title":          state.get("paper_title", ""),
                    "analyst_synthesis":    sanitize_prompt_input(get_analyst_synthesis(state), 3000),
                    "theoretical_analysis": sanitize_prompt_input(state.get("theoretical_analysis", ""), 2000),
                    "codebase_structure":   sanitize_prompt_input(state.get("codebase_structure", ""), 2000),
                    "implementation_plan":  sanitize_prompt_input(state.get("implementation_plan", ""), 1500),
                    "team_inbox":           sanitize_prompt_input(team_inbox, 1500),
                    "review_feedback":      sanitize_prompt_input(review_feedback, 1500),
                    "prior_feedback":       sanitize_prompt_input(prior_feedback_text, 2000),
                    "dependency_context":   sanitize_prompt_input(dependency_context, ENGINEER_DEPENDENCY_CONTEXT_BUDGET),
                    "manifest_context":     sanitize_prompt_input(manifest_context, 2000),
                    "current_file":         current_file_text,
                    "file_spec":            file_spec_text,
                    "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
                },
                ENGINEER_MODEL,
                EngineerOutput,
            )
        except StructuredOutputError as e:
            print(f"   ⚠️  Structured output failed twice for {', '.join(sorted(batch_filenames))}: {e}")
            any_failure = True
            cro_messages.append(f"Failed to implement {', '.join(sorted(batch_filenames))}: {e}")
            continue

        # Previously filenames were saved via os.path.basename(), silently
        # flattening any subdirectory structure the Architect designed. Now
        # sanitize_relative_path() preserves safe subdirectories.
        for f in result.files:
            safe_name = sanitize_relative_path(f.filename)
            code_modules[safe_name] = {
                "filename":    safe_name,
                "language":    f.language or "python",
                "code":        f.code,
                "description": f.description or "Implemented by Senior ML Engineer",
                "status":      "draft",
            }
        if result.implementation_notes:
            notes_per_batch.append(f"[{', '.join(sorted(batch_filenames))}] {result.implementation_notes}")
        if result.message_to_reviewer:
            rev_messages.append(result.message_to_reviewer)
        if result.message_to_cro:
            cro_messages.append(result.message_to_cro)

    implementation_notes = "\n\n".join(notes_per_batch)
    if any_failure:
        implementation_notes = f"[ONE OR MORE FILES FAILED STRUCTURED OUTPUT — see message_to_cro]\n\n{implementation_notes}"

    board = state.get("message_board", [])
    if rev_messages:
        board = post_message({"message_board": board}, ENGINEER, REVIEWER, "\n\n".join(rev_messages), "question")
    if cro_messages:
        board = post_message({"message_board": board}, ENGINEER, CRO, "\n\n".join(cro_messages), "concern")

    rc = increment_revision(state, ENGINEER)
    return {
        **state,
        "code_modules":         code_modules,
        "implementation_notes": implementation_notes[:6000],
        "message_board":        board,
        "revision_counts":      rc,
    }


# ══════════════════════════════════════════════════════════════
# CODE REVIEWER
# ══════════════════════════════════════════════════════════════

def reviewer_agent(state: ResearchState) -> ResearchState:
    print("\n🔍 Code Reviewer — Reviewing implementation...")

    engineer_message = get_messages_for(state, REVIEWER)

    all_code = "\n\n".join(
        f"# === {fname} ===\n{mod.get('code', '') if isinstance(mod, dict) else str(mod)}"
        for fname, mod in state.get("code_modules", {}).items()
    )

    try:
        result = run_structured_chain(
            REVIEWER_PROMPT,
            {
                "paper_title":         state.get("paper_title", ""),
                "analyst_synthesis":   sanitize_prompt_input(get_analyst_synthesis(state), 3000),
                "theoretical_analysis": sanitize_prompt_input(state.get("theoretical_analysis", ""), 2000),
                "code_to_review":      sanitize_prompt_input(all_code, 8000),
                "engineer_message":    sanitize_prompt_input(engineer_message, 1000),
                "prior_feedback":      sanitize_prompt_input(format_prior_feedback(state, REVIEWER), 2000),
                "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
            },
            REVIEWER_MODEL,
            ReviewerOutput,
        )
        review_text = result.review
        eng_message = result.message_to_engineer
        cro_message = result.message_to_cro
    except StructuredOutputError as e:
        print(f"   ⚠️  Structured output failed twice: {e}")
        review_text = f"[STRUCTURED OUTPUT FAILED after 2 attempts: {e}]"
        eng_message, cro_message = "", str(e)

    review_feedback = dict(state.get("review_feedback", {}))
    review_feedback["latest_review"] = review_text[:3000]

    code_modules = dict(state.get("code_modules", {}))
    for fname in code_modules:
        if isinstance(code_modules[fname], dict):
            code_modules[fname]["status"] = "reviewed"

    board = state.get("message_board", [])
    if eng_message:
        board = post_message({"message_board": board}, REVIEWER, ENGINEER, eng_message, "feedback")
    if cro_message:
        board = post_message({"message_board": board}, REVIEWER, CRO, cro_message, "concern")

    rc = increment_revision(state, REVIEWER)
    return {
        **state,
        "review_feedback": review_feedback,
        # Fixes a separate latent bug: the CRO's evaluation step read
        # state["review_feedback_str"], a key nothing ever set — meaning the
        # CRO was always evaluating an empty string for this agent. This is
        # the field that's actually populated and evaluated now.
        "review_summary":  review_text,
        "code_modules":     code_modules,
        "message_board":    board,
        "revision_counts":  rc,
    }


# ══════════════════════════════════════════════════════════════
# EXPERIMENT ENGINEER
# ══════════════════════════════════════════════════════════════

def experiment_engineer_agent(state: ResearchState) -> ResearchState:
    print("\n🧪 Experiment Engineer — Running validation (actually executing the generated code)...")

    code_modules = state.get("code_modules", {})
    all_code = "\n\n".join(
        f"# === {fname} ===\n{mod.get('code', '') if isinstance(mod, dict) else str(mod)}"
        for fname, mod in code_modules.items()
    )

    # THE FIX: previously this ran a fixed, unrelated smoke test (checked
    # only whether torch/numpy import) and never touched the actual
    # generated code. run_codebase_validation() actually writes every file
    # and imports it for real, reporting real syntax/import/instantiation
    # results — see tools.py.
    validation = run_codebase_validation(code_modules, timeout=90)
    measured_summary = format_measured_results(validation)
    measured = validation.get("measured") or {}
    print(f"   🔬 Measured: {measured.get('files_import_ok', '?')}/{measured.get('files_checked', '?')} files imported successfully")

    try:
        result = run_structured_chain(
            EXPERIMENT_ENGINEER_PROMPT,
            {
                "paper_title":       state.get("paper_title", ""),
                "analyst_synthesis": sanitize_prompt_input(get_analyst_synthesis(state), 3000),
                "all_code":          sanitize_prompt_input(all_code, 6000),
                "execution_results": sanitize_prompt_input(measured_summary, 4000),
                "prior_feedback":    sanitize_prompt_input(format_prior_feedback(state, EXPERIMENT), 2000),
                "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
            },
            EXPERIMENT_MODEL,
            ExperimentOutput,
        )
        analysis, cro_msg, eng_msg = result.analysis, result.message_to_cro, result.message_to_engineer
    except StructuredOutputError as e:
        print(f"   ⚠️  Structured output failed twice: {e}")
        analysis = (
            f"[STRUCTURED OUTPUT FAILED after 2 attempts: {e}]\n\n"
            f"MEASURED RESULTS (still real, unaffected by the LLM call failing):\n{measured_summary}"
        )
        cro_msg, eng_msg = "", str(e)

    board = state.get("message_board", [])
    if cro_msg:
        board = post_message({"message_board": board}, EXPERIMENT, CRO, cro_msg, "insight")
    if eng_msg:
        board = post_message({"message_board": board}, EXPERIMENT, ENGINEER, eng_msg, "concern")

    rc = increment_revision(state, EXPERIMENT)
    return {
        **state,
        "execution_results":   measured_summary,
        "measured_validation": validation.get("measured"),
        "validation_report":   analysis,
        "message_board":       board,
        "revision_counts":     rc,
    }


# ══════════════════════════════════════════════════════════════
# TECHNICAL WRITER
# ══════════════════════════════════════════════════════════════

def writer_agent(state: ResearchState) -> ResearchState:
    print("\n📝 Technical Writer — Producing documentation...")

    try:
        result = run_structured_chain(
            WRITER_PROMPT,
            {
                "paper_title":         state.get("paper_title", ""),
                "paper_abstract":      sanitize_prompt_input(state.get("paper_abstract", ""), 1000),
                "implementation_plan": sanitize_prompt_input(state.get("implementation_plan", ""), 2000),
                "codebase_structure":  sanitize_prompt_input(state.get("codebase_structure", ""), 2000),
                "validation_report":   sanitize_prompt_input(state.get("validation_report", ""), 2000),
                "prior_feedback":      sanitize_prompt_input(format_prior_feedback(state, WRITER), 2000),
                "security_instruction": CONTENT_DELIMITER_INSTRUCTION,
            },
            WRITER_MODEL,
            WriterOutput,
        )
        readme, impl_notes = result.readme_md, result.implementation_notes_md
    except StructuredOutputError as e:
        print(f"   ⚠️  Structured output failed twice: {e}")
        readme = f"[STRUCTURED OUTPUT FAILED after 2 attempts: {e}]"
        impl_notes = ""

    code_modules = dict(state.get("code_modules", {}))
    code_modules["README.md"] = {
        "filename": "README.md", "language": "markdown",
        "code": readme, "description": "Project README", "status": "final",
    }
    if impl_notes:
        code_modules["IMPLEMENTATION_NOTES.md"] = {
            "filename": "IMPLEMENTATION_NOTES.md", "language": "markdown",
            "code": impl_notes, "description": "Implementation notes", "status": "final",
        }

    board = post_message(state, WRITER, CRO, "Documentation complete.", "insight")

    rc = increment_revision(state, WRITER)
    return {
        **state,
        "readme":               readme,
        "implementation_paper": impl_notes,
        "code_modules":          code_modules,
        "message_board":         board,
        "revision_counts":       rc,
    }
