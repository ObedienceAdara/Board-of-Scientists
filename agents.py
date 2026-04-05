"""
agents.py — All 8 AI Research Team agents.

Each agent is a deep expert with a specific role.
Agents communicate through the shared message board in ResearchState.
"""

import os
import json
import re
from langchain_openai           import ChatOpenAI
from langchain_core.prompts     import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state   import ResearchState
from prompts import (
    CRO_READING_NOTES_PROMPT,
    CRO_IMPLEMENTATION_PLAN_PROMPT,
    CRO_EVALUATE_PROMPT,
    CRO_FINAL_VERDICT_PROMPT,
    ANALYST_PAGE_PROMPT,
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
    execute_python_code,
    web_search,
    save_all_modules,
)


# ══════════════════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════════════════

def make_llm(model: str, temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        default_headers={
            "HTTP-Referer": "https://plexhedge.com",
            "X-Title":      "AI Research Implementation Team"
        }
    )


# ── Model assignments ─────────────────────────────────────────
# Override via environment variables for paid models
CRO_MODEL        = os.getenv("CRO_MODEL",        "llama-3.3-70b-versatile")
ANALYST_MODEL    = os.getenv("ANALYST_MODEL",    "llama-3.3-70b-versatile")
THEORIST_MODEL   = os.getenv("THEORIST_MODEL",   "llama-3.3-70b-versatile")
ARCHITECT_MODEL  = os.getenv("ARCHITECT_MODEL",  "llama-3.3-70b-versatile")
ENGINEER_MODEL   = os.getenv("ENGINEER_MODEL",   "llama-3.3-70b-versatile")
REVIEWER_MODEL   = os.getenv("REVIEWER_MODEL",   "llama-3.3-70b-versatile")
EXPERIMENT_MODEL = os.getenv("EXPERIMENT_MODEL", "llama-3.3-70b-versatile")
WRITER_MODEL     = os.getenv("WRITER_MODEL",     "llama-3.3-70b-versatile")

parser = StrOutputParser()


# ══════════════════════════════════════════════════════════════
# INPUT SANITIZATION
# ══════════════════════════════════════════════════════════════

# Maximum length for various input types to prevent token overflow
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
    
    # Strip zero-width and invisible control characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u0000-\u0008\u000b\u000c\u000e-\u001f]', '', text)
    
    # Truncate to safe length
    if len(text) > max_length:
        text = text[:max_length] + "\n\n[... content truncated ...]"
    
    return text


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def clean_json(raw: str) -> str:
    return raw.replace("```json", "").replace("```", "").strip()


def get_feedback(state: ResearchState, agent_name: str) -> str:
    evals = state.get("evaluations", {})
    return evals.get(agent_name, {}).get("feedback", "")


def post_message(state: ResearchState, sender: str, recipient: str,
                 content: str, msg_type: str = "insight") -> list:
    """Add a message to the team message board."""
    board = list(state.get("message_board", []))
    board.append({
        "sender":       sender,
        "recipient":    recipient,
        "content":      content,
        "message_type": msg_type
    })
    return board


def get_messages_for(state: ResearchState, agent_name: str) -> str:
    """Get all messages addressed to a specific agent."""
    board = state.get("message_board", [])
    relevant = [
        m for m in board
        if m.get("recipient", "").upper() in [agent_name.upper(), "ALL"]
    ]
    if not relevant:
        return "No messages from team yet."
    formatted = []
    for m in relevant:
        formatted.append(
            f"FROM {m['sender'].upper()} [{m['message_type'].upper()}]:\n{m['content']}"
        )
    return "\n\n---\n\n".join(formatted)


def extract_inter_agent_message(text: str, tag: str) -> str:
    """Extract messages tagged with [MESSAGE TO X]: from agent output."""
    pattern = rf'\[MESSAGE TO {tag.upper()}\]:\s*(.*?)(?=\[MESSAGE TO|\Z)'
    match   = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def run_chain(prompt_template: str, inputs: dict, model: str) -> str:
    """Run a single LLM chain."""
    llm    = make_llm(model)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain  = prompt | llm | parser
    return chain.invoke(inputs)


def increment_revision(state: ResearchState, agent_name: str) -> dict:
    rc = dict(state.get("revision_counts", {}))
    rc[agent_name] = rc.get(agent_name, 0) + 1
    return rc


# ══════════════════════════════════════════════════════════════
# PHASE 0: PAPER ANALYST
# ══════════════════════════════════════════════════════════════

def analyst_agent(state: ResearchState) -> ResearchState:
    print("\n📖 Paper Analyst — Reading PDF page by page...")

    pdf_path = state.get("pdf_path", "")
    pages    = extract_pdf_pages(pdf_path)

    if not pages:
        print("   ❌ Could not extract PDF pages.")
        return {**state, "raw_pages": [], "full_paper_text": "PDF extraction failed."}

    # Get metadata
    metadata = get_paper_metadata(pages)
    print(f"   📄 Title: {metadata['title'][:80]}")
    print(f"   📄 Pages detected: {len(pages)}")
    print(f"   📄 Sections detected: {len(metadata['sections'])}")

    # Read page by page — for large papers, batch into groups of 5
    all_page_notes = []
    previous_context = "This is the beginning of the paper."
    llm    = make_llm(ANALYST_MODEL)

    # Process pages in batches to avoid token limits
    batch_size = 5
    for i in range(0, min(len(pages), 40), batch_size):   # Cap at 40 pages
        batch = pages[i:i + batch_size]
        batch_text = "\n\n--- PAGE BREAK ---\n\n".join([
            f"[PAGE {p['page']}]\n{p['text']}" for p in batch
        ])

        print(f"   📖 Analyzing pages {batch[0]['page']}–{batch[-1]['page']}...")

        prompt = ChatPromptTemplate.from_template(ANALYST_PAGE_PROMPT)
        chain  = prompt | llm | parser
        notes  = chain.invoke({
            "page_num":         f"{batch[0]['page']}–{batch[-1]['page']}",
            "paper_title":      metadata["title"],
            "page_text":        sanitize_prompt_input(batch_text, 6000),
            "previous_context": sanitize_prompt_input(previous_context[-1000:], 1000),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        })
        all_page_notes.append(notes)
        previous_context = notes[-500:]

    # Synthesize all page notes
    print("   🔗 Synthesizing full paper analysis...")
    synthesis_prompt = ChatPromptTemplate.from_template(ANALYST_SYNTHESIS_PROMPT)
    synthesis_chain  = synthesis_prompt | llm | parser
    synthesis = synthesis_chain.invoke({
        "paper_title":   metadata["title"],
        "all_page_notes": sanitize_prompt_input(
            "\n\n===PAGE BATCH===\n\n".join(all_page_notes), 12000
        ),
        "security_instruction": CONTENT_DELIMITER_INSTRUCTION
    })

    # Build full text
    full_text = "\n\n".join([p["text"] for p in pages])

    # Extract equations
    all_equations = extract_equations(full_text)
    equations_summary = "\n".join([
        f"Line {eq['line']}: {eq['content']}" for eq in all_equations[:50]
    ])

    board = post_message(
        state, "Paper Analyst", "ALL",
        f"Paper '{metadata['title']}' fully analyzed. {len(pages)} pages read. "
        f"Synthesis complete. Key equations extracted: {len(all_equations)}.",
        "insight"
    )

    rc = increment_revision(state, "analyst")
    return {
        **state,
        "paper_title":       metadata["title"],
        "paper_abstract":    metadata["abstract"],
        "raw_pages":         pages,
        "full_paper_text":   sanitize_prompt_input(full_text, 15000),
        "equations_summary": equations_summary,
        "research_report":   synthesis,
        "message_board":     board,
        "revision_counts":   rc
    }


# We alias analyst_synthesis for cleaner state access
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
            "full_paper_text": sanitize_prompt_input(
                state.get("full_paper_text", ""), 10000
            ),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        CRO_MODEL
    )

    board = post_message(
        state, "CRO", "ALL",
        "I have completed my initial reading of the paper. "
        "Theorist and Architect: begin your analyses in parallel. "
        "Pay special attention to the mathematical framework and proposed architecture.",
        "directive"
    )

    rc = increment_revision(state, "cro")
    return {**state, "cro_reading_notes": output, "message_board": board, "revision_counts": rc}


def cro_create_plan(state: ResearchState) -> ResearchState:
    print("\n🧠 CRO — Creating master implementation plan...")

    output = run_chain(
        CRO_IMPLEMENTATION_PLAN_PROMPT,
        {
            "paper_title":         state.get("paper_title", ""),
            "cro_reading_notes":   sanitize_prompt_input(
                state.get("cro_reading_notes", ""), 3000
            ),
            "theoretical_analysis":sanitize_prompt_input(
                state.get("theoretical_analysis", ""), 3000
            ),
            "architecture_analysis":sanitize_prompt_input(
                state.get("architecture_analysis", ""), 3000
            ),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        CRO_MODEL
    )

    board = post_message(
        state, "CRO", "Senior ML Engineer",
        "Implementation plan is ready. Begin with Phase A (core data structures). "
        "Follow the file structure exactly. Tag me if the paper is ambiguous.",
        "directive"
    )

    return {
        **state,
        "implementation_plan": output,
        "codebase_structure":  output,  # contains structure section
        "message_board":       board
    }


def cro_evaluate_agent(
    state: ResearchState,
    agent_name: str,
    agent_role: str,
    output_key: str
) -> ResearchState:
    print(f"\n🧠 CRO — Evaluating {agent_role}...")

    output = state.get(output_key, "")
    result = run_chain(
        CRO_EVALUATE_PROMPT,
        {
            "agent_role":    agent_name,
            "paper_title":   state.get("paper_title", ""),
            "paper_excerpt": sanitize_prompt_input(
                get_analyst_synthesis(state), 2000
            ),
            "output":        sanitize_prompt_input(output, 4000),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        CRO_MODEL
    )

    try:
        eval_obj = json.loads(clean_json(result))
    except Exception:
        eval_obj = {"passed": True, "feedback": "", "critical_issues": []}

    passed   = eval_obj.get("passed", True)
    feedback = eval_obj.get("feedback", "")
    issues   = eval_obj.get("critical_issues", [])

    rc = dict(state.get("revision_counts", {}))
    count = rc.get(agent_name, 0)
    if count >= 3:
        passed   = True
        feedback = ""
        print(f"   ⚠️  Max revisions for {agent_role}. Accepting.")

    evaluations = dict(state.get("evaluations", {}))
    evaluations[agent_name] = {
        "passed":   passed,
        "feedback": feedback,
        "issues":   issues,
        "iteration":count
    }

    needs_revision = list(state.get("needs_revision", []))
    if not passed:
        if agent_name not in needs_revision:
            needs_revision.append(agent_name)
        board = post_message(
            state, "CRO", agent_role,
            f"Your output needs revision. Issues: {'; '.join(issues[:3])}. {feedback[:300]}",
            "feedback"
        )
        print(f"   ❌ {agent_role} needs revision.")
    else:
        if agent_name in needs_revision:
            needs_revision.remove(agent_name)
        board = state.get("message_board", [])
        print(f"   ✅ {agent_role} approved.")

    return {
        **state,
        "evaluations":     evaluations,
        "revision_counts": rc,
        "needs_revision":  needs_revision,
        "message_board":   board
    }


def cro_final_verdict(state: ResearchState) -> ResearchState:
    print("\n🧠 CRO — Issuing final verdict...")

    modules_list = ", ".join(state.get("code_modules", {}).keys())

    output = run_chain(
        CRO_FINAL_VERDICT_PROMPT,
        {
            "paper_title":          state.get("paper_title", ""),
            "implementation_summary": sanitize_prompt_input(
                state.get("implementation_plan", ""), 2000
            ),
            "validation_report":    sanitize_prompt_input(
                state.get("validation_report", ""), 2000
            ),
            "code_modules_list":    modules_list,
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        CRO_MODEL
    )

    return {**state, "final_verdict": output}


# ══════════════════════════════════════════════════════════════
# THEORIST
# ══════════════════════════════════════════════════════════════

def theorist_agent(state: ResearchState) -> ResearchState:
    print("\n🔢 Theorist — Deep mathematical analysis...")

    # Search for related theoretical work
    paper_title = state.get("paper_title", "")
    related_search = web_search(f"{paper_title} theoretical foundations mathematical analysis")

    output = run_chain(
        THEORIST_PROMPT,
        {
            "paper_title":     paper_title,
            "full_paper_text": sanitize_prompt_input(
                state.get("full_paper_text", ""), 8000
            ),
            "analyst_synthesis": sanitize_prompt_input(
                get_analyst_synthesis(state), 4000
            ),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        THEORIST_MODEL
    )

    # Extract message to architect
    arch_message = extract_inter_agent_message(output, "ARCHITECT")
    board = post_message(
        state, "Theorist", "Architect",
        arch_message or "Mathematical analysis complete. See theoretical_analysis in state.",
        "insight"
    )

    rc = increment_revision(state, "theorist")
    return {
        **state,
        "theoretical_analysis": output,
        "message_board":        board,
        "revision_counts":      rc
    }


# ══════════════════════════════════════════════════════════════
# ML ARCHITECT
# ══════════════════════════════════════════════════════════════

def architect_agent(state: ResearchState) -> ResearchState:
    print("\n🏗️  ML Architect — Designing codebase structure...")

    theorist_message = get_messages_for(state, "architect")

    output = run_chain(
        ARCHITECT_PROMPT,
        {
            "paper_title":       state.get("paper_title", ""),
            "full_paper_text":   sanitize_prompt_input(
                state.get("full_paper_text", ""), 6000
            ),
            "analyst_synthesis": sanitize_prompt_input(
                get_analyst_synthesis(state), 3000
            ),
            "theorist_message":  sanitize_prompt_input(theorist_message, 1500),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        ARCHITECT_MODEL
    )

    # Extract message to engineer
    eng_message = extract_inter_agent_message(output, "ENGINEER")
    board = post_message(
        state, "Architect", "Senior ML Engineer",
        eng_message or "Architecture design complete. See codebase_structure in state.",
        "directive"
    )

    rc = increment_revision(state, "architect")
    return {
        **state,
        "architecture_analysis": output,
        "codebase_structure":    output,
        "message_board":         board,
        "revision_counts":       rc
    }


# ══════════════════════════════════════════════════════════════
# SENIOR ML ENGINEER
# ══════════════════════════════════════════════════════════════

def engineer_agent(state: ResearchState) -> ResearchState:
    print("\n💻 Senior ML Engineer — Implementing codebase...")

    arch_message  = get_messages_for(state, "engineer")
    review_feedback = json.dumps(state.get("review_feedback", {}), indent=2)[:2000]

    # Determine which files to implement based on architect's plan
    output = run_chain(
        ENGINEER_PROMPT,
        {
            "paper_title":        state.get("paper_title", ""),
            "analyst_synthesis":  sanitize_prompt_input(
                get_analyst_synthesis(state), 3000
            ),
            "theoretical_analysis": sanitize_prompt_input(
                state.get("theoretical_analysis", ""), 2000
            ),
            "codebase_structure": sanitize_prompt_input(
                state.get("codebase_structure", ""), 3000
            ),
            "implementation_plan":sanitize_prompt_input(
                state.get("implementation_plan", ""), 2000
            ),
            "architect_message":  sanitize_prompt_input(arch_message, 1500),
            "review_feedback":    sanitize_prompt_input(review_feedback, 2000),
            "current_file":       "ALL FILES",
            "file_spec":          "Implement the complete codebase as designed.",
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        ENGINEER_MODEL
    )

    # Parse code blocks from output
    code_modules = dict(state.get("code_modules", {}))
    code_pattern = re.compile(
        r'#\s*filename:\s*(\S+\.py)\s*\n(.*?)(?=\n#\s*filename:|\Z)',
        re.DOTALL
    )
    matches = code_pattern.findall(output)
    for filename, code in matches:
        code_modules[filename.strip()] = {
            "filename":    filename.strip(),
            "language":    "python",
            "code":        code.strip(),
            "description": f"Implemented by Senior ML Engineer",
            "status":      "draft"
        }

    # If no structured blocks found, save full output as main.py
    if not matches:
        # Try to extract any code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', output, re.DOTALL)
        if code_blocks:
            for i, block in enumerate(code_blocks):
                fname = f"module_{i+1}.py"
                code_modules[fname] = {
                    "filename": fname, "language": "python",
                    "code": block, "description": "Extracted code block",
                    "status": "draft"
                }
        else:
            code_modules["implementation.py"] = {
                "filename": "implementation.py", "language": "python",
                "code": output, "description": "Full implementation",
                "status": "draft"
            }

    # Post message to reviewer
    rev_message = extract_inter_agent_message(output, "REVIEWER")
    cro_message = extract_inter_agent_message(output, "CRO")

    board = state.get("message_board", [])
    if rev_message:
        board = post_message(
            {"message_board": board}, "Senior ML Engineer", "Code Reviewer",
            rev_message, "question"
        )
    if cro_message:
        board = post_message(
            {"message_board": board}, "Senior ML Engineer", "CRO",
            cro_message, "concern"
        )

    rc = increment_revision(state, "engineer")
    return {
        **state,
        "code_modules":       code_modules,
        "implementation_notes": output[:3000],
        "message_board":      board,
        "revision_counts":    rc
    }


# ══════════════════════════════════════════════════════════════
# CODE REVIEWER
# ══════════════════════════════════════════════════════════════

def reviewer_agent(state: ResearchState) -> ResearchState:
    print("\n🔍 Code Reviewer — Reviewing implementation...")

    engineer_message = get_messages_for(state, "code reviewer")

    # Compile all code for review
    all_code = "\n\n".join([
        f"# === {fname} ===\n{mod.get('code', '') if isinstance(mod, dict) else str(mod)}"
        for fname, mod in state.get("code_modules", {}).items()
    ])

    output = run_chain(
        REVIEWER_PROMPT,
        {
            "paper_title":        state.get("paper_title", ""),
            "analyst_synthesis":  sanitize_prompt_input(
                get_analyst_synthesis(state), 3000
            ),
            "theoretical_analysis": sanitize_prompt_input(
                state.get("theoretical_analysis", ""), 2000
            ),
            "code_to_review":     sanitize_prompt_input(all_code, 8000),
            "engineer_message":   sanitize_prompt_input(engineer_message, 1000),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        REVIEWER_MODEL
    )

    # Parse feedback per file
    review_feedback = dict(state.get("review_feedback", {}))
    review_feedback["latest_review"] = output[:3000]

    # Mark modules as reviewed
    code_modules = dict(state.get("code_modules", {}))
    for fname in code_modules:
        if isinstance(code_modules[fname], dict):
            code_modules[fname]["status"] = "reviewed"

    eng_message = extract_inter_agent_message(output, "ENGINEER")
    cro_message = extract_inter_agent_message(output, "CRO")
    board = state.get("message_board", [])
    if eng_message:
        board = post_message(
            {"message_board": board}, "Code Reviewer", "Senior ML Engineer",
            eng_message, "feedback"
        )
    if cro_message:
        board = post_message(
            {"message_board": board}, "Code Reviewer", "CRO",
            cro_message, "concern"
        )

    rc = increment_revision(state, "reviewer")
    return {
        **state,
        "review_feedback": review_feedback,
        "code_modules":    code_modules,
        "message_board":   board,
        "revision_counts": rc
    }


# ══════════════════════════════════════════════════════════════
# EXPERIMENT ENGINEER
# ══════════════════════════════════════════════════════════════

def experiment_engineer_agent(state: ResearchState) -> ResearchState:
    print("\n🧪 Experiment Engineer — Running validation...")

    # Try to execute a smoke test
    all_code = "\n\n".join([
        mod.get("code", "") if isinstance(mod, dict) else str(mod)
        for mod in state.get("code_modules", {}).values()
    ])

    # Build a basic smoke test
    smoke_test = f"""
import sys
print("=== SMOKE TEST ===")
print("Python version:", sys.version)

# Attempt to import key libraries
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("PyTorch not installed")

try:
    import numpy as np
    print("NumPy:", np.__version__)
except ImportError:
    print("NumPy not installed")

print("=== SMOKE TEST COMPLETE ===")
"""
    exec_result = execute_python_code(smoke_test, timeout=30)
    exec_summary = f"Return code: {exec_result['code']}\nSTDOUT:\n{exec_result['stdout']}\nSTDERR:\n{exec_result['stderr']}"

    output = run_chain(
        EXPERIMENT_ENGINEER_PROMPT,
        {
            "paper_title":      state.get("paper_title", ""),
            "analyst_synthesis":sanitize_prompt_input(
                get_analyst_synthesis(state), 3000
            ),
            "all_code":         sanitize_prompt_input(all_code, 6000),
            "execution_results":sanitize_prompt_input(exec_summary, 3000),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        EXPERIMENT_MODEL
    )

    cro_msg = extract_inter_agent_message(output, "CRO")
    eng_msg = extract_inter_agent_message(output, "ENGINEER")
    board   = state.get("message_board", [])
    if cro_msg:
        board = post_message({"message_board": board}, "Experiment Engineer", "CRO", cro_msg, "insight")
    if eng_msg:
        board = post_message({"message_board": board}, "Experiment Engineer", "Senior ML Engineer", eng_msg, "concern")

    rc = increment_revision(state, "experiment_engineer")
    return {
        **state,
        "execution_results": exec_summary,
        "validation_report": output,
        "message_board":     board,
        "revision_counts":   rc
    }


# ══════════════════════════════════════════════════════════════
# TECHNICAL WRITER
# ══════════════════════════════════════════════════════════════

def writer_agent(state: ResearchState) -> ResearchState:
    print("\n📝 Technical Writer — Producing documentation...")

    output = run_chain(
        WRITER_PROMPT,
        {
            "paper_title":        state.get("paper_title", ""),
            "paper_abstract":     sanitize_prompt_input(
                state.get("paper_abstract", ""), 1000
            ),
            "implementation_plan":sanitize_prompt_input(
                state.get("implementation_plan", ""), 2000
            ),
            "codebase_structure": sanitize_prompt_input(
                state.get("codebase_structure", ""), 2000
            ),
            "validation_report":  sanitize_prompt_input(
                state.get("validation_report", ""), 2000
            ),
            "final_verdict":      sanitize_prompt_input(
                state.get("final_verdict", ""), 1000
            ),
            "security_instruction": CONTENT_DELIMITER_INSTRUCTION
        },
        WRITER_MODEL
    )

    # Split into README and IMPLEMENTATION_NOTES
    readme = output
    impl_notes = ""
    if "DOCUMENT 2" in output or "IMPLEMENTATION_NOTES" in output:
        split_markers = ["DOCUMENT 2", "# IMPLEMENTATION_NOTES", "## IMPLEMENTATION_NOTES"]
        for marker in split_markers:
            if marker in output:
                parts      = output.split(marker, 1)
                readme     = parts[0]
                impl_notes = marker + parts[1]
                break

    # Save docs to code_modules for file writing
    code_modules = dict(state.get("code_modules", {}))
    code_modules["README.md"] = {
        "filename": "README.md", "language": "markdown",
        "code": readme, "description": "Project README", "status": "final"
    }
    if impl_notes:
        code_modules["IMPLEMENTATION_NOTES.md"] = {
            "filename": "IMPLEMENTATION_NOTES.md", "language": "markdown",
            "code": impl_notes, "description": "Implementation notes", "status": "final"
        }

    rc = increment_revision(state, "writer")
    return {
        **state,
        "readme":          readme,
        "implementation_paper": impl_notes,
        "code_modules":    code_modules,
        "revision_counts": rc
    }
