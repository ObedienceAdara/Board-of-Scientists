"""
============================================================
 AI Research Implementation Team — Multi-Agent System
 Stack: LangChain + LangGraph + LangSmith + LangServe
 LLM:   OpenRouter (configurable per agent)
 Input: PDF research paper path
 Output: Complete codebase + PDF report

 Install:
   pip install -r requirements.txt

 Run CLI:
   python main.py path/to/paper.pdf

 Run Server:
   python main.py serve
============================================================
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# LangSmith auto-tracing via env vars:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=your_key
#   LANGCHAIN_PROJECT=AIResearchTeam

from langgraph.graph import StateGraph, END
from state   import ResearchState
from agents  import (
    analyst_agent,
    cro_read_paper,
    cro_create_plan,
    cro_evaluate_agent,
    cro_final_verdict,
    theorist_agent,
    architect_agent,
    engineer_agent,
    reviewer_agent,
    experiment_engineer_agent,
    writer_agent,
)
from tools import save_all_modules, generate_implementation_report


# ══════════════════════════════════════════════════════════════
# GRAPH NODES
# ══════════════════════════════════════════════════════════════

def node_analyst(state):              return analyst_agent(state)
def node_cro_read(state):             return cro_read_paper(state)
def node_theorist(state):             return theorist_agent(state)
def node_architect(state):            return architect_agent(state)
def node_cro_plan(state):             return cro_create_plan(state)
def node_engineer(state):             return engineer_agent(state)
def node_reviewer(state):             return reviewer_agent(state)
def node_experiment(state):           return experiment_engineer_agent(state)
def node_writer(state):               return writer_agent(state)
def node_cro_verdict(state):          return cro_final_verdict(state)

def node_eval_analyst(state):
    return cro_evaluate_agent(state, "analyst",   "Paper Analyst",      "research_report")

def node_eval_theorist(state):
    return cro_evaluate_agent(state, "theorist",  "Theorist",           "theoretical_analysis")

def node_eval_architect(state):
    return cro_evaluate_agent(state, "architect", "ML Architect",       "architecture_analysis")

def node_eval_engineer(state):
    return cro_evaluate_agent(state, "engineer",  "Senior ML Engineer", "implementation_notes")

def node_eval_reviewer(state):
    return cro_evaluate_agent(state, "reviewer",  "Code Reviewer",      "review_feedback_str")

def node_eval_experiment(state):
    return cro_evaluate_agent(state, "experiment","Experiment Engineer", "validation_report")


def node_output(state: ResearchState) -> ResearchState:
    """Save all code files and generate PDF report."""
    print("\n📤 Saving all outputs...")

    # Prepare output directory
    paper_slug = (state.get("paper_title", "paper")
                  .lower()
                  .replace(" ", "_")
                  .replace("/", "_")[:40])
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{paper_slug}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save all code modules to disk
    saved = save_all_modules(output_dir, state.get("code_modules", {}))
    print(f"   💾 Saved {len(saved)} files to {output_dir}/")

    # Save message board log
    board_path = os.path.join(output_dir, "team_communications.json")
    with open(board_path, "w") as f:
        json.dump(state.get("message_board", []), f, indent=2)
    print("   💬 Team communications saved.")

    # Generate PDF report
    sections = [
        {"title": "Abstract & Overview",           "content": state.get("paper_abstract", "")},
        {"title": "CRO Reading Notes",             "content": state.get("cro_reading_notes", "")},
        {"title": "Theoretical Analysis",          "content": state.get("theoretical_analysis", "")},
        {"title": "Architecture Design",           "content": state.get("architecture_analysis", "")},
        {"title": "Implementation Plan",           "content": state.get("implementation_plan", "")},
        {"title": "Code Review Findings",          "content": json.dumps(state.get("review_feedback", {}), indent=2)},
        {"title": "Validation Report",             "content": state.get("validation_report", "")},
        {"title": "CRO Final Verdict",             "content": state.get("final_verdict", "")},
        {"title": "Team Communications Log",       "content": json.dumps(state.get("message_board", []), indent=2)},
    ]

    pdf_path = os.path.join(output_dir, "implementation_report.pdf")
    generate_implementation_report({
        "paper_title": state.get("paper_title", "Research Paper"),
        "date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sections":    sections
    }, pdf_path)

    print(f"\n{'='*60}")
    print("🏁 RESEARCH IMPLEMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"📁 Output directory: {output_dir}/")
    print(f"📄 PDF report:       {pdf_path}")
    print(f"📦 Code modules:     {len(state.get('code_modules', {}))} files")
    print(f"💬 Team messages:    {len(state.get('message_board', []))} exchanges")

    return {**state, "output_dir": output_dir, "pdf_report_path": pdf_path}


# ══════════════════════════════════════════════════════════════
# CONDITIONAL ROUTERS
# ══════════════════════════════════════════════════════════════

def _failed(state, name):
    evals = state.get("evaluations", {})
    return name in evals and not evals[name].get("passed", True)

def route_analyst(state):    return "analyst"    if _failed(state, "analyst")    else "cro_read"
def route_theorist(state):   return "theorist"   if _failed(state, "theorist")   else "architect"
def route_architect(state):  return "architect"  if _failed(state, "architect")  else "cro_plan"
def route_engineer(state):   return "engineer"   if _failed(state, "engineer")   else "reviewer"
def route_reviewer(state):   return "engineer"   if _failed(state, "reviewer")   else "experiment"
def route_experiment(state): return "engineer"   if _failed(state, "experiment") else "writer"
def route_writer(state):     return "writer"     if _failed(state, "writer")     else "cro_verdict"


# ══════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════

def build_research_graph():
    g = StateGraph(ResearchState)

    # Register all nodes
    g.add_node("analyst",        node_analyst)
    g.add_node("eval_analyst",   node_eval_analyst)
    g.add_node("cro_read",       node_cro_read)
    g.add_node("theorist",       node_theorist)
    g.add_node("eval_theorist",  node_eval_theorist)
    g.add_node("architect",      node_architect)
    g.add_node("eval_architect", node_eval_architect)
    g.add_node("cro_plan",       node_cro_plan)
    g.add_node("engineer",       node_engineer)
    g.add_node("eval_engineer",  node_eval_engineer)
    g.add_node("reviewer",       node_reviewer)
    g.add_node("eval_reviewer",  node_eval_reviewer)
    g.add_node("experiment",     node_experiment)
    g.add_node("eval_experiment",node_eval_experiment)
    g.add_node("writer",         node_writer)
    g.add_node("cro_verdict",    node_cro_verdict)
    g.add_node("output",         node_output)

    # Entry point
    g.set_entry_point("analyst")

    # ── Phase 0: Ingest ──────────────────────────────────────
    g.add_edge("analyst",        "eval_analyst")
    g.add_conditional_edges(
        "eval_analyst", route_analyst,
        {"analyst": "analyst", "cro_read": "cro_read"}
    )

    # ── CRO reads paper independently ───────────────────────
    g.add_edge("cro_read",       "theorist")     # Theorist goes after CRO reads

    # ── Phase 1: Theorist → Architect (sequential, each feeds next) ─
    g.add_edge("theorist",       "eval_theorist")
    g.add_conditional_edges(
        "eval_theorist", route_theorist,
        {"theorist": "theorist", "architect": "architect"}
    )

    g.add_edge("architect",      "eval_architect")
    g.add_conditional_edges(
        "eval_architect", route_architect,
        {"architect": "architect", "cro_plan": "cro_plan"}
    )

    # ── CRO synthesizes and creates plan ────────────────────
    g.add_edge("cro_plan",       "engineer")

    # ── Phase 2: Engineer → Reviewer loop ───────────────────
    g.add_edge("engineer",       "eval_engineer")
    g.add_conditional_edges(
        "eval_engineer", route_engineer,
        {"engineer": "engineer", "reviewer": "reviewer"}
    )

    g.add_edge("reviewer",       "eval_reviewer")
    g.add_conditional_edges(
        "eval_reviewer", route_reviewer,
        {"engineer": "engineer",    # Reviewer sends back to engineer
         "experiment": "experiment"}
    )

    # ── Phase 3: Validation ──────────────────────────────────
    g.add_edge("experiment",     "eval_experiment")
    g.add_conditional_edges(
        "eval_experiment", route_experiment,
        {"engineer": "engineer",    # Experiment finds bugs → back to engineer
         "writer": "writer"}
    )

    # ── Phase 4: Documentation + Verdict ─────────────────────
    g.add_edge("writer",         "cro_verdict")
    g.add_edge("cro_verdict",    "output")
    g.add_edge("output",         END)

    return g.compile()


research_graph = build_research_graph()


# ══════════════════════════════════════════════════════════════
# PUBLIC RUNNER
# ══════════════════════════════════════════════════════════════

def run_research_team(pdf_path: str) -> dict:
    """
    Run the full AI Research Implementation Team on a PDF paper.

    Args:
        pdf_path: Path to the research paper PDF

    Returns:
        dict with output_dir, pdf_report_path, final_verdict, code_modules
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n{'='*60}")
    print("🔬 AI RESEARCH IMPLEMENTATION TEAM — CONVENING")
    print(f"{'='*60}")
    print(f"📄 Paper: {pdf_path}")
    print(f"\nTeam:")
    print("  🧠 Dr. Aria Chen      — Chief Research Officer")
    print("  📖 Dr. Marcus Webb    — Paper Analyst")
    print("  🔢 Prof. Elena Vasquez— Theorist")
    print("  🏗️  Dr. James Okafor  — ML Architect")
    print("  💻 Dr. Kai Nakamura   — Senior ML Engineer")
    print("  🔍 Dr. Priya Sharma   — Code Reviewer")
    print("  🧪 Dr. Santiago Reyes — Experiment Engineer")
    print("  📝 Dr. Amara Osei     — Technical Writer")
    print(f"{'='*60}\n")

    initial_state = ResearchState(
        pdf_path=pdf_path,
        paper_title="",
        paper_abstract="",
        raw_pages=[],
        full_paper_text="",
        figures_summary="",
        tables_summary="",
        equations_summary="",
        theoretical_analysis="",
        architecture_analysis="",
        cro_reading_notes="",
        implementation_plan="",
        codebase_structure="",
        code_modules={},
        review_feedback={},
        implementation_notes="",
        execution_results="",
        validation_report="",
        discrepancies="",
        readme="",
        implementation_paper="",
        message_board=[],
        cro_directives={},
        evaluations={},
        revision_counts={},
        needs_revision=[],
        output_dir="",
        pdf_report_path="",
        final_verdict="",
        research_report=""
    )

    final_state = research_graph.invoke(initial_state)

    return {
        "output_dir":      final_state["output_dir"],
        "pdf_report_path": final_state["pdf_report_path"],
        "final_verdict":   final_state["final_verdict"],
        "paper_title":     final_state["paper_title"],
        "modules_count":   len(final_state["code_modules"]),
        "messages_count":  len(final_state["message_board"]),
        "revision_summary":final_state["revision_counts"]
    }


# ══════════════════════════════════════════════════════════════
# LANGSERVE — REST API
# ══════════════════════════════════════════════════════════════

from fastapi               import FastAPI
from langserve             import add_routes
from langchain_core.runnables import RunnableLambda

app = FastAPI(
    title="AI Research Implementation Team",
    description="8-agent AI system that reads and implements ML research papers",
    version="1.0.0"
)

research_runnable = RunnableLambda(
    lambda inputs: run_research_team(inputs["pdf_path"])
)

add_routes(app, research_runnable, path="/implement-paper")

@app.get("/")
async def root():
    return {
        "status":  "running",
        "system":  "AI Research Implementation Team",
        "team": [
            "CRO (Dr. Aria Chen)",
            "Paper Analyst (Dr. Marcus Webb)",
            "Theorist (Prof. Elena Vasquez)",
            "ML Architect (Dr. James Okafor)",
            "Senior ML Engineer (Dr. Kai Nakamura)",
            "Code Reviewer (Dr. Priya Sharma)",
            "Experiment Engineer (Dr. Santiago Reyes)",
            "Technical Writer (Dr. Amara Osei)"
        ],
        "playground": "/implement-paper/playground",
        "docs":       "/docs"
    }


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        print("\n🌐 Starting Research Team API...")
        print("   Playground → http://localhost:8000/implement-paper/playground")
        print("   Docs       → http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)

    elif len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result   = run_research_team(pdf_path)
        print(f"\n📊 SUMMARY:")
        print(f"   Paper:    {result['paper_title'][:60]}")
        print(f"   Modules:  {result['modules_count']} files generated")
        print(f"   Messages: {result['messages_count']} team exchanges")
        print(f"   Output:   {result['output_dir']}/")

    else:
        print("Usage:")
        print("  python main.py path/to/paper.pdf")
        print("  python main.py serve")
