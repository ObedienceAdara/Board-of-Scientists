"""Canonical LangGraph workflow and HTTP control plane for Board of Scientists."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from ..agents.registry import (
    ANALYST,
    ARCHITECT,
    ENGINEER,
    EXPERIMENT,
    REVIEWER,
    THEORIST,
    WRITER,
)
from ..schemas.state import ResearchState, create_initial_state
from .job_manager import build_default_job_manager
from .nodes import (
    node_analyst,
    node_architect,
    node_cro_plan,
    node_cro_read,
    node_cro_verdict,
    node_engineer,
    node_eval_analyst,
    node_eval_architect,
    node_eval_engineer,
    node_eval_experiment,
    node_eval_reviewer,
    node_eval_theorist,
    node_eval_writer,
    node_experiment,
    node_output,
    node_reviewer,
    node_theorist,
    node_writer,
)
from .routers import (
    route_analyst,
    route_architect,
    route_engineer,
    route_experiment,
    route_reviewer,
    route_theorist,
    route_writer,
)

load_dotenv()


def build_research_graph():
    """Compile the complete research implementation workflow."""
    graph = StateGraph(ResearchState)

    graph.add_node("analyst", node_analyst)
    graph.add_node("eval_analyst", node_eval_analyst)
    graph.add_node("cro_read", node_cro_read)
    graph.add_node("theorist", node_theorist)
    graph.add_node("eval_theorist", node_eval_theorist)
    graph.add_node("architect", node_architect)
    graph.add_node("eval_architect", node_eval_architect)
    graph.add_node("cro_plan", node_cro_plan)
    graph.add_node("engineer", node_engineer)
    graph.add_node("eval_engineer", node_eval_engineer)
    graph.add_node("reviewer", node_reviewer)
    graph.add_node("eval_reviewer", node_eval_reviewer)
    graph.add_node("experiment", node_experiment)
    graph.add_node("eval_experiment", node_eval_experiment)
    graph.add_node("writer", node_writer)
    graph.add_node("eval_writer", node_eval_writer)
    graph.add_node("cro_verdict", node_cro_verdict)
    graph.add_node("output", node_output)

    graph.set_entry_point("analyst")

    graph.add_edge("analyst", "eval_analyst")
    graph.add_conditional_edges("eval_analyst", route_analyst, {"analyst": "analyst", "cro_read": "cro_read"})
    graph.add_edge("cro_read", "theorist")
    graph.add_edge("theorist", "eval_theorist")
    graph.add_conditional_edges("eval_theorist", route_theorist, {"theorist": "theorist", "architect": "architect"})
    graph.add_edge("architect", "eval_architect")
    graph.add_conditional_edges("eval_architect", route_architect, {"architect": "architect", "cro_plan": "cro_plan"})
    graph.add_edge("cro_plan", "engineer")
    graph.add_edge("engineer", "eval_engineer")
    graph.add_conditional_edges("eval_engineer", route_engineer, {"engineer": "engineer", "reviewer": "reviewer"})
    graph.add_edge("reviewer", "eval_reviewer")
    graph.add_conditional_edges("eval_reviewer", route_reviewer, {"engineer": "engineer", "experiment": "experiment"})
    graph.add_edge("experiment", "eval_experiment")
    graph.add_conditional_edges("eval_experiment", route_experiment, {"engineer": "engineer", "writer": "writer"})
    graph.add_edge("writer", "eval_writer")
    graph.add_conditional_edges("eval_writer", route_writer, {"writer": "writer", "cro_verdict": "cro_verdict"})
    graph.add_edge("cro_verdict", "output")
    graph.add_edge("output", END)
    return graph.compile()


research_graph = build_research_graph()


def _initial_state(pdf_path: str) -> ResearchState:
    """Create the canonical domain state for a new research run."""
    return create_initial_state(pdf_path)


def run_research_team(pdf_path: str) -> dict:
    """Run the full research implementation workflow for a validated PDF file."""
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Research input must be a .pdf file.")

    final_state = research_graph.invoke(_initial_state(str(path)))
    communication = final_state["communication"]
    implementation = final_state["implementation"]
    output = final_state["output"]
    return {
        "output_dir": output.output_dir,
        "pdf_report_path": output.pdf_report_path,
        "final_verdict": output.final_verdict,
        "paper_title": final_state["research_input"].paper_title,
        "modules_count": len(implementation.code_modules),
        "messages_count": len(communication.message_board),
        "revision_summary": communication.revision_counts,
    }


class ImplementPaperRequest(BaseModel):
    """Request referencing a PDF already staged in UPLOADS_DIR."""

    pdf_filename: str = Field(min_length=1, max_length=255)


UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "./uploads")).resolve()
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()

app = FastAPI(
    title="AI Research Implementation Team",
    description="8-agent AI system that reads and implements ML research papers",
    version="1.2.0",
)
job_manager = build_default_job_manager(run_research_team)


def _authorize(x_api_key: str) -> None:
    """Fail closed in production; allow explicit local development mode."""
    if APP_ENV == "production" and not API_AUTH_TOKEN:
        raise HTTPException(status_code=503, detail="API_AUTH_TOKEN is required in production.")
    if API_AUTH_TOKEN and x_api_key != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _resolve_upload(pdf_filename: str) -> Path:
    """Resolve a basename-only PDF reference inside UPLOADS_DIR."""
    safe_name = Path(pdf_filename).name
    if safe_name != pdf_filename or Path(pdf_filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="pdf_filename must be a plain .pdf filename.")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    resolved = (UPLOADS_DIR / safe_name).resolve()
    if resolved == UPLOADS_DIR or UPLOADS_DIR not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid pdf_filename.")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"No such PDF in uploads directory: {safe_name}")
    return resolved


@app.post("/implement-paper", status_code=status.HTTP_202_ACCEPTED)
def implement_paper(req: ImplementPaperRequest, x_api_key: str = Header(default="")):
    """Queue a research run and return immediately with a job identifier."""
    _authorize(x_api_key)
    resolved = _resolve_upload(req.pdf_filename)
    job = job_manager.submit(str(resolved))
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "status_url": f"/jobs/{job.job_id}",
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str = Header(default="")):
    """Return status and result information for a submitted research job."""
    _authorize(x_api_key)
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "result": job.result,
        "error": job.error,
    }


@app.get("/")
async def root():
    """Return service metadata and the team roster."""
    return {
        "status": "running",
        "system": "AI Research Implementation Team",
        "team": [
            "CRO (Dr. Aria Chen)",
            "Paper Analyst (Dr. Marcus Webb)",
            "Theorist (Prof. Elena Vasquez)",
            "ML Architect (Dr. James Okafor)",
            "Senior ML Engineer (Dr. Kai Nakamura)",
            "Code Reviewer (Dr. Priya Sharma)",
            "Experiment Engineer (Dr. Santiago Reyes)",
            "Technical Writer (Dr. Amara Osei)",
        ],
        "usage": "POST /implement-paper with {'pdf_filename': '<name>'}, then poll GET /jobs/{job_id}",
        "docs": "/docs",
    }


__all__ = ["build_research_graph", "research_graph", "run_research_team", "app", "job_manager"]
