import importlib
from pathlib import Path


def test_target_architecture_modules_exist():
    expected = [
        "board_of_scientists.graph.workflow",
        "board_of_scientists.graph.routers",
        "board_of_scientists.graph.nodes",
        "board_of_scientists.agents.analyst",
        "board_of_scientists.agents.theorist",
        "board_of_scientists.agents.architect",
        "board_of_scientists.agents.engineer",
        "board_of_scientists.agents.reviewer",
        "board_of_scientists.agents.experiment",
        "board_of_scientists.agents.writer",
        "board_of_scientists.agents.cro",
        "board_of_scientists.evidence.claims",
        "board_of_scientists.evidence.equations",
        "board_of_scientists.evidence.traceability",
        "board_of_scientists.evidence.consistency",
        "board_of_scientists.execution.sandbox",
        "board_of_scientists.execution.environment",
        "board_of_scientists.execution.experiments",
        "board_of_scientists.ingestion.pdf",
        "board_of_scientists.ingestion.figures",
        "board_of_scientists.ingestion.equations",
        "board_of_scientists.ingestion.tables",
        "board_of_scientists.schemas.agents",
        "board_of_scientists.schemas.state",
        "board_of_scientists.schemas.evidence",
        "board_of_scientists.reports.pdf",
        "board_of_scientists.reports.provenance",
    ]
    for name in expected:
        importlib.import_module(name)


def test_expected_directory_structure():
    root = Path(__file__).parents[1]
    expected_dirs = [
        "board_of_scientists/graph",
        "board_of_scientists/agents",
        "board_of_scientists/evidence",
        "board_of_scientists/execution",
        "board_of_scientists/ingestion",
        "board_of_scientists/schemas",
        "board_of_scientists/reports",
        "tests",
    ]
    for relative in expected_dirs:
        assert (root / relative).is_dir(), relative
