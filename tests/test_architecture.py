"""Architecture invariants.

These tests deliberately inspect source imports rather than executing the full
application. That makes the dependency contract cheap, deterministic, and safe
to run on every pull request.
"""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "board_of_scientists"

# Hard package dependency DAG. A package may always import itself, but may not
# import a package omitted from its allow-list.
ALLOWED_IMPORTS = {
    "graph": {"graph", "agents", "schemas", "reports"},
    "agents": {"agents", "schemas", "evidence", "ingestion", "execution"},
    "evidence": {"evidence", "schemas"},
    "ingestion": {"ingestion", "schemas"},
    "execution": {"execution", "schemas"},
    "reports": {"reports", "schemas", "evidence"},
    "schemas": {"schemas"},
}

FORBIDDEN_IMPORTS = {
    ("schemas", "agents"),
    ("schemas", "graph"),
    ("schemas", "reports"),
    ("schemas", "execution"),
    ("schemas", "ingestion"),
    ("schemas", "evidence"),
    ("evidence", "agents"),
    ("evidence", "graph"),
    ("ingestion", "graph"),
    ("ingestion", "agents"),
    ("ingestion", "execution"),
    ("ingestion", "reports"),
    ("execution", "graph"),
    ("execution", "agents"),
    ("execution", "ingestion"),
    ("execution", "reports"),
    ("reports", "graph"),
    ("reports", "agents"),
    ("reports", "ingestion"),
}


def _module_from_relative(current_package: str, level: int, module: str | None) -> str | None:
    """Resolve a relative import to its top-level board package name."""
    if level <= 0:
        return None
    parts = current_package.split(".")
    base = parts[: len(parts) - (level - 1)]
    if module:
        base.extend(module.split("."))
    if len(base) < 2 or base[0] != "board_of_scientists":
        return None
    return base[1]


def _imported_packages(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    package = "board_of_scientists." + ".".join(path.relative_to(PACKAGE_ROOT).parts[:-1])
    if package.endswith("."):
        package = package[:-1]

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                target = _module_from_relative(package, node.level, node.module)
            elif node.module and node.module.startswith("board_of_scientists."):
                target = node.module.split(".")[1]
            else:
                target = None
            if target:
                imported.add(target)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("board_of_scientists."):
                    imported.add(alias.name.split(".")[1])
    return imported


def test_dependency_dag_is_explicit_and_enforced():
    violations = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT)
        package = relative.parts[0]
        if package not in ALLOWED_IMPORTS:
            continue
        for imported in _imported_packages(path):
            if imported not in ALLOWED_IMPORTS[package]:
                violations.append(f"{relative}: {package} -> {imported} is not allowed")
            if (package, imported) in FORBIDDEN_IMPORTS:
                violations.append(f"{relative}: explicitly forbidden {package} -> {imported}")
    assert not violations, "\n".join(violations)


def test_expected_directory_structure():
    expected_dirs = {
        "graph",
        "agents",
        "evidence",
        "execution",
        "ingestion",
        "schemas",
        "reports",
    }
    actual = {path.name for path in PACKAGE_ROOT.iterdir() if path.is_dir() and path.name != "__pycache__"}
    assert expected_dirs <= actual


def test_domain_state_is_explicitly_bounded():
    from board_of_scientists.schemas.state import ResearchState

    assert set(ResearchState.__annotations__) == {
        "research_input",
        "paper_corpus",
        "analysis",
        "architecture",
        "implementation",
        "validation",
        "evidence",
        "communication",
        "output",
    }


def test_domain_state_round_trip_preserves_runtime_contract():
    from board_of_scientists.schemas.state import (
        create_initial_state,
        from_runtime_state,
        to_runtime_state,
    )

    state = create_initial_state("paper.pdf")
    state["research_input"].paper_title = "Test Paper"
    state["analysis"].theoretical_analysis = "equation"
    state["implementation"].code_modules = {
        "model.py": {
            "filename": "model.py",
            "language": "python",
            "code": "print('ok')",
            "description": "test",
            "status": "draft",
        }
    }

    runtime = to_runtime_state(state)
    rebuilt = from_runtime_state(runtime)

    assert rebuilt["research_input"].pdf_path == "paper.pdf"
    assert rebuilt["research_input"].paper_title == "Test Paper"
    assert rebuilt["analysis"].theoretical_analysis == "equation"
    assert "model.py" in rebuilt["implementation"].code_modules
