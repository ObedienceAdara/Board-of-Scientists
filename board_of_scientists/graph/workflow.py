"""Canonical workflow boundary.

The graph is deliberately isolated from business logic. During this migration
it delegates to the previously validated graph implementation kept under
``_legacy.main``; the public entry point is now stable and can be split into
nodes/routers without changing graph semantics.
"""
from board_of_scientists._legacy.main import build_research_graph, run_research_team, app
research_graph = build_research_graph()
__all__ = ["build_research_graph", "research_graph", "run_research_team", "app"]
