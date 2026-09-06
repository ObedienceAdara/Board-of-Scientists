"""Compatibility CLI/API launcher for Board of Scientists.

The application implementation lives under ``board_of_scientists``. This file
exists only so the established commands remain valid while the package layout
becomes the canonical import surface.
"""

from board_of_scientists.graph.workflow import app, run_research_team


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif len(sys.argv) > 1:
        result = run_research_team(sys.argv[1])
        print(result)
    else:
        print("Usage: python main.py path/to/paper.pdf | python main.py serve")
