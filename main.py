"""Compatibility CLI/API launcher for Board of Scientists."""

import os
import sys

from board_of_scientists.graph.workflow import app, run_research_team


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn

        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host=host, port=port)
    elif len(sys.argv) > 1:
        result = run_research_team(sys.argv[1])
        print(result)
    else:
        print("Usage: python main.py path/to/paper.pdf | python main.py serve")
