"""Artifact and message persistence."""
import json
from pathlib import Path
from ..execution._runtime import save_all_modules

def save_message_board(output_dir: str, messages: list) -> str:
    path = Path(output_dir) / "team_communications.json"
    path.write_text(json.dumps(messages, indent=2), encoding="utf-8")
    return str(path)

__all__ = ["save_all_modules", "save_message_board"]
