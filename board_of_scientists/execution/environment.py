"""Execution-environment helpers."""
from pathlib import Path

def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
