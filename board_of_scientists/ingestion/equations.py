"""Equation extraction primitives."""

from __future__ import annotations

import re


def extract_equations(text: str) -> list[dict[str, str | int]]:
    """Extract equation-like lines and mathematical expressions from text."""
    equations: list[dict[str, str | int]] = []
    numbered = re.compile(r"\((\d+)\)")
    math_keywords = (
        "∀", "∃", "∑", "∏", "∫", "→", "←", "⟹", "≤", "≥",
        "argmax", "argmin", "softmax", "sigmoid", "relu",
        "\\mathcal", "\\mathbb", "\\frac", "\\sum", "\\prod",
    )

    for line_number, line in enumerate(text.splitlines(), start=1):
        if numbered.search(line) or any(keyword in line for keyword in math_keywords):
            equations.append({"line": line_number, "content": line.strip()})

    return equations


__all__ = ["extract_equations"]
