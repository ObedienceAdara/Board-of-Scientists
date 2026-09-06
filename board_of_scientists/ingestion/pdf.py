"""PDF ingestion primitives.

PDF parsing belongs to the ingestion boundary. It must not depend on graph or
execution orchestration.
"""

from __future__ import annotations

import re
from typing import Any


def extract_pdf_pages(pdf_path: str) -> list[dict[str, Any]]:
    """Extract page-level text and lightweight structural signals from a PDF."""
    pages: list[dict[str, Any]] = []

    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                lower = text.lower()
                images = page.get_images(full=True)
                pages.append(
                    {
                        "page": page_num,
                        "text": text.strip(),
                        "has_figures": any(k in lower for k in ("figure", "fig.", "fig ")),
                        "has_tables": any(k in lower for k in ("table", "tab.")),
                        "has_equations": any(k in lower for k in ("equation", "eq.", "theorem", "proof", "lemma")),
                        "image_count": len(images),
                        "images": [f"[Image {i + 1} on page {page_num}]" for i in range(len(images))],
                        "char_count": len(text),
                    }
                )
        finally:
            doc.close()
        return pages
    except ImportError:
        pass

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                lower = text.lower()
                tables = page.extract_tables() or []
                pages.append(
                    {
                        "page": page_num,
                        "text": text.strip(),
                        "has_figures": "figure" in lower or "fig." in lower,
                        "has_tables": bool(tables),
                        "has_equations": any(k in lower for k in ("equation", "theorem", "proof")),
                        "image_count": 0,
                        "images": [],
                        "char_count": len(text),
                    }
                )
        return pages
    except ImportError:
        return [
            {
                "page": 1,
                "text": f"[Could not extract PDF: {pdf_path}]",
                "has_figures": False,
                "has_tables": False,
                "has_equations": False,
                "image_count": 0,
                "images": [],
                "char_count": 0,
            }
        ]


def get_paper_metadata(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract title, abstract, and likely section headings from page data."""
    if not pages:
        return {"title": "Unknown", "abstract": "", "sections": []}

    first_page_text = pages[0].get("text", "")
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    title = lines[0] if lines else "Unknown Paper"

    full_text = "\n".join(str(page.get("text", "")) for page in pages[:3])
    abstract = ""
    lower = full_text.lower()
    if "abstract" in lower:
        start = lower.find("abstract")
        end = lower.find("introduction", start)
        end = end if end != -1 else start + 2000
        abstract = full_text[start:end].strip()[:1500]

    heading_pattern = re.compile(r"^(\d+\.?\s+[A-Z][A-Za-z\s]+|[A-Z]{2,}[A-Z\s]+)$")
    sections = []
    for page in pages:
        for line in str(page.get("text", "")).splitlines():
            line = line.strip()
            if len(line) < 80 and heading_pattern.match(line):
                sections.append({"page": page.get("page"), "heading": line})

    return {"title": title[:200], "abstract": abstract, "sections": sections[:30]}


__all__ = ["extract_pdf_pages", "get_paper_metadata"]
