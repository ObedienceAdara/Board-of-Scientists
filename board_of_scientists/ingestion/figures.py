"""Figure discovery boundary.

The current implementation exposes figure metadata discovered by the PDF
extractor. Visual interpretation is intentionally a later phase.
"""

def summarize_figure_signals(pages):
    return [{"page": p.get("page"), "image_count": p.get("image_count", 0), "has_figures": p.get("has_figures", False)} for p in pages]
