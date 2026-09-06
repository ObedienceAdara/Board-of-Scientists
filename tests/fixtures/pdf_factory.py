from pathlib import Path

from reportlab.pdfgen import canvas


def create_sample_paper(path: Path) -> Path:
    """Create a small deterministic PDF representative of an academic paper."""
    pdf = canvas.Canvas(str(path))
    pdf.setTitle("Deterministic Test Paper")
    pdf.drawString(72, 760, "Deterministic Test Paper")
    pdf.drawString(72, 735, "Abstract")
    pdf.drawString(72, 718, "This is a test abstract for the ingestion pipeline.")
    pdf.drawString(72, 690, "1 Introduction")
    pdf.drawString(72, 670, "We evaluate a deterministic method.")
    pdf.drawString(72, 642, "E = mc^2 (1)")
    pdf.drawString(72, 620, "Figure 1 shows the measured result.")
    pdf.showPage()
    pdf.drawString(72, 760, "2 Method")
    pdf.drawString(72, 735, "The model uses softmax(x) and a simple baseline.")
    pdf.drawString(72, 710, "Table 1 contains the evaluation numbers.")
    pdf.save()
    return path
