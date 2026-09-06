"""PDF implementation report generation."""

from __future__ import annotations

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer


def generate_implementation_report(data: dict, output_path: str) -> str:
    """Generate the implementation report PDF from structured report data."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#0d1117"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#555"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#0d1117"),
        spaceBefore=16,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=15,
        textColor=colors.HexColor("#222"),
        spaceAfter=6,
    )

    story = [
        Spacer(1, 3 * cm),
        Paragraph("AI RESEARCH IMPLEMENTATION TEAM", subtitle_style),
        Paragraph("Implementation Report", title_style),
        Spacer(1, 0.4 * cm),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0d1117")),
        Spacer(1, 0.4 * cm),
        Paragraph(data.get("paper_title", "Research Paper"), subtitle_style),
        Spacer(1, 1 * cm),
        Paragraph(
            f"Generated: {data.get('date', datetime.now().strftime('%Y-%m-%d'))}",
            subtitle_style,
        ),
        PageBreak(),
    ]

    for section in data.get("sections", []):
        story.append(Paragraph(section.get("title", "Section"), heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 0.3 * cm))
        content = (
            str(section.get("content", ""))
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        story.append(Paragraph(content, body_style))
        story.append(PageBreak())

    doc.build(story)
    return output_path


__all__ = ["generate_implementation_report"]
