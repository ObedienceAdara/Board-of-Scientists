from board_of_scientists.ingestion.equations import extract_equations
from board_of_scientists.ingestion.pdf import extract_pdf_pages, get_paper_metadata
from tests.fixtures.pdf_factory import create_sample_paper


def test_pdf_ingestion_extracts_pages_and_metadata(tmp_path):
    pdf = create_sample_paper(tmp_path / "paper.pdf")
    pages = extract_pdf_pages(str(pdf))

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "Deterministic Test Paper" in pages[0]["text"]
    assert pages[0]["has_figures"] is True
    assert pages[0]["has_equations"] is True
    assert pages[1]["has_tables"] is True

    metadata = get_paper_metadata(pages)
    assert metadata["title"] == "Deterministic Test Paper"
    assert "test abstract" in metadata["abstract"].lower()
    assert metadata["sections"]


def test_ingested_text_can_feed_equation_parser(tmp_path):
    pdf = create_sample_paper(tmp_path / "paper.pdf")
    pages = extract_pdf_pages(str(pdf))
    text = "\n".join(page["text"] for page in pages)
    equations = extract_equations(text)
    assert any(eq["content"] == "E = mc^2 (1)" for eq in equations)
    assert any("softmax" in eq["content"] for eq in equations)
