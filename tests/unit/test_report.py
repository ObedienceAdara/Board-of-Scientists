from pathlib import Path

from board_of_scientists.reports.pdf import generate_implementation_report
from board_of_scientists.reports.provenance import save_all_modules, save_message_board


def test_report_generation_creates_nonempty_pdf(tmp_path: Path):
    output = tmp_path / "report.pdf"
    result = generate_implementation_report(
        {
            "paper_title": "Test Paper",
            "date": "2026-09-06",
            "sections": [
                {"title": "Summary", "content": "Measured value: 42\nA & B"},
            ],
        },
        str(output),
    )
    assert result == str(output)
    assert output.exists()
    assert output.stat().st_size > 100
    assert output.read_bytes().startswith(b"%PDF")


def test_provenance_persists_modules_and_message_board(tmp_path: Path):
    saved = save_all_modules(
        str(tmp_path),
        {
            "src/model.py": {"code": "x = 1\n"},
            "README.md": "# test\n",
        },
    )
    assert len(saved) == 2
    assert (tmp_path / "src/model.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# test\n"

    board = save_message_board(str(tmp_path), [{"sender": "analyst", "content": "done"}])
    assert Path(board).exists()
    assert '"sender": "analyst"' in Path(board).read_text(encoding="utf-8")
