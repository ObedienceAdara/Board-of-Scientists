from pathlib import Path

from board_of_scientists.reports.provenance import sanitize_relative_path, save_code_file


def test_safe_relative_paths_are_preserved():
    assert sanitize_relative_path("models/backbone.py") == "models/backbone.py"
    assert sanitize_relative_path("./models//backbone.py") == "models/backbone.py"
    assert sanitize_relative_path(r"models\backbone.py") == "models/backbone.py"


def test_traversal_and_absolute_paths_fall_back_safely():
    assert sanitize_relative_path("../escape.py") == "unnamed_module.py"
    assert sanitize_relative_path("models/../../escape.py") == "unnamed_module.py"
    assert sanitize_relative_path("/tmp/escape.py") == "tmp/escape.py"
    assert sanitize_relative_path("C:\\tmp\\escape.py") == "tmp/escape.py"
    assert sanitize_relative_path("bad\x00name.py") == "unnamed_module.py"


def test_saved_artifact_cannot_escape_output_root(tmp_path: Path):
    saved = Path(save_code_file(str(tmp_path), "models/model.py", "x = 1\n"))
    assert saved == (tmp_path / "models/model.py").resolve()
    assert saved.read_text(encoding="utf-8") == "x = 1\n"
    assert saved.is_relative_to(tmp_path.resolve())
