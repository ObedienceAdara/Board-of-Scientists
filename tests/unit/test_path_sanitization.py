from pathlib import Path

import pytest

from board_of_scientists.reports.provenance import ArtifactPathError, sanitize_relative_path, save_code_file


def test_safe_relative_paths_are_preserved():
    assert sanitize_relative_path("models/backbone.py") == "models/backbone.py"
    assert sanitize_relative_path("./models//backbone.py") == "models/backbone.py"
    assert sanitize_relative_path(r"models\backbone.py") == "models/backbone.py"


def test_traversal_and_absolute_paths_are_rejected():
    for value in (
        "../escape.py",
        "models/../../escape.py",
        "/tmp/escape.py",
        r"C:\\tmp\\escape.py",
        "bad\x00name.py",
        "",
    ):
        with pytest.raises(ArtifactPathError):
            sanitize_relative_path(value)


def test_saved_artifact_cannot_escape_output_root(tmp_path: Path):
    saved = Path(save_code_file(str(tmp_path), "models/model.py", "x = 1\n"))
    assert saved == (tmp_path / "models/model.py").resolve()
    assert saved.read_text(encoding="utf-8") == "x = 1\n"
    assert saved.is_relative_to(tmp_path.resolve())
