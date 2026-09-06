from board_of_scientists.agents import _runtime
from board_of_scientists.schemas.agents import FileSpec


def test_manifest_is_sanitized_and_deduplicated():
    manifest = [
        FileSpec(filename="models\\model.py", description="model"),
        FileSpec(filename="models/model.py", description="duplicate"),
        FileSpec(filename="../escape.py", description="unsafe"),
    ]
    result = _runtime._sanitize_manifest(manifest)
    assert result == [
        {"filename": "models/model.py", "description": "model", "depends_on": [], "group": ""}
    ]


def test_manifest_filters_unknown_and_self_dependencies():
    result = _runtime._sanitize_manifest(
        [
            FileSpec(filename="base.py", description="base"),
            FileSpec(filename="model.py", description="model", depends_on=["base.py", "missing.py", "model.py"]),
        ]
    )
    assert result[1]["depends_on"] == ["base.py"]


def test_build_batches_preserves_dependency_order_and_limits_groups():
    manifest = [
        {"filename": "a.py", "group": ""},
        {"filename": "b.py", "group": "pair"},
        {"filename": "c.py", "group": "pair"},
        {"filename": "d.py", "group": ""},
    ]
    batches = _runtime._build_batches(manifest)
    assert [[item["filename"] for item in batch] for batch in batches] == [
        ["a.py"], ["b.py", "c.py"], ["d.py"]
    ]
