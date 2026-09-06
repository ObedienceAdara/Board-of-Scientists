from board_of_scientists.ingestion.equations import extract_equations


def test_extract_equations_preserves_source_line_numbers():
    text = "plain text\nE = mc^2 (1)\nmore text\nsoftmax(x)"
    equations = extract_equations(text)
    assert equations == [
        {"line": 2, "content": "E = mc^2 (1)"},
        {"line": 4, "content": "softmax(x)"},
    ]


def test_extract_equations_ignores_non_math_lines():
    assert extract_equations("hello\nworld\n") == []
