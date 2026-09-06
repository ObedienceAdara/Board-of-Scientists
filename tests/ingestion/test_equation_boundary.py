from board_of_scientists.ingestion.equations import extract_equations


def test_ingestion_equation_boundary_is_deterministic():
    assert extract_equations(r"Loss = -\sum_i y_i log(p_i)") == [
        {"line": 1, "content": r"Loss = -\sum_i y_i log(p_i)"}
    ]
