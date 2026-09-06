"""Table discovery boundary."""

def extract_table_signals(pages):
    return [{"page": p.get("page"), "has_tables": p.get("has_tables", False)} for p in pages]
