from board_of_scientists.evidence.traceability import TraceEdge, TraceabilityGraph


def test_traceability_graph_validates_and_preserves_edge_order():
    graph = TraceabilityGraph(
        edges=[
            TraceEdge(source="claim:c1", relation="implemented_by", target="model.py", evidence="paper:p2"),
            TraceEdge(source="model.py", relation="validated_by", target="test:t1"),
        ]
    )
    assert [edge.relation for edge in graph.edges] == ["implemented_by", "validated_by"]
    assert graph.edges[0].source == "claim:c1"
