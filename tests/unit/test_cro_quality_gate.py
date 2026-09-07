from board_of_scientists.agents import cro


def test_cro_evaluation_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(cro, "_run_structured", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider unavailable")))
    state = {
        "paper_title": "Test Paper",
        "research_report": "ground truth",
        "review_summary": "candidate output",
        "evaluations": {},
        "revision_counts": {},
        "needs_revision": [],
        "message_board": [],
    }

    result = cro.cro_evaluate_agent(state, "reviewer", "review_summary")

    assert result["evaluations"]["reviewer"]["passed"] is False
    assert "Evaluator failure" in result["evaluations"]["reviewer"]["issues"][0]
    assert "reviewer" in result["needs_revision"]


def test_cro_evaluation_success_can_approve(monkeypatch):
    class Result:
        passed = True
        feedback = ""
        critical_issues = []

    monkeypatch.setattr(cro, "_run_structured", lambda *args, **kwargs: Result())
    state = {
        "paper_title": "Test Paper",
        "research_report": "ground truth",
        "review_summary": "candidate output",
        "evaluations": {},
        "revision_counts": {},
        "needs_revision": [],
        "message_board": [],
    }

    result = cro.cro_evaluate_agent(state, "reviewer", "review_summary")

    assert result["evaluations"]["reviewer"]["passed"] is True
    assert result["needs_revision"] == []
