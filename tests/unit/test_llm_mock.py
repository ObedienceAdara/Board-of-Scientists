from langchain_core.runnables import RunnableLambda

from board_of_scientists.agents import _runtime
from board_of_scientists.schemas.agents import EvaluationResult


class FakeLLM:
    def with_structured_output(self, schema):
        return RunnableLambda(lambda _prompt: schema(passed=True))


def test_structured_chain_uses_mock_llm_without_credentials(monkeypatch):
    monkeypatch.setattr(_runtime, "make_llm", lambda *args, **kwargs: FakeLLM())

    result = _runtime.run_structured_chain(
        "Return an evaluation for {item}.",
        {"item": "test"},
        "fake-model",
        EvaluationResult,
    )

    assert isinstance(result, EvaluationResult)
    assert result.passed is True
