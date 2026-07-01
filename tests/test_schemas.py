"""Bonus 1 — the structured-output schema and extraction wiring.

The schema's constraints are tested directly against Pydantic; the extraction
function is tested with a stub model, so neither test needs a live Ollama.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import QueryAnalysis, analyze_query


def test_sentiment_is_constrained_to_the_enum():
    QueryAnalysis(topics=["ai agents"], sentiment="neutral")  # ok
    with pytest.raises(ValidationError):
        QueryAnalysis(topics=["ai agents"], sentiment="angry")


def test_topics_must_be_non_empty_and_bounded():
    with pytest.raises(ValidationError):
        QueryAnalysis(topics=[], sentiment="neutral")
    with pytest.raises(ValidationError):
        QueryAnalysis(topics=["a", "b", "c", "d", "e", "f"], sentiment="neutral")


class _StubStructured:
    def __init__(self, result: QueryAnalysis):
        self._result = result

    def invoke(self, _messages) -> QueryAnalysis:
        return self._result


class _StubModel:
    """Stands in for ChatOllama: `with_structured_output` returns a runnable
    whose `invoke` yields an already-parsed QueryAnalysis."""

    def __init__(self, result: QueryAnalysis):
        self._result = result

    def with_structured_output(self, schema):
        assert schema is QueryAnalysis
        return _StubStructured(self._result)


def test_analyze_query_returns_validated_model():
    expected = QueryAnalysis(topics=["react", "agent loops"], sentiment="neutral")
    out = analyze_query("How does the ReAct pattern work?", model=_StubModel(expected))
    assert isinstance(out, QueryAnalysis)
    assert out == expected
