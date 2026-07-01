"""Part 3 — the agent's routing decision.

`route_after_agent` is the conditional edge: it decides tool-vs-direct purely
from the last message, so it is testable without invoking any model.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END

from src.agent import looks_like_tool_json, route_after_agent


def _state(msg: AIMessage) -> dict:
    return {"messages": [msg]}


def test_routes_to_tools_when_model_emits_tool_calls():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "rag_search", "args": {"query": "x"}, "id": "1", "type": "tool_call"}],
    )
    assert route_after_agent(_state(msg)) == "tools"


def test_routes_to_end_on_a_plain_answer():
    assert route_after_agent(_state(AIMessage(content="Paris is the capital."))) == END


def test_guard_flags_raw_tool_call_json():
    assert looks_like_tool_json('{"name": "wikipedia", "parameters": {"query": "x"}}')
    assert looks_like_tool_json('  {"function": "search"}')


def test_guard_leaves_natural_answers_alone():
    assert not looks_like_tool_json("The capital of France is Paris.")
    assert not looks_like_tool_json("")
