"""Verify the Part 3 agent ROUTING without a tool-capable model.

gemma3:1b can't tool-call and llama3.2:3b isn't pulled yet, so we stub only the
chat model — the part that decides — while using the REAL rag_search tool (real
FAISS + nomic retrieval, which works locally). This proves both branches of the
graph: a knowledge question routes through the tool; a general question answers
directly. When llama3.2:3b is available, the same graph runs unchanged with the
real model.

Run from the repo root:
    python tests/test_agent_routing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.agent import answer, build_agent  # noqa: E402
from src.config import settings  # noqa: E402
from src.tools import rag_search  # noqa: E402

AGENT_TERMS = ("agent", "react", "rag", "tool calling", "memory", "multi-agent")


class FakeToolCallingLLM:
    """Mimics what a real tool-calling model would decide, so we can exercise the
    graph: call the tool for AI-agent questions, answer directly otherwise."""

    def invoke(self, messages) -> AIMessage:
        last = messages[-1]
        # We just received a tool result -> produce the final grounded answer.
        if isinstance(last, ToolMessage):
            if last.content == "NO_RELEVANT_CONTEXT":
                return AIMessage(content="I couldn't find anything relevant in the knowledge base.")
            return AIMessage(content="ReAct interleaves reasoning (thoughts) and acting (tool calls).")
        # Otherwise decide based on the user's question.
        question = next((m.content for m in messages if isinstance(m, HumanMessage)), "").lower()
        if any(term in question for term in AGENT_TERMS):
            return AIMessage(
                content="",
                tool_calls=[{"name": "rag_search", "args": {"query": question}, "id": "call_1", "type": "tool_call"}],
            )
        return AIMessage(content="The capital of France is Paris.")


def main() -> int:
    # Fresh trace for this run.
    Path(settings.agent_trace_log).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.agent_trace_log).write_text("", encoding="utf-8")

    agent = build_agent(FakeToolCallingLLM(), [rag_search])

    knowledge_q = "What is the ReAct pattern in AI agents?"
    general_q = "What is the capital of France?"

    a1 = answer(knowledge_q, agent=agent)
    a2 = answer(general_q, agent=agent)

    trace = Path(settings.agent_trace_log).read_text(encoding="utf-8")

    checks = [
        ("knowledge Q invoked the tool", "TOOL 'rag_search'" in trace),
        ("knowledge Q answer is grounded", "ReAct" in a1),
        ("general Q skipped the tool", "DECISION: answer directly" in trace),
        ("general Q answered directly", "Paris" in a2),
        ("general Q did NOT call the tool", trace.count("TOOL 'rag_search'") == 1),
    ]

    print("=== agent routing checks ===")
    ok = True
    for label, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
        ok = ok and passed

    print(f"\nknowledge Q -> {a1!r}")
    print(f"general Q   -> {a2!r}")
    print(f"\n(full trace in {settings.agent_trace_log})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
