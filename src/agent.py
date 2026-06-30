"""Part 3 — the agentic orchestrator, built as a small explicit LangGraph.

The graph is deliberately minimal so the control flow is readable at a glance:

        START → agent ──(tool_calls?)──► tools ──┐
                  ▲                               │
                  └───────────────────────────────┘
                  │
                  └──(no tool_calls)──► END

`agent`  : the LLM decides — emit a tool call, or a final answer.
`tools`  : execute the requested tool(s), feed results back, loop to `agent`.
The conditional edge after `agent` is the "use a tool vs. answer directly"
decision the assignment asks for. Every decision and tool I/O is written to
logs/agent_trace.log (the Part 3 deliverable).

CLI:
    python -m src.agent "What is the ReAct pattern in AI agents?"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import settings

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant with access to a tool, `rag_search`, that "
        "searches a local knowledge base about AI agents (agent loops, tool "
        "calling, ReAct, RAG, agent memory, multi-agent systems).\n"
        "- For questions about AI agents, call `rag_search` and ground your answer "
        "in the returned passages.\n"
        "- For general questions unrelated to AI agents, answer directly from your "
        "own knowledge WITHOUT calling the tool.\n"
        "- If `rag_search` returns NO_RELEVANT_CONTEXT, answer directly and say the "
        "knowledge base had nothing relevant."
    )
)

# Bound on agent/tool turns so a confused model cannot loop forever.
RECURSION_LIMIT = 8


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --------------------------------------------------------------------------- #
# Trace logging (the Part 3 deliverable)
# --------------------------------------------------------------------------- #
def _get_logger() -> logging.Logger:
    Path(settings.agent_trace_log).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("agent_trace")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(settings.agent_trace_log, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _trace(msg: str) -> None:
    _get_logger().info(msg)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_agent(llm_with_tools, tools: list[BaseTool]):
    """Compile the agent graph.

    `llm_with_tools` is any runnable returning an AIMessage (typically
    `get_chat_model().bind_tools(tools)`); injecting it keeps the graph testable
    with a stub model. `tools` are the executable tools, keyed by name.
    """
    tool_map = {t.name: t for t in tools}

    def agent_node(state: AgentState) -> dict:
        response: AIMessage = llm_with_tools.invoke(state["messages"])
        if response.tool_calls:
            for tc in response.tool_calls:
                _trace(f"  DECISION: call tool '{tc['name']}' args={tc['args']}")
        else:
            preview = (response.content or "").replace("\n", " ")[:160]
            _trace(f"  DECISION: answer directly -> {preview!r}")
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        outputs: list[ToolMessage] = []
        for tc in last.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            preview = str(result).replace("\n", " ")[:160]
            _trace(f"  TOOL '{tc['name']}' input={tc['args']} -> {preview!r}")
            outputs.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
            )
        return {"messages": outputs}

    def route(state: AgentState) -> str:
        # The "use a tool vs. answer directly" decision.
        return "tools" if state["messages"][-1].tool_calls else END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


def default_agent():
    """Wire the real ChatOllama + rag_search tool into the graph."""
    from .llm import get_chat_model
    from .tools import rag_search

    tools: list[BaseTool] = [rag_search]
    llm_with_tools = get_chat_model().bind_tools(tools)
    return build_agent(llm_with_tools, tools)


# --------------------------------------------------------------------------- #
# Run helper + CLI
# --------------------------------------------------------------------------- #
def answer(question: str, agent=None) -> str:
    """Run the agent on one question, log the trace, return the final answer."""
    agent = agent or default_agent()
    _trace("=" * 70)
    _trace(f"USER: {question}")
    state = agent.invoke(
        {"messages": [SYSTEM_PROMPT, HumanMessage(content=question)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    final = state["messages"][-1].content
    _trace(f"FINAL: {final}")
    return final


async def stream_agent(question: str, agent=None):
    """Async generator of typed agent events for SSE (Part 4).

    Yields dicts: {"type":"tool_call",...}, {"type":"tool_result",...}, and
    {"type":"token","text":...} as the final answer is generated. Uses LangGraph's
    dual stream: "updates" gives node-level tool events, "messages" gives the LLM
    tokens as they arrive (so the client sees the answer build up incrementally).
    """
    agent = agent or default_agent()
    inputs = {"messages": [SYSTEM_PROMPT, HumanMessage(content=question)]}
    config = {"recursion_limit": RECURSION_LIMIT}
    _trace("=" * 70)
    _trace(f"USER (stream): {question}")

    async for mode, chunk in agent.astream(
        inputs, config=config, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node_output in chunk.values():
                for m in (node_output or {}).get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            yield {"type": "tool_call", "name": tc["name"], "args": tc["args"]}
                    elif isinstance(m, ToolMessage):
                        yield {"type": "tool_result", "name": m.name,
                               "preview": str(m.content)[:200]}
        else:  # "messages": (token_chunk, metadata)
            msg_chunk, _meta = chunk
            text = getattr(msg_chunk, "content", "") or ""
            if text and isinstance(msg_chunk, AIMessageChunk):
                yield {"type": "token", "text": text}


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python -m src.agent "your question"', file=sys.stderr)
        raise SystemExit(2)

    question = " ".join(argv)
    final = answer(question)
    print(f"\nQ: {question}\n")
    print(final)
    print(f"\n(trace logged to {settings.agent_trace_log})")


if __name__ == "__main__":
    main()
