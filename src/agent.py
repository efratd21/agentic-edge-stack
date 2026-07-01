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
import re
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
        "You are a helpful assistant with a tool, `rag_search`, that searches a "
        "local knowledge base about AI agents (agent loops, tool calling, ReAct, "
        "RAG, agent memory, multi-agent systems).\n"
        "- For questions about AI agents, call `rag_search` and ground your answer "
        "in the returned passages.\n"
        "- For any other question, answer directly and concisely from your own "
        "knowledge, in plain natural language.\n"
        "- If `rag_search` returns NO_RELEVANT_CONTEXT, answer directly from your "
        "own knowledge instead."
    )
)

# Bound on agent/tool turns so a confused model cannot loop forever.
RECURSION_LIMIT = 8


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def route_after_agent(state: AgentState) -> str:
    """The "use a tool vs. answer directly" decision: go to `tools` if the model
    emitted tool calls, otherwise end. Module-level so it is unit-testable."""
    return "tools" if state["messages"][-1].tool_calls else END


# Guard for a known small-model failure mode: emitting a JSON tool-call (often
# for an invented tool) as the answer TEXT instead of natural language. A real
# answer effectively never starts with `{"name": ...`.
_TOOL_JSON_RE = re.compile(r'^\s*\{\s*"(?:name|tool|function|action)"\s*:', re.IGNORECASE)
FALLBACK_ANSWER = "Sorry, I don't have information on that."


def looks_like_tool_json(text: str) -> bool:
    return bool(_TOOL_JSON_RE.match(text or ""))


def _plain_messages(question: str) -> list[BaseMessage]:
    return [
        SystemMessage(
            content="Answer the question directly in plain, natural language from "
            "your own knowledge. Do not use tools or output JSON. If you truly don't "
            "know, say so briefly."
        ),
        HumanMessage(content=question),
    ]


def plain_answer(question: str) -> str:
    """Recovery path (sync, for the CLI): regenerate a direct answer with NO tools
    bound, so the model answers in plain language instead of a tool-call JSON."""
    from .llm import get_chat_model

    text = (get_chat_model().invoke(_plain_messages(question)).content or "").strip()
    return FALLBACK_ANSWER if (not text or looks_like_tool_json(text)) else text


async def plain_answer_stream(question: str):
    """Recovery path (streaming, for /chat): same as plain_answer but yields tokens
    as they arrive, so a recovered answer still streams to the client."""
    from .llm import get_chat_model

    got = ""
    async for chunk in get_chat_model().astream(_plain_messages(question)):
        text = getattr(chunk, "content", "") or ""
        if text and isinstance(chunk, AIMessageChunk):
            got += text
            yield text
    if not got.strip():
        yield FALLBACK_ANSWER


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
            tool = tool_map.get(tc["name"])
            if tool is None:  # model hallucinated a tool that doesn't exist
                result = f"ERROR: no tool named '{tc['name']}'. Answer the user directly."
            else:
                result = tool.invoke(tc["args"])
            preview = str(result).replace("\n", " ")[:160]
            _trace(f"  TOOL '{tc['name']}' input={tc['args']} -> {preview!r}")
            outputs.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
            )
        return {"messages": outputs}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
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
    if looks_like_tool_json(final):
        final = plain_answer(question)
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

    # Stream tokens as they arrive — but if the answer starts with "{" hold it
    # back until we can tell whether it is a raw tool-call JSON (the small-model
    # failure mode) and, if so, replace it with a clean fallback. Natural answers
    # don't start with "{", so they stream unaffected.
    buffer, streaming = "", None  # streaming: None=undecided, True=flush freely

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
            if not (text and isinstance(msg_chunk, AIMessageChunk)):
                continue
            if streaming:
                yield {"type": "token", "text": text}
                continue
            buffer += text
            stripped = buffer.lstrip()
            if stripped and stripped[0] != "{":
                streaming = True  # a normal answer — flush and stream the rest
                yield {"type": "token", "text": buffer}
                buffer = ""

    # Anything still buffered starts with "{": if it's a raw tool-call, recover by
    # streaming a fresh plain answer; otherwise emit the buffered text as-is.
    if buffer:
        if looks_like_tool_json(buffer):
            async for tok in plain_answer_stream(question):
                yield {"type": "token", "text": tok}
        else:
            yield {"type": "token", "text": buffer}


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
