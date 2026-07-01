"""Bonus 1 — structured output with strict JSON via Pydantic v2.

The agent can extract structured metadata from a user message — a small set of
topic tags plus the overall sentiment — as a *validated* object rather than
free text. We lean on Ollama's native JSON-schema constrained decoding
(surfaced by LangChain as `ChatOllama.with_structured_output`): the model is
forced to emit JSON matching `QueryAnalysis`'s schema, and Pydantic then
validates it. A bad `sentiment` value or a missing field raises instead of
silently passing through.

This lives *beside* the streaming answer, not inside it: the API emits the
analysis as its own SSE event (`{"type":"analysis", ...}`) before the answer
tokens, so the natural-language stream stays pure text and never has to carry —
or be broken by — a JSON blob. See `stream_agent` (agent.py) and `/extract`
(api.py).

CLI (captures a real example to logs/structured_output.log — the deliverable):
    python -m src.schemas "How do I get an agent to stop looping forever?"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .config import settings

# Guides the extraction. `with_structured_output` already tells the model to
# fill the schema; this just sharpens what "topics" and "sentiment" should mean.
_EXTRACT_SYSTEM = SystemMessage(
    content=(
        "You extract structured metadata from the user's message. Identify 1-5 "
        "concise topic tags (short lowercase noun phrases describing what the "
        "message is about) and the overall sentiment of the message. Base the "
        "sentiment on the user's tone, not on how you feel about the topic."
    )
)


class QueryAnalysis(BaseModel):
    """Strict, validated analysis of a single user message (Bonus 1)."""

    topics: list[str] = Field(
        description="1-5 concise topic tags (short noun phrases) the message is about.",
        min_length=1,
        max_length=5,
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment of the user's message."
    )


def _messages(question: str) -> list:
    return [_EXTRACT_SYSTEM, HumanMessage(content=question)]


def analyze_query(question: str, model=None) -> QueryAnalysis:
    """Extract a validated `QueryAnalysis` from `question` (sync, for the CLI/tests).

    `model` is injected so tests can pass a stub; in production it is the real
    ChatOllama. `with_structured_output` binds the JSON schema so the model's
    reply is constrained to it, and returns a parsed `QueryAnalysis`.
    """
    if model is None:
        from .llm import get_chat_model

        model = get_chat_model()
    return model.with_structured_output(QueryAnalysis).invoke(_messages(question))


async def aanalyze_query(question: str, model=None) -> QueryAnalysis:
    """Async twin of `analyze_query`, for the /chat stream and /extract route.

    Uses `ainvoke` so the extraction LLM call does not block the event loop
    while the API is serving the SSE stream.
    """
    if model is None:
        from .llm import get_chat_model

        model = get_chat_model()
    return await model.with_structured_output(QueryAnalysis).ainvoke(_messages(question))


# --------------------------------------------------------------------------- #
# Example-output logging (the Bonus 1 deliverable)
# --------------------------------------------------------------------------- #
def _get_logger() -> logging.Logger:
    Path(settings.structured_output_log).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("structured_output")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(settings.structured_output_log, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python -m src.schemas "your message here"', file=sys.stderr)
        raise SystemExit(2)

    question = " ".join(argv)
    analysis = analyze_query(question)

    logger = _get_logger()
    logger.info("MESSAGE: %s", question)
    logger.info("  SCHEMA-VALID JSON: %s", analysis.model_dump_json())

    print(f'\nMESSAGE: "{question}"\n')
    print(analysis.model_dump_json(indent=2))
    print(f"\n(logged to {settings.structured_output_log})")


if __name__ == "__main__":
    main()
