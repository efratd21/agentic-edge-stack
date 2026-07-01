"""Part 4 — FastAPI app exposing the agent over a streaming /chat endpoint.

The agent's answer is streamed token-by-token via Server-Sent Events, so the
client sees it build up rather than waiting for the whole block. Tool usage is
streamed too, as distinct events, so the live interaction is legible.

Run:
    uvicorn src.api:app --reload
Verify (tokens should trickle in, not arrive all at once):
    curl -N -X POST localhost:8000/chat \
      -H 'content-type: application/json' \
      -d '{"message":"What is the ReAct pattern in AI agents?"}'
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import stream_agent
from .config import settings
from .schemas import QueryAnalysis, aanalyze_query

app = FastAPI(title="Agentic Edge Stack", version="0.1.0")

_INDEX_HTML = Path(__file__).parent / "web" / "index.html"


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def index() -> HTMLResponse:
    """The chat web UI (talks to /chat over SSE)."""
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


def format_sse(event: dict) -> str:
    """Frame one event as a Server-Sent Event. JSON-wrapping each event keeps the
    stream valid even when a token contains a newline."""
    return f"data: {json.dumps(event)}\n\n"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": settings.model_name}


@app.post("/extract", response_model=QueryAnalysis)
async def extract(body: ChatRequest) -> QueryAnalysis:
    """Bonus 1 — return a strict, schema-validated {topics, sentiment} object.

    This is the same structured extraction that /chat emits as its `analysis`
    SSE event, exposed on its own route so the JSON contract is easy to test.
    FastAPI validates the response against `QueryAnalysis` before sending it.
    """
    return await aanalyze_query(body.message)


@app.post("/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    async def event_stream():
        async for event in stream_agent(body.message):
            if await request.is_disconnected():
                break  # client navigated away — stop wasting compute
            yield format_sse(event)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (e.g. nginx)
        },
    )
