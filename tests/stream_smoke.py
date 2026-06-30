"""Smoke app for the Part 4 SSE transport — and a live preview of the chat UI,
runnable NOW with gemma3:1b.

The full agent (src.api) needs a tool-capable model, but the streaming transport
and the web UI are model-agnostic. This app serves the SAME chat page
(`src/web/index.html`) and a `/chat` endpoint that streams plain gemma3:1b tokens
through the SAME SSE framing (`src.api.format_sse`) — so you can open the UI in a
browser and watch tokens stream in before llama3.2:3b is available.

    uvicorn tests.stream_smoke:app --port 8001
    # then open http://localhost:8001  (or:  curl -N "localhost:8001/demo")
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402
from langchain_core.messages import AIMessageChunk, SystemMessage, HumanMessage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.api import format_sse  # noqa: E402
from src.llm import get_chat_model  # noqa: E402

app = FastAPI()

# The full agent model (llama3.2:3b) may not be pulled yet; the live UI demo runs
# a model that already is, so you can watch real streaming now.
DEMO_MODEL = "gemma3:1b"

_INDEX_HTML = Path(__file__).resolve().parents[1] / "src" / "web" / "index.html"
_SYSTEM = SystemMessage(content="You are a concise, helpful assistant. Answer in a few sentences.")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": f"{DEMO_MODEL} (demo: plain streaming, no tools)"}


async def _stream(message: str):
    llm = get_chat_model(model=DEMO_MODEL, temperature=0.2)
    async for chunk in llm.astream([_SYSTEM, HumanMessage(content=message)]):
        text = getattr(chunk, "content", "") or ""
        if text and isinstance(chunk, AIMessageChunk):
            yield format_sse({"type": "token", "text": text})
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    async def gen():
        async for frame in _stream(body.message):
            if await request.is_disconnected():
                break
            yield frame

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/demo")
async def demo(q: str = "List three colors, one per line."):
    return StreamingResponse(
        _stream(q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
