"""Smoke test for the Part 4 SSE transport — runnable NOW with gemma3:1b.

The full agent needs a tool-capable model, but the streaming machinery is
model-agnostic. This minimal app streams plain gemma3:1b tokens through the SAME
SSE framing (`src.api.format_sse`) so `curl -N` proves tokens arrive
incrementally rather than as one block.

    uvicorn tests.stream_smoke:app --port 8001
    curl -N "localhost:8001/demo?q=List+three+colors,+one+per+line."
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langchain_core.messages import AIMessageChunk  # noqa: E402

from src.api import format_sse  # noqa: E402
from src.llm import get_chat_model  # noqa: E402

app = FastAPI()


@app.get("/demo")
async def demo(q: str = "List three colors, one per line."):
    llm = get_chat_model(temperature=0.2)

    async def gen():
        async for chunk in llm.astream(q):
            text = getattr(chunk, "content", "") or ""
            if text and isinstance(chunk, AIMessageChunk):
                yield format_sse({"type": "token", "text": text})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
