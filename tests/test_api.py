"""Part 4 — SSE framing and the /chat streaming endpoint.

`stream_agent` is replaced with a fixed async generator so we test the HTTP/SSE
transport (framing, event order, [DONE] terminator) without loading the model.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

import src.api as api
from src.api import app, format_sse


def test_format_sse_is_valid_json_frame():
    event = {"type": "token", "text": "hi\nthere"}
    frame = format_sse(event)
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    assert json.loads(frame[len("data: ") :].strip()) == event


def test_chat_streams_events_then_done(monkeypatch):
    async def fake_stream_agent(message: str):
        yield {"type": "tool_call", "name": "rag_search", "args": {"query": message}}
        yield {"type": "token", "text": "Paris"}

    monkeypatch.setattr(api, "stream_agent", fake_stream_agent)

    with TestClient(app) as client:
        resp = client.post("/chat", json={"message": "q"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text

    assert '"type": "tool_call"' in body
    assert '"type": "token", "text": "Paris"' in body
    assert body.strip().endswith("data: [DONE]")
