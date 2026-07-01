"""Shared test fixtures.

Keep test runs from appending to the committed deliverable logs
(`logs/rag_retrieval.log`, `logs/agent_trace.log`): point them at a temp dir and
reset the cached file handlers so a `pytest` run never mutates tracked files.
"""

from __future__ import annotations

import logging

import pytest

from src.config import settings


@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "rag_log", str(tmp_path / "rag_retrieval.log"))
    monkeypatch.setattr(settings, "agent_trace_log", str(tmp_path / "agent_trace.log"))
    for name in ("rag_retrieval", "agent_trace"):
        lg = logging.getLogger(name)
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
    yield
