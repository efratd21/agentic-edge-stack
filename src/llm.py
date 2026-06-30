"""Thin wrapper around the local chat model (Ollama via LangChain).

Centralizes how the chat model is constructed so the agent (Part 3) and the API
(Part 4) build it the same way, from the shared config. The model must support
tool-calling for the agent to choose tools (gemma3:1b does not; llama3.2:3b does).
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from .config import settings


def get_chat_model(*, model: str | None = None, temperature: float = 0.0, **kwargs) -> ChatOllama:
    """Build a ChatOllama pointed at the configured host/model.

    `model` overrides settings.model_name (used by the streaming demo to run a
    model that is already pulled). temperature=0 by default for a deterministic,
    reproducible agent trace.
    """
    return ChatOllama(
        model=model or settings.model_name,
        base_url=settings.ollama_host,
        temperature=temperature,
        **kwargs,
    )
