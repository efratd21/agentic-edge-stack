"""Central configuration, loaded from environment / .env via pydantic-settings.

Keeping every host, model name and tunable in one typed object means the rest of
the code never hardcodes a URL or a magic number, and a grader can change the
model or the corpus path from `.env` alone.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Ollama / models -----------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    # Chat / agent model. Must support tool-calling for the Part 3 agent
    # (gemma3:1b does NOT support tools in Ollama; llama3.2:3b does).
    model_name: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"  # embeddings via Ollama (part 2)
    embed_dim: int = 768                   # nomic-embed-text dim; MUST match index

    # --- RAG retrieval -------------------------------------------------------
    top_k: int = 3                         # assignment asks for top-3
    chunk_size: int = 1800                 # ~chars  (~450 tokens)
    chunk_overlap: int = 200               # ~chars  (~50 tokens)
    # Below this top-1 cosine score, rag_search reports NO_RELEVANT_CONTEXT so the
    # agent falls back to a direct answer (in-corpus ~0.81 vs off-corpus ~0.48).
    relevance_threshold: float = 0.6

    # --- Paths ---------------------------------------------------------------
    corpus_dir: str = "data/corpus"
    rag_log: str = "logs/rag_retrieval.log"
    agent_trace_log: str = "logs/agent_trace.log"          # part 3 deliverable
    structured_output_log: str = "logs/structured_output.log"  # bonus 1 deliverable


settings = Settings()
