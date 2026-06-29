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
    model_name: str = "gemma3:1b"          # chat / generation model (parts 3-4)
    embed_model: str = "nomic-embed-text"  # embeddings via Ollama (part 2)
    embed_dim: int = 768                   # nomic-embed-text dim; MUST match index

    # --- RAG retrieval -------------------------------------------------------
    top_k: int = 3                         # assignment asks for top-3
    chunk_size: int = 1800                 # ~chars  (~450 tokens)
    chunk_overlap: int = 200               # ~chars  (~50 tokens)

    # --- Paths ---------------------------------------------------------------
    corpus_dir: str = "data/corpus"
    rag_log: str = "logs/rag_retrieval.log"


settings = Settings()
