"""Part 1 — verification ("Hello World") against the local Ollama endpoint.

Confirms the server is up and that BOTH models the stack relies on respond:

  1. chat model  (gemma3:1b)        -> a non-empty completion (the Part 1 ask)
  2. embedder    (nomic-embed-text) -> a vector of the expected dim (used by Part 2)

Exits 0 on success, non-zero on the first failure, so it doubles as a health
check before launching the API (Part 4) or in CI. Host/model/dim are read from
the shared src/config.py (single source of truth).

Run from the repo root:
    python scripts/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when run as `python scripts/verify.py` (sys.path[0] would
# otherwise be scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ollama import Client  # noqa: E402

from src.config import settings  # noqa: E402

PROMPT = "Say hello in one short sentence."

GREEN, RED, RESET = "\033[1;32m", "\033[1;31m", "\033[0m"


def _ok(msg: str) -> None:
    print(f"{GREEN}[ok]{RESET}   {msg}")


def _fail(msg: str) -> None:
    print(f"{RED}[fail]{RESET} {msg}")


def _installed(client: Client) -> set[str]:
    """Model names known to the server (e.g. {'gemma3:1b', 'nomic-embed-text:latest'})."""
    return {m.model for m in client.list().models}


def _is_present(name: str, installed: set[str]) -> bool:
    # A model pulled without an explicit tag is listed as '<name>:latest'.
    return name in installed or f"{name}:latest" in installed


def main() -> int:
    client = Client(host=settings.ollama_host)
    try:
        # 1. Server reachable + required models installed.
        installed = _installed(client)
        _ok(f"server reachable at {settings.ollama_host}; {len(installed)} model(s) available")
        for needed in (settings.model_name, settings.embed_model):
            if not _is_present(needed, installed):
                _fail(f"required model '{needed}' is not installed; run ./scripts/deploy.sh")
                return 1

        # 2. Chat model returns a non-empty completion (the 'Hello World').
        resp = client.chat(
            model=settings.model_name,
            messages=[{"role": "user", "content": PROMPT}],
        )
        text = resp["message"]["content"].strip()
        if not text:
            raise RuntimeError("chat model returned an empty response")
        _ok(f"chat model '{settings.model_name}' responded: {text!r}")

        # 3. Embedder returns a vector of the expected dimension.
        emb = client.embeddings(
            model=settings.embed_model,
            prompt="search_document: hello world",
        )["embedding"]
        if len(emb) != settings.embed_dim:
            raise RuntimeError(
                f"embedding dim {len(emb)} != expected {settings.embed_dim}; "
                f"check EMBED_DIM matches '{settings.embed_model}'"
            )
        _ok(f"embedder '{settings.embed_model}' responded: {len(emb)}-dim vector")

    except Exception as exc:  # connection refused, model missing, dim mismatch, etc.
        _fail(str(exc))
        print(
            "\nIs Ollama running and are the models pulled? Try:  ./scripts/deploy.sh",
            file=sys.stderr,
        )
        return 1

    print(f"\n{GREEN}All checks passed — Part 1 endpoint is healthy.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
