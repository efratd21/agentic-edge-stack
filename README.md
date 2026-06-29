# Agentic Edge Stack

> A locally hosted, agentic AI assistant: a small instruct LLM served on-device,
> grounded in a domain corpus through an in-memory RAG tool, driven by an agentic
> tool-calling loop, and exposed over a streaming FastAPI endpoint.

This repository is the solution to the **Machine Learning Systems Engineer**
take-home assessment ("The Agentic Edge Stack"). It is built and documented
**part by part**; each part ships a runnable artifact **and** a captured log so
every claim in this README can be reproduced from a clean checkout.

---

## Architecture

```
                ┌─────────────────────────────────────────────┐
   HTTP (SSE)   │                FastAPI  /chat                │   (Part 4)
 user ───────►  │   StreamingResponse (text/event-stream)     │
                └───────────────┬─────────────────────────────┘
                                │
                       ┌────────▼─────────┐
                       │   Agent loop      │   decides: tool or direct answer
                       │  (tool-calling)   │   (Part 3)
                       └───┬───────────┬───┘
                           │           │
                ┌──────────▼───┐   ┌───▼────────────────┐
                │  rag_search  │   │   answer directly   │
                │   (tool)     │   │   (LLM only)        │
                └──────┬───────┘   └────────────────────┘
                       │
        ┌──────────────▼───────────────┐
        │ embed query → FAISS top-k=3  │   ◄── Part 2
        │ → inject chunks into prompt  │
        └──────────────┬───────────────┘
                       │
              ┌────────▼─────────┐
              │  Local LLM        │   Ollama (OpenAI-compatible) :11434
              │  gemma3:1b        │   embeddings: nomic-embed-text
              └──────────────────┘
```

The retrieval engine at the bottom (Part 2) is exposed to the agent as the
`rag_search` tool.

---

## Tech stack & key decisions

| Concern | Choice | Why |
|---|---|---|
| Inference engine | **Ollama** | One-command model pull, OpenAI-compatible API, native tool-calling, serves both the LLM and the embedder from one stack. |
| LLM | **`gemma3:1b`** | Lightweight, instruct-tuned, tool-calling capable; runs comfortably on the edge. |
| Embeddings | **`nomic-embed-text`** (via Ollama) | 768-dim, instruction-tuned. Served by the *same* Ollama engine as the LLM — no separate torch/HuggingFace stack at runtime. See [why](#why-nomic-embed-text-for-embeddings). |
| Vector store | **FAISS** `IndexFlatIP` + L2-normalized vectors | In-memory, deterministic, zero external deps. Inner product over unit vectors == **cosine similarity**. |
| Config | **pydantic-settings** + `.env` | No hardcoded hosts or magic numbers; every tunable is typed and overridable. |
| Host environment | **WSL2 (Ubuntu)** on Windows | Ollama, FAISS and k3s behave far more predictably on Linux. |

### Why `nomic-embed-text` for embeddings

The assignment suggests `all-MiniLM-L6-v2` / `bge-small-en-v1.5`, which run on a
separate sentence-transformers (PyTorch + HuggingFace) stack. We deliberately use
**`nomic-embed-text` served by Ollama** so that **a single inference engine
serves both the LLM and the embedder** — the essence of a self-contained "edge
stack". Concretely:

- **One runtime, not two.** No second PyTorch/HuggingFace dependency, no separate
  model download or cache, no device/CUDA juggling — just `ollama pull`.
- **Reproducibility.** A reviewer on a fresh machine gets a working system from
  the model pull alone; nothing is fetched at query time.
- **Consistency.** Embeddings and chat share one client, host and
  OpenAI-compatible surface (`:11434`).
- **No quality trade-off.** `nomic-embed-text` is a modern, instruction-tuned
  768-dim embedder, competitive with or stronger than MiniLM (384-dim) on
  retrieval.

(MiniLM was also blocked on the development network, which first surfaced the
question — but the decision stands on the architectural merits above.)

`nomic-embed-text` is instruction-tuned and **requires task prefixes** —
`search_document:` for stored passages, `search_query:` for queries — which the
pipeline applies automatically.

---

## Part 1 — Model Serving & Deployment

The "brain" of the stack is a small instruct LLM served locally by **Ollama**,
which serves the embedding model from the same engine. Two scripts cover bringing
the server up and proving it answers.

### Environment & prerequisites

All commands assume a **WSL2 (Ubuntu)** shell on Windows. This is a deliberate
choice: Ollama, FAISS and (for the K8s bonus) k3s all behave far more predictably
on Linux than on Windows-native. Where a command differs, a `> Windows-native:`
note is given.

- **[Ollama](https://ollama.com)** — the local inference server. Install with
  `curl -fsSL https://ollama.com/install.sh | sh`
  (`deploy.sh` checks for it and prints this hint if missing).
  > Windows-native: download the installer from <https://ollama.com/download>.
- **[`uv`](https://github.com/astral-sh/uv)** — Python environment / dependency
  manager. Create the virtualenv from the lockfile:

  ```bash
  uv sync            # creates .venv from pyproject.toml / uv.lock
  ```

- **Config (optional)** — defaults work out of the box; override via `.env`:

  ```bash
  cp .env.example .env
  ```

### Deploy

[`scripts/deploy.sh`](scripts/deploy.sh) starts Ollama (if it is not already
running) and pulls the chat and embedding models named in `.env` (or their
defaults). It is **idempotent** — safe to re-run.

```bash
./scripts/deploy.sh
```

```
[deploy] Ollama host : http://localhost:11434
[deploy] Chat model  : gemma3:1b
[deploy] Embed model : nomic-embed-text
[deploy] Ollama already installed: ollama version is 0.30.11
[deploy] Ollama server already responding at http://localhost:11434
[deploy] Model already present: gemma3:1b
[deploy] Model already present: nomic-embed-text
[deploy] Done. Installed models:
NAME                       ID              SIZE      MODIFIED
nomic-embed-text:latest    0a109f422b47    274 MB    ...
gemma3:1b                  8648f39daa8f    815 MB    ...

Next: python scripts/verify.py
```

### Verify ("Hello World")

[`scripts/verify.py`](scripts/verify.py) confirms the server is up and that
**both** models the stack relies on actually respond — the chat model (the
"Hello World") **and** the embedder (a 768-dim vector, used by Part 2). It exits
non-zero on the first failure, so it doubles as a health check. Host, model and
embedding dimension are read from the shared `src/config.py`:

```bash
python scripts/verify.py
```

```
[ok]   server reachable at http://localhost:11434; 2 model(s) available
[ok]   chat model 'gemma3:1b' responded: 'Hello there! 😊'
[ok]   embedder 'nomic-embed-text' responded: 768-dim vector

All checks passed — Part 1 endpoint is healthy.
```

<sub>Only the chat "Hello World" is strictly required by Part 1; the embedder
check is added because `deploy.sh` pulls both models and Part 2 depends on the
embedder. The chat wording varies between runs — generation is
non-deterministic.</sub>

---

## Part 2 — In-Memory RAG

A user query is embedded, matched against a FAISS index of the corpus by cosine
similarity, and the **top-3** chunks are returned with their provenance. In
Part 3 this becomes the `rag_search` tool the agent can choose to call.

### The dataset

A concise, focused corpus on **AI agents** (~1,800 words of Markdown, within the
assignment's 2–10 page range) under [`data/corpus/`](data/corpus):

| File | Topic |
|---|---|
| [`01-what-is-an-ai-agent.md`](data/corpus/01-what-is-an-ai-agent.md) | Definition, agent vs. workflow, core components |
| [`02-agent-loop-and-reasoning.md`](data/corpus/02-agent-loop-and-reasoning.md) | The control loop, ReAct, plan-and-execute, reflection |
| [`03-tool-calling.md`](data/corpus/03-tool-calling.md) | Function calling, tool descriptions, OpenAI-compatible tools |
| [`04-rag-memory-multi-agent.md`](data/corpus/04-rag-memory-multi-agent.md) | RAG, chunking, memory, multi-agent, evaluation |

The topic is deliberate: it gives clear **in-domain** questions for the agent to
answer via retrieval (Part 3), while a general-knowledge question (e.g. *"what is
the capital of France?"*) cleanly exercises the **direct-answer** fallback.

### Pipeline

The full implementation is in [`src/rag.py`](src/rag.py); configuration in
[`src/config.py`](src/config.py).

1. **Load & chunk** — every `.md` file is split **on Markdown headings first**,
   so each chunk stays topically coherent; an oversized section falls back to a
   fixed character window (`1800` chars, `200` overlap ≈ 450/50 tokens). For this
   corpus every section fits in one chunk, so **no chunk is split mid-sentence**
   (20 chunks, max 1198 chars). Each chunk keeps its `source` file and `section`
   heading for the log.
2. **Embed** — each text is sent to `nomic-embed-text` via Ollama with the
   required prefix (`search_document:` for chunks, `search_query:` for queries),
   then **L2-normalized**.
3. **Index** — vectors are added to `faiss.IndexFlatIP(768)`. Because the vectors
   are unit-length, inner product **is** cosine similarity. The index is built
   **once** at startup, not per query.
4. **Search** — the query is embedded and the index returns the **top-3** chunks
   with their cosine scores and provenance.

### Running it

```bash
python -m src.rag "how does an agent decide whether to use a tool or answer directly?"
```

Output (top-3 chunks with score + source + section), and a record appended to
`logs/rag_retrieval.log`:

```
QUERY: "how does an agent decide whether to use a tool or answer directly?"
Built index over 20 chunks; top-3:

  #1  score=0.810  02-agent-loop-and-reasoning.md § Deciding to use a tool vs. answering directly
      A key behavior of a well-built agent is **knowing when *not* to act**...
  #2  score=0.760  03-tool-calling.md § How it works
      The developer describes each tool with a name, a natural-language description...
  #3  score=0.742  02-agent-loop-and-reasoning.md § The Agent Loop and Reasoning Patterns
      The heart of an agent is its **control loop**...
```

### Deliverable: retrieval log

The full captured log is committed at
[`logs/rag_retrieval.log`](logs/rag_retrieval.log). It contains three
representative queries — one clearly in-corpus, one borderline, one out-of-corpus
— each with the chunks retrieved from memory:

| Query | Type | Top-1 result | Top score |
|---|---|---|---|
| *…decide whether to use a tool or answer directly?* | in-corpus | § Deciding to use a tool | **0.810** |
| *…how is RAG related to an agent's memory?* | borderline | § Retrieval-Augmented Generation | **0.815** |
| *what is the capital of France?* | out-of-corpus | (best effort, irrelevant) | **0.481** |

The ~0.33 gap between in-corpus (~0.81) and out-of-corpus (~0.48) is the signal
that lets the agent fall back to a direct answer in Part 3: below a relevance
threshold, `rag_search` will return `NO_RELEVANT_CONTEXT` instead of injecting an
irrelevant chunk and inviting a hallucination.

### Correctness checks

Four sanity checks guard the pipeline (run from the repo root):

| Check | Result | What it proves |
|---|---|---|
| Index dimension | `768` | Index dim matches the embedder; a mismatch would silently corrupt search. |
| Cosine metric | doc-vs-itself = **1.0000** | `IndexFlatIP` + L2-normalization really computes cosine. |
| Self-retrieval | a chunk retrieves **itself at rank 1** (score 0.978*) | Normalization and indexing are wired correctly. |
| Negative control | in-topic **0.863** ≫ off-topic **0.477** | Off-corpus queries score low → fallback is feasible. |

<sub>* Self-retrieval scores 0.978, not 1.0, **by design**: the query is embedded
with the `search_query:` prefix while the stored chunk used `search_document:`,
so the two vectors differ slightly. This is confirmation that the asymmetric
prefixes are being applied.</sub>

---

## Repository layout

```
agentic-edge-stack/
├── README.md                 # this file
├── pyproject.toml            # deps (managed by uv)
├── .env.example              # config template (OLLAMA_HOST, MODEL_NAME, EMBED_MODEL, …)
├── scripts/
│   ├── deploy.sh             # Part 1: start Ollama + pull models (idempotent)
│   └── verify.py             # Part 1: "Hello World" call to the endpoint
├── data/
│   └── corpus/               # Part 2 dataset (AI-agents corpus, ~1,800 words)
├── src/
│   ├── config.py             # pydantic-settings configuration
│   └── rag.py                # Part 2: load → chunk → embed → FAISS → search + CLI
└── logs/
    └── rag_retrieval.log     # Part 2 deliverable: query → retrieved chunks
```
