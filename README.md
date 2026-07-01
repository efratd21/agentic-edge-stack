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
              │  llama3.2:3b      │   embeddings: nomic-embed-text
              └──────────────────┘
```

The retrieval engine at the bottom (Part 2) is exposed to the agent as the
`rag_search` tool.

---

## Tech stack & key decisions

| Concern | Choice | Why |
|---|---|---|
| Inference engine | **Ollama** | One-command model pull, OpenAI-compatible API, native tool-calling, serves both the LLM and the embedder from one stack. |
| LLM | **`llama3.2:3b`** | Lightweight, instruct-tuned, **tool-calling capable** (required by the Part 3 agent). See [why](#why-llama32-3b-for-the-agent). |
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

The "brain" of the stack is a lightweight instruct LLM (`llama3.2:3b`) served
locally by **Ollama**, which serves the embedding model from the same engine. Two
scripts cover bringing the server up and proving it answers.

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
[deploy] Chat model  : llama3.2:3b
[deploy] Embed model : nomic-embed-text
[deploy] Ollama already installed: ollama version is 0.30.11
[deploy] Ollama server already responding at http://localhost:11434
[deploy] Pulling llama3.2:3b ...
[deploy] Pulling nomic-embed-text ...
[deploy] Done. Installed models:
NAME                       ID              SIZE      MODIFIED
llama3.2:3b                a80c4f17acd5    2.0 GB    ...
nomic-embed-text:latest    0a109f422b47    274 MB    ...

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
[ok]   server reachable at http://localhost:11434; 3 model(s) available
[ok]   chat model 'llama3.2:3b' responded: 'Hello!'
[ok]   embedder 'nomic-embed-text' responded: 768-dim vector

All checks passed — Part 1 endpoint is healthy.
```

<sub>Only the chat "Hello World" is strictly required by Part 1; the embedder
check is added because `deploy.sh` pulls both models and Part 2 depends on the
embedder. The greeting wording varies between runs — generation is
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

## Part 3 — The Agentic Orchestrator

The RAG flow from Part 2 is wrapped as a single tool, `rag_search`, that an
autonomous agent can **choose** to call. The agent is a small, explicit
**LangGraph** graph, kept minimal so the control flow is readable at a glance:

```
        START → agent ──(tool_calls?)──► tools ──┐
                  ▲                               │
                  └───────────────────────────────┘
                  │
                  └──(no tool_calls)──► END
```

The conditional edge after `agent` **is** the "use a tool vs. answer directly"
decision the assignment asks for: if the model emitted tool calls, execute them
and loop back; otherwise end with the answer. A recursion limit bounds the loop.

Code: [`src/agent.py`](src/agent.py) (graph + trace), [`src/tools.py`](src/tools.py)
(`rag_search`), [`src/llm.py`](src/llm.py) (model wrapper).

### Why LangGraph (and not a hand-rolled loop or LangChain)

The assignment lists *"LangChain or LangGraph (or build a native loop)"*. We chose
**LangGraph**: a conditional-edge graph is the most readable expression of this
exact tool-vs-direct decision, and LangGraph is the current standard for agentic
orchestration (LangChain's `AgentExecutor` is legacy). The graph is deliberately
minimal — for a single tool it stays a few nodes — so it does not hide control
flow behind framework magic.

### Why `llama3.2:3b` for the agent

The agent needs **native tool-calling**. `gemma3:1b` does not support tools in
Ollama (the server returns `400 ... does not support tools`), so the chat model
is **`llama3.2:3b`** — listed in the assignment's Part 1 model options and
tool-capable. Only `MODEL_NAME` changes; the graph, tool and embedder are
unchanged.

### The tool and the fallback

`rag_search`'s description names the corpus domain (AI agents) to steer the model
toward using it for in-domain questions. But a small model is an imperfect router
— in practice `llama3.2:3b` sometimes reaches for the tool even on a general
question. That is exactly why there is a **second line of defence**: a relevance
threshold (top-1 cosine `< 0.6`, chosen from the 0.81-vs-0.48 gap measured in
Part 2) turns a weak hit into the sentinel `NO_RELEVANT_CONTEXT`, so the agent
falls back to a direct answer instead of grounding on an irrelevant chunk. The
trace below shows both layers at work.

### Running it

```bash
python -m src.agent "What is the ReAct pattern in AI agents?"   # → calls rag_search
python -m src.agent "What is the capital of France?"            # → falls back to a direct answer
```

Every decision and tool I/O is written to
[`logs/agent_trace.log`](logs/agent_trace.log) — the interaction-trace
deliverable. The committed trace (abridged) shows both branches:

```
USER: What is the ReAct pattern in AI agents?
  DECISION: call tool 'rag_search' args={'query': 'ReAct pattern in AI agents'}
  TOOL 'rag_search' ... -> '[score=0.76] The most influential agent pattern is **ReAct** ...'
  DECISION: answer directly -> 'The ReAct pattern in AI agents is a control flow that ...'
FINAL: The ReAct pattern ... interleaves reasoning and action-taking. ...

USER: What is the capital of France?
  DECISION: call tool 'rag_search' args={'query': 'capital of France'}
  TOOL 'rag_search' input={'query': 'capital of France'} -> 'NO_RELEVANT_CONTEXT'
  DECISION: answer directly -> 'The capital of France is Paris.'
FINAL: The capital of France is Paris.
```

For the general question the model *did* reach for the tool, but retrieval scored
below threshold (`NO_RELEVANT_CONTEXT`) and the agent fell back to a direct,
correct answer — both the tool decision and the relevance fallback visible in one
trace.

A third safety net guards the **output**: a small model sometimes emits a JSON
tool-call (often for a tool it invented) as its answer *text* instead of using
the tool-calling format. When the answer starts like a raw tool-call, the agent
regenerates it once with no tools bound (`plain_answer`) so the model replies in
plain language — so no JSON blob ever leaks to the user, and a general question
still gets a real answer. Natural answers (which don't start with `{`) stream
through untouched.

---

## Part 4 — API Serving & Streaming

[`src/api.py`](src/api.py) wraps the agent in a FastAPI app with a streaming
`POST /chat` endpoint. The agent's answer is streamed **token-by-token** over
Server-Sent Events, so the client sees it build up rather than waiting for the
whole block; tool usage is streamed too, as distinct events.

```bash
uvicorn src.api:app --port 8000
```
```bash
curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"What is the ReAct pattern in AI agents?"}'
```

Each SSE frame is a JSON event (the contract emitted by `stream_agent`):

| Event | Meaning |
|---|---|
| `{"type":"tool_call","name":…,"args":{…}}` | the agent decided to search |
| `{"type":"tool_result","name":…,"preview":…}` | the tool's output |
| `{"type":"token","text":…}` | one chunk of the final answer |
| `[DONE]` | end of stream |

A real `/chat` run is captured (abridged) in
[`logs/chat_stream.log`](logs/chat_stream.log):

```
data: {"type": "tool_call", "name": "rag_search", "args": {"query": "tool calling in AI agents"}}
data: {"type": "tool_result", "name": "rag_search", "preview": "[score=0.77] **Tool calling**, also called function calling, is the mechanism ..."}
data: {"type": "token", "text": "Tool"}
data: {"type": "token", "text": " calling"}
data: {"type": "token", "text": " in"}
    ... (token frames continue) ...
data: {"type": "token", "text": "."}
data: [DONE]
```

### Streaming is real, not buffer-then-flush

Under the hood, [`src/agent.py`](src/agent.py)'s `stream_agent` uses LangGraph's
dual stream (`stream_mode=["updates","messages"]`) to yield node-level tool events
and LLM token chunks **as they arrive**. Timestamping each SSE frame as it arrives
confirms it — tokens trickle in one at a time rather than landing all at once:

```
+ 1747.3 ms   data: {"type":"token","text":"Here"}     ← first token (model latency)
+ 1809.2 ms   data: {"type":"token","text":" are"}      Δ  62 ms
+ 1836.3 ms   data: {"type":"token","text":" two"}      Δ  27 ms
   ... ~40 frames, each ~55 ms apart ...
+ 4013.7 ms   data: [DONE]
```

The steady ~55 ms gaps between frames are the signature of real streaming; a
buffer-then-flush implementation would instead show one long pause and then every
frame at once.

### Web UI (bonus, beyond the requirement)

Part 4's requirement — a FastAPI `/chat` endpoint that streams over SSE — is met
by [`src/api.py`](src/api.py) alone; `curl -N` is enough to see it. As an extra,
a minimal chat page ([`src/web/index.html`](src/web/index.html)) is served at
`GET /`: open `http://localhost:8000` and the answer builds up token-by-token,
with a chip appearing when the agent searches the knowledge base. It is a single
self-contained file — no build step and no CDN, so it works offline.

---

## Tests

A `pytest` suite covers the central logic of each part. It tests **real logic**,
not a mocked model — the only place a model is replaced is the `/chat` transport
test, which stubs `stream_agent` to isolate the HTTP/SSE layer.

```bash
uv run pytest          # all 14 tests   (or: .venv/bin/python -m pytest)
uv run pytest -v       # list each test by name
```

| File | Part | Covers |
|---|---|---|
| [`tests/test_rag.py`](tests/test_rag.py) | 2 | chunking, embedding normalization, cosine metric, in- vs off-corpus retrieval |
| [`tests/test_tools.py`](tests/test_tools.py) | 3 | `rag_search` returns passages in-corpus, `NO_RELEVANT_CONTEXT` off-corpus |
| [`tests/test_agent.py`](tests/test_agent.py) | 3 | the tool-vs-direct routing decision (`route_after_agent`) |
| [`tests/test_api.py`](tests/test_api.py) | 4 | SSE framing and the `/chat` event stream (`tool_call` → `token` → `[DONE]`) |

The RAG and tool tests use the real `nomic-embed-text` embedder, so Ollama must
be running — they double as integration checks that cosine retrieval separates
in-corpus from off-corpus queries.

<details>
<summary>What each of the 14 tests checks</summary>

**`test_rag.py`** — Part 2
- `test_window_keeps_short_text_whole` — text under the window size stays one chunk
- `test_window_splits_large_text_within_size` — oversized text splits into ≤-size pieces
- `test_split_sections_on_markdown_headings` — sections split on `#`/`##` headings
- `test_corpus_chunks_do_not_exceed_window` — no corpus chunk exceeds the window
- `test_embeddings_normalized_and_correct_dim` — vectors are 768-dim and unit-length
- `test_cosine_self_similarity_is_one` — a text vs itself ≈ 1.0 (the metric is cosine)
- `test_in_corpus_scores_higher_than_off_corpus` — in-topic beats off-topic, above threshold
- `test_search_returns_top_k` — search returns exactly top-3

**`test_tools.py`** — Part 3
- `test_in_corpus_query_returns_passages` — in-domain query returns scored passages
- `test_off_corpus_query_returns_sentinel` — off-domain query returns `NO_RELEVANT_CONTEXT`

**`test_agent.py`** — Part 3
- `test_routes_to_tools_when_model_emits_tool_calls` — tool calls → the `tools` node
- `test_routes_to_end_on_a_plain_answer` — no tool calls → `END`

**`test_api.py`** — Part 4
- `test_format_sse_is_valid_json_frame` — each frame is `data: {json}\n\n`
- `test_chat_streams_events_then_done` — `/chat` streams events and ends with `[DONE]`
</details>

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
│   ├── rag.py                # Part 2: load → chunk → embed → FAISS → search + CLI
│   ├── tools.py              # Part 3: rag_search tool (+ relevance fallback)
│   ├── llm.py                # Part 3/4: ChatOllama wrapper from config
│   ├── agent.py              # Part 3: LangGraph tool-calling agent + trace
│   ├── api.py                # Part 4: FastAPI /chat (SSE) + web UI at /
│   └── web/index.html        # Part 4 (bonus): minimal streaming chat web UI
├── tests/                    # pytest suite — one file per part with core logic
│   ├── test_rag.py           # Part 2
│   ├── test_tools.py         # Part 3
│   ├── test_agent.py         # Part 3
│   └── test_api.py           # Part 4
└── logs/
    ├── rag_retrieval.log     # Part 2 deliverable: query → retrieved chunks
    ├── agent_trace.log       # Part 3 deliverable: agent interaction trace
    └── chat_stream.log       # Part 4 deliverable: captured /chat SSE stream
```
