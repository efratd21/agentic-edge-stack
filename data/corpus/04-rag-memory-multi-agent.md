# Retrieval, Memory, and Multi-Agent Systems

## Retrieval-Augmented Generation (RAG)

**Retrieval-Augmented Generation** gives an agent access to knowledge that is not
in the model's weights. Instead of relying only on what the model memorized
during training, the system retrieves relevant passages from an external store at
query time and injects them into the prompt.

The pipeline has two phases:

1. **Indexing (offline):** documents are split into **chunks**, each chunk is
   converted to a vector by an **embedding model**, and the vectors are stored in
   a vector index.
2. **Retrieval (online):** the user query is embedded with the same model, the
   index is searched for the nearest chunk vectors, and the top-k chunks are
   placed into the prompt as context.

Similarity is usually measured with **cosine similarity**. A common and efficient
implementation is to L2-normalize all vectors and use an inner-product index, so
that the inner product of two unit vectors equals their cosine. Retrieving a
small `k` (often 3) keeps the prompt focused; retrieving too much dilutes the
signal and wastes the context window.

RAG is the standard way to ground an agent in private or up-to-date data, and it
reduces hallucination by giving the model a real source to quote from.

## Chunking

How documents are split matters as much as the embedding model. Chunks that are
too large blur multiple topics into one vector and retrieve imprecisely; chunks
that are too small lose the surrounding context needed to be useful. A typical
target is a few hundred tokens per chunk with a small overlap so that a sentence
split across a boundary still appears whole in one chunk. Splitting on document
structure — headings and paragraphs — before falling back to a fixed window keeps
each chunk topically coherent.

## Memory

Agents use two kinds of memory:

- **Short-term memory** is the conversation context: the running list of
  messages, tool calls, and observations the model sees on each step. It is
  bounded by the context window.
- **Long-term memory** is an external store the agent can write to and read from
  across sessions. RAG is one form of long-term memory; others include
  key-value stores of facts and summaries of past conversations. Long-term memory
  lets an agent remember user preferences or earlier results without keeping every
  token in the prompt.

## Multi-agent systems

A **multi-agent system** decomposes a task across several specialized agents that
communicate. A common shape is an **orchestrator–worker** pattern: a coordinator
agent breaks the goal into subtasks and delegates each to a worker agent (for
example, a researcher, a coder, and a reviewer). Multi-agent designs help when
subtasks need different tools or different system prompts, and when parallelism
speeds things up. The trade-off is added complexity in communication and in
keeping the agents' shared state consistent, so a single agent is preferable
until the task clearly outgrows it.

## Evaluating agents

Because agents are non-deterministic and multi-step, evaluation looks beyond a
single output. Useful signals include **task success rate** (did it reach the
goal), **trace inspection** (were the right tools called with the right
arguments), **number of steps** and **cost**, and **groundedness** (did the
answer actually come from retrieved context). Capturing real traces and retrieval
logs is essential for debugging and for trusting an agent in production.
