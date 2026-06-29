# Tool Calling (Function Calling)

**Tool calling**, also called function calling, is the mechanism by which an LLM
requests that the harness run a named function with structured arguments. It is
what turns a text generator into an agent that can act.

## How it works

The developer describes each tool with a name, a natural-language description,
and a JSON Schema for its parameters. These descriptions are passed to the model
alongside the user's message. When the model decides a tool is needed, instead of
emitting prose it emits a structured tool call:

```json
{
  "name": "rag_search",
  "arguments": { "query": "how does the agent decide to use a tool" }
}
```

The harness parses this, validates the arguments against the schema, executes the
real function, and returns the result to the model as a new message. The model
then continues — either calling another tool or producing a final answer.

Crucially, the model does **not** execute anything itself. It only *requests*
calls; the harness is the trusted component that actually runs code. This
separation is the main safety boundary in an agent.

## Anatomy of a good tool description

The description is the single most important factor in whether the model uses a
tool correctly. A good description states **what the tool does, what it returns,
and when to use it**. For a retrieval tool, naming the specific corpus domain
("search the local knowledge base about AI agents") helps the model distinguish
in-domain questions from general ones.

Parameters should be minimal and typed. Fewer, well-named arguments lead to more
reliable calls than many optional ones.

## Tool results are strings

Whatever a tool returns is serialized to text before it goes back to the model —
the LLM consumes strings, not Python objects. Returning a clear sentinel such as
`NO_RELEVANT_CONTEXT` when a search finds nothing lets the model fall back to a
direct answer instead of hallucinating from empty context.

## OpenAI-compatible tool calling

Most local servers, including Ollama, expose an **OpenAI-compatible** chat
endpoint that supports a `tools` array and returns `tool_calls` in the response.
This means the same agent code can target a local model or a hosted one by
changing only the base URL and model name. Native tool-calling support in the
serving engine is preferable to prompt-hacking the model into emitting JSON,
because the engine enforces the call format.

## Errors and retries

Tool calls fail: bad arguments, network errors, empty results. A resilient agent
catches the error, returns the error message to the model as the observation, and
lets the model decide how to recover — retry with different arguments, try another
tool, or explain to the user that it could not complete the task.
