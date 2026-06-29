# What Is an AI Agent?

An **AI agent** is a system that uses a large language model (LLM) as its
reasoning engine to decide *which actions to take* in order to accomplish a
goal, rather than producing a single fixed response. The defining property of an
agent is **autonomy over control flow**: the model itself chooses, step by step,
whether to call a tool, ask a follow-up question, or return a final answer.

A plain LLM call is a function from prompt to text. An agent wraps that call in a
loop: the model emits an action, the environment executes it and returns an
observation, and the observation is fed back into the model for the next
decision. This loop continues until the model decides the task is complete.

## Agent vs. workflow

It is useful to distinguish two patterns:

- A **workflow** is an orchestration where the control flow is fixed in code. The
  developer decides in advance "first retrieve, then summarize, then translate."
  The LLM fills in steps but does not choose the path.
- An **agent** lets the model decide the path at runtime. The number of steps,
  the tools used, and the order are determined by the model's own reasoning.

Agents are more flexible but harder to debug and to bound. A good rule of thumb
is to use the simplest thing that works: prefer a fixed workflow when the task
structure is known, and reach for an agent only when the task genuinely requires
dynamic decision-making.

## Core components

Every agent, regardless of framework, is built from the same parts:

1. **Model** — the LLM that does the reasoning and produces actions.
2. **Tools** — functions the model can call to affect or observe the world
   (search, code execution, API calls, retrieval).
3. **Memory** — the context the model carries between steps, both short-term
   (the running conversation) and long-term (external stores).
4. **Control loop** — the harness that runs the model, executes tool calls, feeds
   results back, and decides when to stop.

## Why "agentic edge"

Running an agent locally ("on the edge") means the model, the tools, and the
retrieval store all live on the user's own machine. This trades raw model
capability for privacy, offline operation, and predictable cost. Small instruct
models such as Gemma 3 1B are well suited to this setting: they are capable
enough to drive a tool-calling loop while remaining cheap to run.
