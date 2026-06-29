# The Agent Loop and Reasoning Patterns

The heart of an agent is its **control loop**. On each iteration the model is
given the goal, the history so far, and the set of available tools. It responds
either with a tool call or with a final answer. If it calls a tool, the harness
executes the tool, appends the result to the history, and loops again.

```
loop:
    action = model(goal, history, tools)
    if action is final_answer:
        return action.text
    observation = execute(action.tool, action.args)
    history.append(action, observation)
```

A robust loop bounds the number of iterations (a `max_steps` limit) so a
confused model cannot spin forever, and it handles tool errors by feeding the
error text back to the model instead of crashing.

## ReAct: reason + act

The most influential agent pattern is **ReAct** (Reasoning + Acting). The model
interleaves *thoughts* (free-text reasoning about what to do next) with *actions*
(tool calls). A ReAct trace looks like:

```
Thought: I need the user's account balance, which is not in the prompt.
Action: lookup_balance(user_id="42")
Observation: balance = $120.50
Thought: I now have the answer.
Action: finish("Your balance is $120.50")
```

Exposing the thought step improves reliability because the model commits to a
rationale before acting, and it makes the trace far easier to debug. The cost is
extra tokens.

## Plan-and-execute

An alternative to step-by-step ReAct is **plan-and-execute**: the model first
drafts a full plan (a list of subtasks), then executes the steps in order,
optionally re-planning if a step fails. This reduces per-step model calls and
keeps long tasks on track, at the cost of less moment-to-moment adaptivity.

## Reflection

**Reflection** adds a self-critique step: after producing a draft answer the
model is asked to evaluate it against the goal and revise. Reflection improves
quality on tasks with verifiable criteria (code that must compile, math that must
check out) but adds latency and can loop if no stopping condition is set.

## Deciding to use a tool vs. answering directly

A key behavior of a well-built agent is **knowing when *not* to act**. For a
general-knowledge question the model should answer directly from its own
parameters; for a question about private or domain-specific data it should call a
retrieval tool. Forcing a tool call on every turn wastes latency and can inject
irrelevant context; never calling tools defeats the point of the agent. The tool
*descriptions* are what the model uses to make this decision, so they must be
specific and accurate.
