# Ollamanauts

`Ollamanauts` is a small Python package for building Ollama-backed agents with a minimal public API.

It exposes:

- `Agent` for package-style usage
- `InteractiveAgent` for terminal-oriented sessions
- `SubAgent` for focused delegated work
- `ToolOrchestrator` and `ToolResult` for lower-level control

The package is intentionally conservative by default:

- no default user tools are registered
- subagents are optional
- extra tools must be provided explicitly as Python callables

## What The Package Does

At its core, an Ollamanauts agent:

1. sends a chat request to an Ollama model
2. exposes a set of registered Python functions as tools
3. executes any tool calls returned by the model
4. feeds those tool results back into the conversation
5. returns the assistant's final text response

The public package API is designed for this shape:

```python
from ollamanauts import Agent

agent = Agent(
    model="gemma4:31b",
    extra_instructions="You are a concise research assistant.",
)

reply = agent.run("What kinds of tasks can you help with?")
print(reply)
```

## Installation

Prerequisites:

- Python `3.13+`
- an available local Ollama installation
- the target model already pulled into Ollama

Install the package from the repository root:

```bash
pip install .
```

Or with `uv`:

```bash
uv pip install .
```

The package depends on:

- `ollama`

## Quick Start

Minimal package usage:

```python
from ollamanauts import Agent

agent = Agent(
    model="gemma4:31b",
    extra_instructions="Answer in short paragraphs.",
    enable_subagents=False,
)

print(agent.run("Introduce yourself and describe your capabilities."))
```

## Example Implementations

Concrete examples live in:

- examples/customer_agent.py
- examples/customer_cli.py
- examples/customer_tools.py

Together they show how to:

- register explicit tools
- keep the agent constrained to a narrow domain
- run a package-style agent without the terminal CLI
- run an interactive terminal loop with the same tools loaded

Example:

```python
from ollamanauts import Agent


CUSTOMERS = {
    "CUST-100": {"name": "Ada Lovelace", "status": "active", "plan": "pro"},
    "CUST-200": {"name": "Grace Hopper", "status": "trial", "plan": "starter"},
}


def lookup_customer(customer_id: str) -> dict[str, str]:
    """Return one customer record by ID."""
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        return {"id": customer_id, "found": "false"}
    return {"id": customer_id, "found": "true", **customer}


def explain_plan(plan: str) -> dict[str, str]:
    """Explain the features of a named plan."""
    descriptions = {
        "starter": "Starter includes basic support and a small usage quota.",
        "pro": "Pro includes priority support and a higher usage quota.",
    }
    return {"plan": plan, "description": descriptions.get(plan, "Unknown plan")}


agent = Agent(
    model="gemma4:31b",
    extra_instructions=(
        "You are a support assistant. Use the provided tools for customer"
        " and plan questions. If data is missing, say so plainly."
    ),
    tools=[lookup_customer, explain_plan],
    enable_subagents=False,
)


response = agent.run(
    "Customer CUST-100 wants to know whether their plan includes priority support."
)

print(response)
```

Interactive example:

```bash
python -m examples.customer_cli --model gemma4:31b --think medium
```

That example reuses the customer tools from `examples/customer_tools.py` and runs an `InteractiveAgent` with the same `/help`, `/tools`, `/clear`, `/compact`, and `/exit` commands as the built-in terminal interface.

## Public API

### `Agent`

`Agent` is the package-friendly entrypoint:

```python
from ollamanauts import Agent

agent = Agent(
    model="gemma4:31b",
    system_prompt=None,
    extra_instructions="Always answer in JSON.",
    think_mode="medium",
    tools=[],
    verbose=True,
    enable_subagents=True,
    max_context_tokens=32000,
    compact_threshold=0.85,
    compact_target=0.60,
)
```

Constructor arguments:

- `model`: Ollama model name, default `gemma4:31b`
- `system_prompt`: full replacement system prompt
- `extra_instructions`: appended to the default interactive prompt when `system_prompt` is not provided
- `think_mode`: forwarded to `ollama.chat`; defaults to `"medium"`
- `tools`: explicit user-supplied Python callables exposed as tools
- `verbose`: when `True`, prints runtime events (thinking/tool/subagent activity) to the terminal
- `enable_subagents`: when `True`, registers `deploy_subagent`
- `max_context_tokens`: optional approximate context budget used for token telemetry and compaction checks
- `subagent_max_context_tokens`: optional subagent budget; defaults to `max_context_tokens` when not provided
- `enable_auto_compaction`: enables automatic history compaction when threshold is exceeded
- `compact_threshold`: usage ratio that triggers compaction (for example, `0.85` = 85% of budget)
- `compact_target`: target usage ratio after compaction passes
- `compaction_preserve_last_n_turns`: number of recent turns kept verbatim during deterministic compaction
- `compaction_model`: optional model used for summary generation during compaction (defaults to the primary model)

Methods:

- `run(prompt, *, on_tool_result=None, on_thinking_chunk=None, on_thinking_end=None) -> str`
- `reset() -> None`
- `describe_tools() -> list[str]`

### Verbose runtime output

Use `verbose=True` to emit runtime traces while using the package `Agent` API:

```python
from ollamanauts import Agent

agent = Agent(
    model="gemma4:31b",
    verbose=True,
)

reply = agent.run("Investigate this issue and summarize next steps.")
```

With verbose mode enabled, the terminal output includes:

- tool call results (`[ok]` / `[error]`)
- model thinking stream markers (`[thinking] ... [end thinking]`)
- subagent lifecycle traces when subagents are used (`[subagent]`, subagent thinking/tool traces, and `[subagent result]`)


### Context compaction

When `max_context_tokens` is configured, the agent estimates current context usage before each model call.
If usage crosses `compact_threshold` and `enable_auto_compaction=True`, the agent compacts history by:

- preserving the system prompt
- preserving unresolved tool episodes and recent turns
- summarizing older context into a structured memory block (`Facts`, `Decisions`, `Constraints`, `Open Questions`, `Pending Actions`)

If context is still above target, additional compaction passes run and may summarize more aggressively while keeping tool interaction boundaries intact.

### `BaseAgent`, `InteractiveAgent`, and `SubAgent`

These lower-level classes share the same tool loop but have different intended roles:

- `BaseAgent`: generic agent implementation
- `InteractiveAgent`: uses the main interactive system prompt
- `SubAgent`: uses a separate delegation prompt and exposes a `run(task)` convenience method

If you want the package abstraction, use `Agent`. If you want more control over prompts and orchestration, instantiate the lower-level classes directly.

### `ToolOrchestrator` and `ToolResult`

`ToolOrchestrator` holds registered callables and executes tool calls returned by Ollama. `ToolResult` is the normalized execution result object:

```python
ToolResult(
    name="lookup_customer",
    arguments={"customer_id": "CUST-100"},
    content='{"id": "CUST-100", "status": "active"}',
    ok=True,
)
```

Non-string tool return values are JSON-serialized before being sent back to the model.

## Tool Model

Tools are plain Python callables registered by name:

```python
def lookup_customer(customer_id: str) -> dict[str, str]:
    return {"id": customer_id, "status": "active"}
```

Pass them into `Agent`:

```python
agent = Agent(
    model="gemma4:31b",
    tools=[lookup_customer],
    enable_subagents=False,
)
```

The callable's `__name__` becomes the exposed tool name. Keep names specific and docstrings useful so the model can choose tools accurately.

Current defaults:

- `DEFAULT_TOOLS` is empty
- `Agent(enable_subagents=False)` registers no tools
- `Agent(enable_subagents=True)` registers only `deploy_subagent`
- subagents do not receive `deploy_subagent`, so recursive delegation is disabled by default

## Subagents

Subagent support is created through `make_deploy_subagent_tool(...)`, which returns a `deploy_subagent(task: str) -> str` callable.

When enabled on the main `Agent`, the model can delegate one focused subtask. The delegated worker:

- uses the `SubAgent` prompt
- gets its own message history
- does not receive recursive subagent deployment by default

This keeps delegation narrow and predictable.

## CLI Usage

The project also ships a terminal interface:

```bash
ollamanauts --model gemma4:31b --think medium
```

Or from the repository root:

```bash
python main.py --model gemma4:31b --think medium
```

Customer-support interactive example:

```bash
python -m examples.customer_cli --model gemma4:31b --think medium
```

CLI commands:

- `/help` shows command help
- `/tools` lists registered tools
- `/clear` resets conversation state
- `/compact` triggers manual context compaction
- `/exit` quits

The CLI uses `InteractiveAgent` and prints tool activity plus thinking output through helpers in ollamanauts/terminal_output.py.

If you want interactive mode with your own tools, see examples/customer_cli.py. The built-in `ollamanauts` CLI does not currently load external tools from imports or config.

## Prompt Behavior

Prompt files are loaded from package resources in:

- ollamanauts/prompts/interactive_agent.md
- ollamanauts/prompts/subagent.md

Prompt composition works like this:

- if `system_prompt` is provided, it completely replaces the default prompt
- otherwise, `extra_instructions` is appended under `Additional user instructions:`

Example:

```python
agent = Agent(
    extra_instructions="Always answer in valid JSON.",
    enable_subagents=False,
)
```

## Project Layout

```text
ollamanauts/
  __init__.py
  agent.py
  cli.py
  terminal_output.py
  tool_orchestrator.py
  prompts/
    __init__.py
    interactive_agent.md
    subagent.md
  tools/
    __init__.py
    subagents.py
examples/
  customer_agent.py
  customer_cli.py
  customer_tools.py
tests/
  test_package_behavior.py
main.py
pyproject.toml
```

## Testing

Run the current test suite with:

```bash
python -m unittest tests/test_package_behavior.py
```

The tests currently cover:

- package exports
- prompt loading from package resources
- prompt composition behavior
- safe default tool registration
- explicit user tool registration
- subagent registration boundaries

## Current Caveats

A few details are worth knowing before building on this package:

- The package registers no default user tools.
- The CLI currently uses the same empty `DEFAULT_TOOLS` set plus optional subagent deployment.
- This project assumes an Ollama server is already reachable through the `ollama` Python client.

That means the safest production-style usage today is:

- provide explicit narrow-domain tools
- set `enable_subagents` intentionally
- use `system_prompt` when you want fully explicit prompt control
