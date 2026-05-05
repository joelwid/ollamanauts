# Ollamanauts Package Migration Plan

Goal: make this repository importable as a Python package so users can create a configured Ollama agent with a small public API, safe defaults, optional subagents, and user-supplied tools.

The package should not ship script creation, script reading, or script execution tools. Those capabilities are too broad for a default package API and should not be available unless a user intentionally provides their own tool implementation.

## Desired User API

Target import shape:

```python
from ollamanauts import Agent

agent = Agent(
    model="gemma4:31b",
    extra_instructions="You are a helpful data analysis agent.",
)

reply = agent.run("Summarize what you can help with.")
print(reply)
```

Advanced configuration should allow users to provide explicit tools:

```python
from ollamanauts import Agent

def lookup_customer(customer_id: str) -> dict[str, str]:
    """Look up one customer by ID."""
    return {"id": customer_id, "status": "active"}

agent = Agent(
    model="gemma4:31b",
    extra_instructions="Use customer tools only when needed.",
    tools=[lookup_customer],
    enable_subagents=True,
    think_mode="medium",
)
```

## 1. Create a Package Directory

Move the importable code out of repo-root modules and into a package directory:

```text
ollamanauts/
  __init__.py
  agent.py
  tool_orchestrator.py
  terminal_output.py
  prompts/
    __init__.py
    interactive_agent.md
    subagent.md
  tools/
    __init__.py
    subagents.py
```

Do not include:

```text
legacy script workspaces
local code-generation tools
local code-inspection tools
local code-execution tools
```

Move CLI logic to:

```text
ollamanauts/cli.py
```

Keep root `main.py` as a small development wrapper if useful.

## 2. Define a Public Agent Class

Add a user-facing `Agent` class that sets up the common defaults:

- model
- think mode
- prompt composition
- user-supplied tools
- optional subagent tool
- tool orchestrator

Keep lower-level classes internally available:

- `BaseAgent`: shared Ollama/tool loop
- `InteractiveAgent`: terminal-oriented agent with callbacks
- `SubAgent`: non-interactive delegated worker
- `Agent`: package-friendly convenience class

The public `Agent` should avoid terminal assumptions. It should return strings and optionally accept callbacks.

Example constructor shape:

```python
class Agent:
    def __init__(
        self,
        *,
        model: str = "gemma4:31b",
        system_prompt: str | None = None,
        extra_instructions: str | None = None,
        think_mode: bool | str | None = "medium",
        tools: Sequence[Callable[..., Any]] | None = None,
        enable_subagents: bool = True,
    ) -> None:
        ...

    def run(self, prompt: str) -> str:
        ...

    def reset(self) -> None:
        ...
```

## 3. Use Safe Tool Defaults

The package should not provide default tools that can read, write, or execute arbitrary local code.

Recommended default behavior:

- `tools=None` means no user action tools.
- `enable_subagents=True` adds only `deploy_subagent`.
- Subagents should not receive `deploy_subagent`, preventing recursive delegation by default.
- User-supplied tools are passed explicitly through `tools=[...]`.

The package may still expose `ToolOrchestrator` and `ToolResult` for advanced users.

## 4. Clarify Prompt Composition

Users should be able to add their own instructions without losing necessary agent/delegation rules.

Support two prompt modes:

1. `system_prompt`: full replacement prompt.
2. `extra_instructions`: appended user-specific instructions.

Recommended API:

```python
agent = Agent(
    extra_instructions="Always answer in JSON.",
)
```

Internally:

```text
default interactive prompt

Additional user instructions:
...
```

This is safer than making every user copy the tool-use and delegation rules.

## 5. Load Prompt Files as Package Resources

Do not rely on `Path(__file__).with_name("prompts")` once the project is packaged, especially if wheels or zip imports are used.

Use `importlib.resources`:

```python
from importlib.resources import files

def load_prompt(filename: str) -> str:
    return files("ollamanauts.prompts").joinpath(filename).read_text(encoding="utf-8")
```

Add `__init__.py` to `ollamanauts/prompts/` or configure package data so `.md` files are included.

## 6. Update Imports to Package-Relative Imports

Once files move under `ollamanauts/`, update imports like:

```python
from tool_orchestrator import ToolOrchestrator
```

to:

```python
from .tool_orchestrator import ToolOrchestrator
```

Inside `ollamanauts/tools/`, use:

```python
from ..agent import SubAgent
from ..tool_orchestrator import ToolOrchestrator
```

## 7. Configure pyproject.toml for Packaging

Change:

```toml
[tool.uv]
package = false
```

to package-enabled metadata.

Use `Ollamanauts` as the project name:

```toml
[project]
name = "Ollamanauts"
version = "0.1.0"
description = "Composable Ollama agents with safe defaults"
requires-python = ">=3.13"
dependencies = [
    "ollama",
]
```

Review dependencies before publishing. Remove libraries that are no longer used by the package.

Add a build backend:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Add package data for prompts:

```toml
[tool.hatch.build.targets.wheel]
packages = ["ollamanauts"]

[tool.hatch.build.targets.wheel.force-include]
"ollamanauts/prompts" = "ollamanauts/prompts"
```

Or use Hatch's package-data configuration if preferred.

## 8. Add a Console Script Entry Point

Expose the terminal app as an installed command:

```toml
[project.scripts]
ollamanauts = "ollamanauts.cli:main"
```

Then users can run:

```bash
ollamanauts --model gemma4:31b
```

The CLI should instantiate `InteractiveAgent` or the package-level `Agent` with terminal callbacks.

## 9. Expose a Stable Public API

Create `ollamanauts/__init__.py`:

```python
from .agent import Agent
from .agent import BaseAgent
from .agent import InteractiveAgent
from .agent import SubAgent
from .tool_orchestrator import ToolOrchestrator
from .tool_orchestrator import ToolResult

__all__ = [
    "Agent",
    "BaseAgent",
    "InteractiveAgent",
    "SubAgent",
    "ToolOrchestrator",
    "ToolResult",
]
```

Keep this API small and intentional.

## 10. Add Tests Around Package Behavior

Add focused tests for:

- package imports work
- prompt files load through package resources
- custom `system_prompt` replaces the default prompt
- `extra_instructions` appends to the default prompt
- no dangerous local script tools are registered by default
- user-supplied tools are registered when passed explicitly
- subagent tool is registered only on the main agent, not on subagents

Mock `ollama.chat` for agent loop tests so tests do not require a running Ollama server.

## 11. Migration Order

Recommended implementation order:

1. Delete legacy script-tool modules and the example script workspace.
2. Remove script instructions from prompts.
3. Create `ollamanauts/` package directory and move modules.
4. Convert imports to package-relative imports.
5. Move CLI code into `ollamanauts/cli.py`.
6. Add `ollamanauts/__init__.py`.
7. Replace prompt loading with `importlib.resources`.
8. Add package-level `Agent` convenience class.
9. Ensure `Agent(tools=None)` registers no dangerous tools by default.
10. Add optional `deploy_subagent` registration when `enable_subagents=True`.
11. Update `pyproject.toml` to enable packaging.
12. Add console script entry point.
13. Add tests.
14. Run `uv build` and test installing the wheel locally.

## 12. Open Design Decisions

- Package name: use `Ollamanauts` as the project name and `ollamanauts` as the import name.
- Prompt customization: support both full replacement and appended instructions.
- Default tools: no local file, code-generation, or code-execution tools by default.
- User tools: require users to explicitly pass any tool with side effects.
- Public class name: expose `Agent` as the ergonomic default and keep `InteractiveAgent` for CLI-specific usage.
- Subagent logging: package-level `Agent` should not print by default; terminal logging should remain a CLI concern.
