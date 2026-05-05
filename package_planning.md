# Python Package Migration Plan

Goal: make this repository importable as a Python package so users can create a configured agent with a small public API, while still keeping the terminal app available.

## Desired User API

Target import shape:

```python
from python_agent import Agent

agent = Agent(
    model="gemma4:31b",
    system_prompt="You are a helpful data analysis agent.",
)

reply = agent.run("Inspect the available scripts and summarize them.")
print(reply)
```

The package should also support advanced configuration:

```python
from python_agent import Agent
from python_agent.tools import DEFAULT_TOOLS

agent = Agent(
    model="gemma4:31b",
    system_prompt="Custom instructions.",
    tools=DEFAULT_TOOLS,
    enable_subagents=True,
    think_mode="medium",
)
```

## 1. Create a Package Directory

Move the importable code out of repo-root modules and into a package directory:

```text
python_agent/
  __init__.py
  agent.py
  tool_orchestrator.py
  terminal_output.py
  prompts/
    interactive_agent.md
    subagent.md
  tools/
    __init__.py
    create_script.py
    execute_script.py
    list_scripts.py
    read_script.py
    subagents.py
```

Keep `main.py` as a thin development entry point, or move it to:

```text
python_agent/cli.py
```

Recommended: move CLI logic to `python_agent/cli.py` and leave root `main.py` as a small compatibility wrapper if needed.

## 2. Define a Public Agent Class

Add a user-facing `Agent` class that sets up the common defaults:

- model
- think mode
- system prompt
- default tools
- optional subagent tool
- tool orchestrator

This should be the class most users import. Internally, keep the current lower-level classes:

- `BaseAgent`: shared Ollama/tool loop
- `InteractiveAgent`: terminal-oriented agent with callbacks
- `SubAgent`: non-interactive delegated worker
- `Agent`: package-friendly convenience class

The package `Agent` should avoid terminal assumptions. It should return strings and optionally accept callbacks.

Example constructor shape:

```python
class Agent:
    def __init__(
        self,
        *,
        model: str = "gemma4:31b",
        system_prompt: str | None = None,
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

## 3. Clarify Prompt Composition

Users should be able to provide a custom system prompt without losing necessary tool/delegation instructions.

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

This is safer than making every user copy the tool-use rules.

## 4. Load Prompt Files as Package Resources

Do not rely on `Path(__file__).with_name("prompts")` once the project is packaged, especially if wheels or zip imports are used.

Use `importlib.resources`:

```python
from importlib.resources import files

def load_prompt(filename: str) -> str:
    return files("python_agent.prompts").joinpath(filename).read_text(encoding="utf-8")
```

Add `__init__.py` to `python_agent/prompts/` or configure package data so `.md` files are included.

## 5. Make Script Storage Configurable

Current tools assume scripts live in `agent_scripts` under the project root. For a package, users need control over where generated scripts go.

Introduce a script workspace concept:

```python
agent = Agent(script_dir="./agent_scripts")
```

Then update script tools so they are created by factories:

```python
make_create_script_tool(script_dir: Path)
make_read_script_tool(script_dir: Path)
make_list_scripts_tool(script_dir: Path)
make_execute_script_tool(script_dir: Path)
```

This avoids writing into the installed package directory and makes behavior predictable for library users.

## 6. Convert Tools to Configurable Factories

The package should expose both:

- simple default tools for the common case
- factories for custom workspaces

Example:

```python
from python_agent.tools import make_default_tools

tools = make_default_tools(script_dir="./agent_scripts")
```

`make_default_tools(...)` should return:

- create script
- execute script
- list scripts
- read script

Subagent deployment should be added separately by the top-level `Agent` when `enable_subagents=True`, so subagents do not recursively receive `deploy_subagent`.

## 7. Update Imports to Package-Relative Imports

Once files move under `python_agent/`, update imports like:

```python
from tool_orchestrator import ToolOrchestrator
```

to:

```python
from .tool_orchestrator import ToolOrchestrator
```

Inside `python_agent/tools/`, use:

```python
from ..agent import SubAgent
from ..tool_orchestrator import ToolOrchestrator
```

## 8. Configure pyproject.toml for Packaging

Change:

```toml
[tool.uv]
package = false
```

to package-enabled metadata.

Add a build backend:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Add package data for prompts:

```toml
[tool.hatch.build.targets.wheel]
packages = ["python_agent"]

[tool.hatch.build.targets.wheel.force-include]
"python_agent/prompts" = "python_agent/prompts"
```

Or use Hatch's package-data configuration if preferred.

## 9. Add a Console Script Entry Point

Expose the terminal app as an installed command:

```toml
[project.scripts]
python-agent = "python_agent.cli:main"
```

Then users can run:

```bash
python-agent --model gemma4:31b
```

The CLI should instantiate `InteractiveAgent` or the package-level `Agent` with terminal callbacks.

## 10. Expose a Stable Public API

Create `python_agent/__init__.py`:

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

## 11. Add Tests Around Package Behavior

Add focused tests for:

- package imports work
- prompt files load through package resources
- custom `system_prompt` replaces the default prompt
- `extra_instructions` appends to the default prompt
- default tool names are registered
- subagent tool is registered only on the main agent, not on subagents
- script tools write into a configured temp directory

Mock `ollama.chat` for agent loop tests so tests do not require a running Ollama server.

## 12. Migration Order

Recommended implementation order:

1. Create `python_agent/` package directory and move modules.
2. Convert imports to package-relative imports.
3. Move CLI code into `python_agent/cli.py`.
4. Add `python_agent/__init__.py`.
5. Replace prompt loading with `importlib.resources`.
6. Add package-level `Agent` convenience class.
7. Convert script tools to factories that accept `script_dir`.
8. Add `make_default_tools(script_dir=...)`.
9. Update `pyproject.toml` to enable packaging.
10. Add console script entry point.
11. Add tests.
12. Run `uv build` and test installing the wheel locally.

## 13. Open Design Decisions

- Package name: keep `python-agent` as the distribution name, but use `python_agent` as the import name.
- Prompt customization: support both full replacement and appended instructions.
- Default script directory: use `Path.cwd() / "agent_scripts"` unless the user provides `script_dir`.
- Public class name: expose `Agent` as the ergonomic default and keep `InteractiveAgent` for CLI-specific usage.
- Subagent logging: package-level `Agent` should not print by default; terminal logging should remain a CLI concern.
