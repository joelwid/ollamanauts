from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agent import SubAgent
from tool_orchestrator import ToolOrchestrator

from .create_script import create_script
from .execute_script import execute_script
from .list_scripts import list_scripts
from .read_script import read_script


SUBAGENT_TOOLS = [
    create_script,
    execute_script,
    list_scripts,
    read_script,
]


def make_deploy_subagent_tool(
    *,
    model: str,
    think_mode: bool | str | None,
    tools: Sequence[Callable[..., Any]] = SUBAGENT_TOOLS,
) -> Callable[[str], str]:
    def deploy_subagent(task: str) -> str:
        """Deploy a non-interactive subagent for a focused task.

        Args:
            task: A clear, self-contained task for the subagent.

        Returns:
            The subagent's concise result.
        """
        stripped_task = task.strip()
        if not stripped_task:
            raise ValueError("task must not be empty")

        agent = SubAgent(
            model=model,
            orchestrator=ToolOrchestrator(tools),
            think_mode=think_mode,
        )
        return agent.run(stripped_task)

    return deploy_subagent
