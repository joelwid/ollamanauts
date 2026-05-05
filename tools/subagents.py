from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agent import SubAgent
from tool_orchestrator import ToolOrchestrator
from tool_orchestrator import ToolResult

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
    on_start: Callable[[str], None] | None = None,
    on_tool_result: Callable[[ToolResult], None] | None = None,
    on_thinking_chunk: Callable[[str], None] | None = None,
    on_thinking_end: Callable[[], None] | None = None,
    on_result: Callable[[str], None] | None = None,
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

        if on_start is not None:
            on_start(stripped_task)

        agent = SubAgent(
            model=model,
            orchestrator=ToolOrchestrator(tools),
            think_mode=think_mode,
        )
        try:
            result = agent.run(
                stripped_task,
                on_tool_result=on_tool_result,
                on_thinking_chunk=on_thinking_chunk,
                on_thinking_end=on_thinking_end,
            )
        finally:
            if on_thinking_end is not None:
                on_thinking_end()

        if on_result is not None:
            on_result(result)
        return result

    return deploy_subagent
