from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..agent import SubAgent
from ..tool_orchestrator import ToolOrchestrator
from ..tool_orchestrator import ToolResult


SUBAGENT_TOOLS: tuple[Callable[..., object], ...] = ()


def _filter_nested_subagent_tool(tools: Sequence[Callable[..., Any]]) -> list[Callable[..., Any]]:
    return [tool for tool in tools if getattr(tool, "__name__", "") != "deploy_subagent"]


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
    max_context_tokens: int | None = None,
    on_token_budget: Callable[[int, int | None], None] | None = None,
) -> Callable[[str], str]:
    filtered_tools = _filter_nested_subagent_tool(tools)

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
            orchestrator=ToolOrchestrator(filtered_tools),
            think_mode=think_mode,
            max_context_tokens=max_context_tokens,
            on_token_budget=on_token_budget,
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
