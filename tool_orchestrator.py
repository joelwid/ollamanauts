from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    name: str
    arguments: dict[str, Any]
    content: str
    ok: bool = True


class ToolOrchestrator:
    def __init__(self, tools: Sequence[Callable[..., Any]]) -> None:
        self._tools = {tool.__name__: tool for tool in tools}

    def ollama_tools(self) -> list[Callable[..., Any]]:
        return list(self._tools.values())

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, tool_calls: Sequence[Any]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for tool_call in tool_calls:
            function_call = tool_call.function
            name = function_call.name
            arguments = dict(function_call.arguments)

            tool = self._tools.get(name)
            if tool is None:
                results.append(
                    ToolResult(
                        name=name,
                        arguments=arguments,
                        content=f"Unknown tool: {name}",
                        ok=False,
                    )
                )
                continue

            try:
                value = tool(**arguments)
                content = self._serialize(value)
                results.append(ToolResult(name=name, arguments=arguments, content=content))
            except Exception as exc:  # pragma: no cover - defensive path
                results.append(
                    ToolResult(
                        name=name,
                        arguments=arguments,
                        content=f"{type(exc).__name__}: {exc}",
                        ok=False,
                    )
                )

        return results

    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, sort_keys=True, default=str)
