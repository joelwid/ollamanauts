from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import ollama

from tool_orchestrator import ToolOrchestrator
from tool_orchestrator import ToolResult

SYSTEM_PROMPT = """You are a pragmatic terminal agent.
Work step by step, use tools when they would improve accuracy, and keep answers concise.
When you inspect local files, prefer the filesystem tools over making assumptions.
Only call tools that are necessary for the current user request.
"""


@dataclass
class Agent:
    model: str
    orchestrator: ToolOrchestrator
    system_prompt: str = SYSTEM_PROMPT
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def run_turn(
        self,
        user_input: str,
        on_tool_result: Callable[[ToolResult], None] | None = None,
    ) -> str:
        self.messages.append({"role": "user", "content": user_input})
        final_text = ""

        while True:
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.orchestrator.ollama_tools(),
            )
            message = response.message
            assistant_message = {
                "role": message.role,
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]

            self.messages.append(assistant_message)

            if message.content:
                final_text = message.content

            if not message.tool_calls:
                return final_text

            tool_results = self.orchestrator.execute(message.tool_calls)
            for result in tool_results:
                if on_tool_result is not None:
                    on_tool_result(result)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_name": result.name,
                        "content": result.content,
                    }
                )

    def describe_tools(self) -> list[str]:
        return self.orchestrator.tool_names()
