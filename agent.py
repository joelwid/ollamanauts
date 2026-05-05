from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ollama

from tool_orchestrator import ToolOrchestrator
from tool_orchestrator import ToolResult

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.md")
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class Agent:
    model: str
    orchestrator: ToolOrchestrator
    think_mode: bool | str | None = "medium"
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
        on_thinking_chunk: Callable[[str], None] | None = None,
        on_thinking_end: Callable[[], None] | None = None,
    ) -> str:
        self.messages.append({"role": "user", "content": user_input})
        final_text = ""

        while True:
            response_stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.orchestrator.ollama_tools(),
                stream=True,
                think=self.think_mode,
            )
            role = "assistant"
            content_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            thinking_open = False

            for chunk in response_stream:
                message = chunk.message
                role = message.role or role

                if message.thinking:
                    thinking_parts.append(message.thinking)
                    if on_thinking_chunk is not None:
                        on_thinking_chunk(message.thinking)
                    thinking_open = True

                if message.content:
                    content_parts.append(message.content)
                    if thinking_open and on_thinking_end is not None:
                        on_thinking_end()
                        thinking_open = False

                if message.tool_calls:
                    tool_calls = [tool_call.model_dump() for tool_call in message.tool_calls]

            if thinking_open and on_thinking_end is not None:
                on_thinking_end()

            message = {
                "role": role,
                "content": "".join(content_parts),
                "thinking": "".join(thinking_parts) or None,
            }
            if tool_calls:
                message["tool_calls"] = tool_calls

            assistant_message = {
                "role": message["role"],
                "content": message["content"],
            }
            if message["thinking"]:
                assistant_message["thinking"] = message["thinking"]
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            self.messages.append(assistant_message)

            if message["content"]:
                final_text = message["content"]

            if not tool_calls:
                return final_text

            tool_results = self.orchestrator.execute(
                [ollama.Message.ToolCall.model_validate(tool_call) for tool_call in tool_calls]
            )
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
