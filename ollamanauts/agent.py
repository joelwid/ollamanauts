from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import ollama

from .tool_orchestrator import ToolOrchestrator
from .tool_orchestrator import ToolResult


def load_prompt(filename: str) -> str:
    return files("ollamanauts.prompts").joinpath(filename).read_text(encoding="utf-8")


INTERACTIVE_AGENT_PROMPT = load_prompt("interactive_agent.md")
SUBAGENT_PROMPT = load_prompt("subagent.md")


def _compose_system_prompt(
    *,
    system_prompt: str | None,
    extra_instructions: str | None,
) -> str:
    if system_prompt is not None:
        return system_prompt

    if not extra_instructions:
        return INTERACTIVE_AGENT_PROMPT

    return f"{INTERACTIVE_AGENT_PROMPT}\n\nAdditional user instructions:\n{extra_instructions.strip()}"


@dataclass
class BaseAgent:
    model: str
    orchestrator: ToolOrchestrator
    think_mode: bool | str | None = "medium"
    system_prompt: str = ""
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


@dataclass
class InteractiveAgent(BaseAgent):
    system_prompt: str = INTERACTIVE_AGENT_PROMPT


@dataclass
class SubAgent(BaseAgent):
    system_prompt: str = SUBAGENT_PROMPT

    def run(
        self,
        task: str,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_thinking_chunk: Callable[[str], None] | None = None,
        on_thinking_end: Callable[[], None] | None = None,
    ) -> str:
        return self.run_turn(
            task,
            on_tool_result=on_tool_result,
            on_thinking_chunk=on_thinking_chunk,
            on_thinking_end=on_thinking_end,
        )


class Agent:
    """Package-friendly Ollama agent with safe defaults.

    Args:
        model: Ollama model name to use for chat completions.
        system_prompt: Full replacement system prompt. When provided, it replaces
            the agent's entire default system prompt instead of appending to it.
        extra_instructions: Additional instructions appended to the default
            system prompt. Ignored when `system_prompt` is provided.
        think_mode: Thinking mode passed through to Ollama.
        tools: Explicit user-supplied tools to register.
        enable_subagents: When true, register only the `deploy_subagent` tool in
            addition to any explicit `tools`.
    """

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
        from .tools import DEFAULT_TOOLS
        from .tools import make_deploy_subagent_tool

        configured_tools = [*DEFAULT_TOOLS, *(tools or ())]
        if enable_subagents:
            configured_tools.append(
                make_deploy_subagent_tool(
                    model=model,
                    think_mode=think_mode,
                )
            )

        self._agent = BaseAgent(
            model=model,
            orchestrator=ToolOrchestrator(configured_tools),
            think_mode=think_mode,
            system_prompt=_compose_system_prompt(
                system_prompt=system_prompt,
                extra_instructions=extra_instructions,
            ),
        )

    def run(
        self,
        prompt: str,
        *,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_thinking_chunk: Callable[[str], None] | None = None,
        on_thinking_end: Callable[[], None] | None = None,
    ) -> str:
        return self._agent.run_turn(
            prompt,
            on_tool_result=on_tool_result,
            on_thinking_chunk=on_thinking_chunk,
            on_thinking_end=on_thinking_end,
        )

    def reset(self) -> None:
        self._agent.reset()

    def describe_tools(self) -> list[str]:
        return self._agent.describe_tools()
