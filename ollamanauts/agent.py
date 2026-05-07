from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any
from typing import TypeVar

import ollama

from .token_usage import estimate_messages_tokens
from .tool_orchestrator import ToolOrchestrator
from .tool_orchestrator import ToolResult


def load_prompt(filename: str) -> str:
    return files("ollamanauts.prompts").joinpath(filename).read_text(encoding="utf-8")


INTERACTIVE_AGENT_PROMPT = load_prompt("interactive_agent.md")
SUBAGENT_PROMPT = load_prompt("subagent.md")


T = TypeVar("T")


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


def _fanout_callbacks(
    first: Callable[[T], None] | None,
    second: Callable[[T], None] | None,
) -> Callable[[T], None] | None:
    if first is None:
        return second
    if second is None:
        return first

    def combined(value: T) -> None:
        first(value)
        second(value)

    return combined


def _fanout_done_callbacks(
    first: Callable[[], None] | None,
    second: Callable[[], None] | None,
) -> Callable[[], None] | None:
    if first is None:
        return second
    if second is None:
        return first

    def combined() -> None:
        first()
        second()

    return combined


@dataclass
class BaseAgent:
    model: str
    orchestrator: ToolOrchestrator
    think_mode: bool | str | None = "medium"
    system_prompt: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_context_tokens: int | None = None
    on_token_budget: Callable[[int, int | None], None] | None = None
    enable_auto_compaction: bool = True
    compact_threshold: float = 0.85
    compact_target: float = 0.60
    compaction_preserve_last_n_turns: int = 4
    compaction_model: str | None = None

    def __post_init__(self) -> None:
        if self.compaction_model is None:
            self.compaction_model = self.model
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
            estimated = estimate_messages_tokens(self.messages)
            if self.on_token_budget is not None:
                self.on_token_budget(
                    estimated_tokens=estimated.estimated_tokens,
                    max_context_tokens=self.max_context_tokens,
                )

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
        subagent_tools: Optional explicit tools made available to subagents.
            Defaults to the same non-subagent tools available to this agent.
        verbose: Enables terminal runtime output for thinking/tool/subagent events.
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
        subagent_tools: Sequence[Callable[..., Any]] | None = None,
        verbose: bool = False,
        enable_subagents: bool = True,
        max_context_tokens: int | None = None,
        subagent_max_context_tokens: int | None = None,
        enable_auto_compaction: bool = True,
        compact_threshold: float = 0.85,
        compact_target: float = 0.60,
        compaction_preserve_last_n_turns: int = 4,
        compaction_model: str | None = None,
        subagent_enable_auto_compaction: bool | None = None,
        subagent_compact_threshold: float | None = None,
        subagent_compact_target: float | None = None,
        subagent_compaction_preserve_last_n_turns: int | None = None,
        subagent_compaction_model: str | None = None,
    ) -> None:
        from .tools import DEFAULT_TOOLS
        from .tools import make_deploy_subagent_tool
        from .verbose_output import VerbosePrinter

        self._verbose_printer: VerbosePrinter | None = VerbosePrinter() if verbose else None
        base_tools = [*DEFAULT_TOOLS, *(tools or ())]
        configured_subagent_tools = [*base_tools] if subagent_tools is None else [*subagent_tools]
        configured_tools = [*base_tools]
        effective_subagent_max_context_tokens = (
            max_context_tokens
            if subagent_max_context_tokens is None
            else subagent_max_context_tokens
        )
        effective_subagent_enable_auto_compaction = (
            enable_auto_compaction
            if subagent_enable_auto_compaction is None
            else subagent_enable_auto_compaction
        )
        effective_subagent_compact_threshold = (
            compact_threshold
            if subagent_compact_threshold is None
            else subagent_compact_threshold
        )
        effective_subagent_compact_target = (
            compact_target
            if subagent_compact_target is None
            else subagent_compact_target
        )
        effective_subagent_compaction_preserve_last_n_turns = (
            compaction_preserve_last_n_turns
            if subagent_compaction_preserve_last_n_turns is None
            else subagent_compaction_preserve_last_n_turns
        )
        effective_subagent_compaction_model = (
            compaction_model
            if subagent_compaction_model is None
            else subagent_compaction_model
        )
        if enable_subagents:
            configured_tools.append(
                make_deploy_subagent_tool(
                    model=model,
                    think_mode=think_mode,
                    tools=configured_subagent_tools,
                    on_start=(
                        self._verbose_printer.on_subagent_start
                        if self._verbose_printer is not None
                        else None
                    ),
                    on_tool_result=(
                        self._verbose_printer.on_subagent_tool_result
                        if self._verbose_printer is not None
                        else None
                    ),
                    on_thinking_chunk=(
                        self._verbose_printer.on_subagent_thinking_chunk
                        if self._verbose_printer is not None
                        else None
                    ),
                    on_thinking_end=(
                        self._verbose_printer.on_subagent_thinking_end
                        if self._verbose_printer is not None
                        else None
                    ),
                    on_result=(
                        self._verbose_printer.on_subagent_result
                        if self._verbose_printer is not None
                        else None
                    ),
                    max_context_tokens=effective_subagent_max_context_tokens,
                    on_token_budget=(
                        self._verbose_printer.on_subagent_token_budget
                        if self._verbose_printer is not None
                        else None
                    ),
                    enable_auto_compaction=effective_subagent_enable_auto_compaction,
                    compact_threshold=effective_subagent_compact_threshold,
                    compact_target=effective_subagent_compact_target,
                    compaction_preserve_last_n_turns=effective_subagent_compaction_preserve_last_n_turns,
                    compaction_model=effective_subagent_compaction_model,
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
            max_context_tokens=max_context_tokens,
            on_token_budget=(
                self._verbose_printer.on_token_budget if self._verbose_printer is not None else None
            ),
            enable_auto_compaction=enable_auto_compaction,
            compact_threshold=compact_threshold,
            compact_target=compact_target,
            compaction_preserve_last_n_turns=compaction_preserve_last_n_turns,
            compaction_model=compaction_model,
        )
        self._verbose = verbose

    def run(
        self,
        prompt: str,
        *,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_thinking_chunk: Callable[[str], None] | None = None,
        on_thinking_end: Callable[[], None] | None = None,
    ) -> str:
        combined_tool_result = _fanout_callbacks(
            on_tool_result,
            self._verbose_printer.on_tool_result if self._verbose_printer is not None else None,
        )
        combined_thinking_chunk = _fanout_callbacks(
            on_thinking_chunk,
            self._verbose_printer.on_thinking_chunk if self._verbose_printer is not None else None,
        )
        combined_thinking_end = _fanout_done_callbacks(
            on_thinking_end,
            self._verbose_printer.on_thinking_end if self._verbose_printer is not None else None,
        )
        return self._agent.run_turn(
            prompt,
            on_tool_result=combined_tool_result,
            on_thinking_chunk=combined_thinking_chunk,
            on_thinking_end=combined_thinking_end,
        )

    def reset(self) -> None:
        self._agent.reset()

    def describe_tools(self) -> list[str]:
        return self._agent.describe_tools()
