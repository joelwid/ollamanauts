from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.resources import files
import random
import string
from typing import Any
from typing import TypeVar

import ollama

from .token_usage import estimate_messages_tokens
from .token_usage import should_compact
from .tool_orchestrator import ToolOrchestrator
from .tool_orchestrator import ToolResult


def load_prompt(filename: str) -> str:
    return files("ollamanauts.prompts").joinpath(filename).read_text(encoding="utf-8")


INTERACTIVE_AGENT_PROMPT = load_prompt("interactive_agent.md")
SUBAGENT_PROMPT = load_prompt("subagent.md")


T = TypeVar("T")


def _random_suffix(length: int = 4) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choices(alphabet, k=length))


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
    """Core stateful agent engine used by interactive and subagent wrappers.

    Args:
        model: Ollama model name used for generation and (by default) compaction.
        orchestrator: Tool registry/executor used to expose callable tools.
        think_mode: Thinking mode value forwarded to ``ollama.chat``.
        system_prompt: System instruction stored as the first conversation message.
        messages: Internal conversation message list; reset during initialization.
        max_context_tokens: Optional context budget used to trigger compaction.
        on_token_budget: Optional callback receiving estimated and max token counts.
        on_compaction_needed: Optional callback invoked before each compaction pass.
        enable_auto_compaction: Enables automatic context compaction when over budget.
        compact_threshold: Fraction of max context that triggers compaction checks.
        compact_target: Fraction of max context to target after compaction.
        compaction_preserve_last_n_turns: Number of recent user/assistant turns to keep verbatim.
        compaction_model: Optional model used for summarization during compaction.
        max_compaction_passes: Maximum sequential compaction passes per turn.
    """
    model: str
    orchestrator: ToolOrchestrator
    think_mode: bool | str | None = "medium"
    system_prompt: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_context_tokens: int | None = None
    on_token_budget: Callable[[int, int | None], None] | None = None
    on_compaction_needed: Callable[[int, int, float, float], None] | None = None
    enable_auto_compaction: bool = True
    compact_threshold: float = 0.85
    compact_target: float = 0.60
    compaction_preserve_last_n_turns: int = 4
    compaction_model: str | None = None
    max_compaction_passes: int = 3

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

            if self.enable_auto_compaction and should_compact(
                estimated_tokens=estimated.estimated_tokens,
                max_context_tokens=self.max_context_tokens,
                compact_threshold=self.compact_threshold,
            ):
                if self.on_compaction_needed is not None:
                    self.on_compaction_needed(
                        estimated.estimated_tokens,
                        self.max_context_tokens or 0,
                        self.compact_threshold,
                        self.compact_target,
                    )
                self._compact_until_within_budget()

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

    def _compact_until_within_budget(self) -> None:
        if self.max_context_tokens is None or self.max_context_tokens <= 0:
            return

        pass_index = 0
        preserve_turns = self.compaction_preserve_last_n_turns
        while pass_index < self.max_compaction_passes and should_compact(
            estimated_tokens=estimate_messages_tokens(self.messages).estimated_tokens,
            max_context_tokens=self.max_context_tokens,
            compact_threshold=self.compact_target,
        ):
            self._compact_once(preserve_last_n_turns=preserve_turns)
            pass_index += 1
            preserve_turns = 0

    def _compact_once(self, *, preserve_last_n_turns: int) -> None:
        if len(self.messages) <= 2:
            return

        system_message = self.messages[0]
        non_system = self.messages[1:]
        preserve_count = max(0, preserve_last_n_turns * 2)
        kept_tail = non_system[-preserve_count:] if preserve_count > 0 else []
        kept_tail = self._expand_tail_for_tool_integrity(non_system=non_system, kept_tail=kept_tail)
        protected_indices = self._collect_unresolved_tool_indices(non_system)
        compaction_boundary = len(non_system) - len(kept_tail) if kept_tail else len(non_system)
        summarize_candidates = non_system[:compaction_boundary]
        protected_messages = [non_system[i] for i in sorted(protected_indices) if i < compaction_boundary]
        to_summarize = [
            message
            for i, message in enumerate(summarize_candidates)
            if i not in protected_indices
        ]
        kept_tail = [*protected_messages, *kept_tail]
        if not to_summarize:
            return

        summary = self._summarize_messages(to_summarize)
        summary_message = {
            "role": "system",
            "content": (
                "Conversation summary for context compaction:\n"
                f"{summary}"
            ),
        }
        self.messages = [system_message, summary_message, *kept_tail]

    def _expand_tail_for_tool_integrity(
        self,
        *,
        non_system: list[dict[str, Any]],
        kept_tail: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not kept_tail:
            if not non_system:
                return kept_tail
            boundary_index = len(non_system)
            while boundary_index > 0 and non_system[boundary_index - 1].get("role") == "tool":
                boundary_index -= 1
                if boundary_index > 0 and non_system[boundary_index - 1].get("tool_calls"):
                    boundary_index -= 1
            return non_system[boundary_index:]

        boundary_index = len(non_system) - len(kept_tail)
        while boundary_index > 0:
            previous = non_system[boundary_index - 1]
            first_tail = non_system[boundary_index]

            previous_has_tool_calls = bool(previous.get("tool_calls"))
            first_tail_is_tool_result = first_tail.get("role") == "tool"
            if previous_has_tool_calls and first_tail_is_tool_result:
                boundary_index -= 1
                continue

            previous_is_tool_result = previous.get("role") == "tool"
            first_tail_has_tool_calls = bool(first_tail.get("tool_calls"))
            if previous_is_tool_result and first_tail_has_tool_calls:
                boundary_index -= 1
                continue

            break

        return non_system[boundary_index:]

    def _collect_unresolved_tool_indices(self, non_system: list[dict[str, Any]]) -> set[int]:
        protected: set[int] = set()
        index = 0
        while index < len(non_system):
            message = non_system[index]
            if not message.get("tool_calls"):
                index += 1
                continue

            protected.add(index)
            cursor = index + 1
            consumed_tool_result = False
            while cursor < len(non_system) and non_system[cursor].get("role") == "tool":
                protected.add(cursor)
                consumed_tool_result = True
                cursor += 1

            if consumed_tool_result and cursor < len(non_system):
                follow_up = non_system[cursor]
                if follow_up.get("role") == "assistant":
                    protected.add(cursor)

            if not consumed_tool_result:
                protected.add(index)
            index = cursor if cursor > index else index + 1

        return protected

    def _summarize_messages(self, messages: list[dict[str, Any]]) -> str:
        serialized = "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            for message in messages
        )
        response = ollama.chat(
            model=self.compaction_model or self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the following conversation for future context.\n"
                        "Output exactly these markdown sections in order:\n"
                        "## Facts\n"
                        "## Decisions\n"
                        "## Constraints\n"
                        "## Open Questions\n"
                        "## Pending Actions\n"
                        "Use concise bullet points under each section.\n"
                        "If a section has no items, write '- None'.\n"
                        "Do not invent details."
                    ),
                },
                {"role": "user", "content": serialized},
            ],
            stream=False,
            think=False,
        )
        content = response.message.content if getattr(response, "message", None) is not None else ""
        return content or "No summary available."

    def describe_tools(self) -> list[str]:
        return self.orchestrator.tool_names()


class InteractiveAgent:
    """Interactive convenience wrapper around :class:`Agent`.

    This class keeps conversation state across multiple ``run`` calls and
    provides an optional terminal input loop with slash-command handling.

    Args:
        model: Ollama model name to use for chat completions.
        system_prompt: Full replacement system prompt; overrides default prompt assembly.
        extra_instructions: Extra instructions appended to the default system prompt when
            ``system_prompt`` is not provided.
        think_mode: Thinking mode passed through to Ollama.
        tools: Additional top-level tools to register alongside package defaults.
        subagent_tools: Explicit tool list for subagents; defaults to top-level tools.
        verbose: Enables terminal runtime output for thinking/tool/subagent events.
        enable_subagents: When true, register the ``deploy_subagent`` tool.
        max_context_tokens: Optional token budget for the primary agent context.
        subagent_max_context_tokens: Optional token budget for spawned subagents.
        enable_auto_compaction: Enables automatic context compaction for the primary agent.
        compact_threshold: Compaction trigger threshold for the primary agent.
        compact_target: Post-compaction target threshold for the primary agent.
        compaction_preserve_last_n_turns: Number of recent turns preserved for the
            primary agent during compaction.
        compaction_model: Optional summarization model for primary agent compaction.
        subagent_enable_auto_compaction: Optional override for subagent auto-compaction.
        subagent_compact_threshold: Optional override for subagent compaction trigger.
        subagent_compact_target: Optional override for subagent compaction target.
        subagent_compaction_preserve_last_n_turns: Optional override for how many
            recent subagent turns are preserved during compaction.
        subagent_compaction_model: Optional override summarization model for subagents.
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
        name: str | None = None,
    ) -> None:
        effective_name = name or f"IA-{_random_suffix()}"
        effective_system_prompt = INTERACTIVE_AGENT_PROMPT if system_prompt is None else system_prompt
        self._verbose = verbose
        self._agent = Agent(
            model=model,
            system_prompt=effective_system_prompt,
            extra_instructions=extra_instructions,
            think_mode=think_mode,
            tools=tools,
            subagent_tools=subagent_tools,
            verbose=verbose,
            enable_subagents=enable_subagents,
            max_context_tokens=max_context_tokens,
            subagent_max_context_tokens=subagent_max_context_tokens,
            enable_auto_compaction=enable_auto_compaction,
            compact_threshold=compact_threshold,
            compact_target=compact_target,
            compaction_preserve_last_n_turns=compaction_preserve_last_n_turns,
            compaction_model=compaction_model,
            subagent_enable_auto_compaction=subagent_enable_auto_compaction,
            subagent_compact_threshold=subagent_compact_threshold,
            subagent_compact_target=subagent_compact_target,
            subagent_compaction_preserve_last_n_turns=subagent_compaction_preserve_last_n_turns,
            subagent_compaction_model=subagent_compaction_model,
            name=effective_name,
        )
        self.name = self._agent.name

    def run(
        self,
        prompt: str,
        *,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_thinking_chunk: Callable[[str], None] | None = None,
        on_thinking_end: Callable[[], None] | None = None,
    ) -> str:
        return self._agent.run(
            prompt,
            on_tool_result=on_tool_result,
            on_thinking_chunk=on_thinking_chunk,
            on_thinking_end=on_thinking_end,
        )

    def reset(self) -> None:
        self._agent.reset()

    def describe_tools(self) -> list[str]:
        return self._agent.describe_tools()

    def interact(self) -> None:
        from .terminal_output import finish_thinking
        from .terminal_output import print_help
        from .terminal_output import print_thinking_chunk
        from .terminal_output import print_tool_result

        print_help()
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return

            if not user_input:
                continue

            if user_input == "/exit":
                print("Exiting.")
                return
            if user_input == "/help":
                print_help()
                continue
            if user_input == "/tools":
                for tool_name in self.describe_tools():
                    print(f"- {tool_name}")
                continue
            if user_input == "/clear":
                self.reset()
                print("Conversation cleared.")
                continue
            if user_input == "/compact":
                self._agent._compact()
                print("Conversation compacted.")
                continue

            try:
                on_tool_result = None if self._verbose else print_tool_result
                on_thinking_chunk = None if self._verbose else print_thinking_chunk
                on_thinking_end = None if self._verbose else finish_thinking
                reply = self.run(
                    user_input,
                    on_tool_result=on_tool_result,
                    on_thinking_chunk=on_thinking_chunk,
                    on_thinking_end=on_thinking_end,
                )
                print(f"\n{reply}")
            except ollama.ResponseError as exc:
                finish_thinking()
                print(f"\nOllama API error: {exc}")
            except ollama.RequestError as exc:
                finish_thinking()
                print(f"\nConnection error: {exc}")


@dataclass
class SubAgent(BaseAgent):
    """Specialized :class:`BaseAgent` that runs focused one-shot subagent tasks."""
    system_prompt: str = SUBAGENT_PROMPT
    name: str = field(default_factory=lambda: f"SA-{_random_suffix()}")

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
            system prompt. Ignored when ``system_prompt`` is provided.
        think_mode: Thinking mode passed through to Ollama.
        tools: Explicit user-supplied tools to register.
        subagent_tools: Optional explicit tools made available to subagents.
            Defaults to the same non-subagent tools available to this agent.
        verbose: Enables terminal runtime output for thinking/tool/subagent events.
        enable_subagents: When true, register only the ``deploy_subagent`` tool in
            addition to any explicit ``tools``.
        max_context_tokens: Optional token budget for primary-agent context.
        subagent_max_context_tokens: Optional token budget for subagent context.
            When omitted, subagents inherit ``max_context_tokens``.
        enable_auto_compaction: Enables automatic context compaction for the
            primary agent.
        compact_threshold: Fraction of max context that triggers compaction for
            the primary agent.
        compact_target: Fraction of max context to target after primary-agent
            compaction.
        compaction_preserve_last_n_turns: Number of recent turns kept verbatim
            during primary-agent compaction.
        compaction_model: Optional model used for primary-agent summarization.
        subagent_enable_auto_compaction: Optional override for whether spawned
            subagents auto-compact context.
        subagent_compact_threshold: Optional override trigger threshold for
            subagent compaction.
        subagent_compact_target: Optional override post-compaction target for
            subagents.
        subagent_compaction_preserve_last_n_turns: Optional override for how many
            recent subagent turns are kept verbatim during compaction.
        subagent_compaction_model: Optional override summarization model for
            subagent compaction.
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
        name: str | None = None,
    ) -> None:
        from .tools import DEFAULT_TOOLS
        from .tools import make_deploy_subagent_tool
        from .verbose_output import VerbosePrinter

        self.name = name or f"A-{_random_suffix()}"
        self._verbose_printer: VerbosePrinter | None = (
            VerbosePrinter(agent_name=self.name) if verbose else None
        )
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
                    on_compaction_needed=(
                        self._verbose_printer.on_subagent_compaction_needed
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
            on_compaction_needed=(
                self._verbose_printer.on_compaction_needed if self._verbose_printer is not None else None
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

    def _compact(self) -> None:
        self._agent._compact_until_within_budget()
