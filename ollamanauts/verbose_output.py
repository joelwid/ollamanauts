from __future__ import annotations

import sys
from typing import TextIO

from .tool_orchestrator import ToolResult


class VerbosePrinter:
    """Terminal printer for verbose agent/subagent runtime events."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream: TextIO = stream or sys.stdout
        self._thinking_open = False
        self._subagent_thinking_open = False


    def on_token_budget(self, *, estimated_tokens: int, max_context_tokens: int | None = None) -> None:
        if max_context_tokens is None:
            print(f"\n[tokens] estimated={estimated_tokens}", file=self._stream)
            return

        usage_percent = (estimated_tokens / max_context_tokens) * 100
        print(
            f"\n[tokens] estimated={estimated_tokens} / {max_context_tokens} ({usage_percent:.1f}%)",
            file=self._stream,
        )
    def on_compaction_needed(
        self,
        estimated_tokens: int,
        max_context_tokens: int,
        compact_threshold: float,
        compact_target: float,
    ) -> None:
        usage_percent = (estimated_tokens / max_context_tokens) * 100
        threshold_percent = compact_threshold * 100
        target_percent = compact_target * 100
        print(
            f"\n[compact] usage={usage_percent:.1f}% threshold={threshold_percent:.1f}% target={target_percent:.1f}%",
            file=self._stream,
        )

    def on_tool_result(self, result: ToolResult) -> None:
        status = "ok" if result.ok else "error"
        print(f"\n[{status}] {result.name}({result.arguments})", file=self._stream)

    def on_thinking_chunk(self, chunk: str) -> None:
        if not self._thinking_open:
            print("\n[thinking]", file=self._stream)
            self._thinking_open = True
        print(chunk, end="", file=self._stream)

    def on_thinking_end(self) -> None:
        if self._thinking_open:
            print("\n[end thinking]", file=self._stream)
            self._thinking_open = False

    def on_subagent_start(self, task: str) -> None:
        print(f"\n[subagent] {task}", file=self._stream)

    def on_subagent_tool_result(self, result: ToolResult) -> None:
        status = "ok" if result.ok else "error"
        print(f"\n[subagent {status}] {result.name}({result.arguments})", file=self._stream)

    def on_subagent_thinking_chunk(self, chunk: str) -> None:
        if not self._subagent_thinking_open:
            print("\n[subagent thinking]", file=self._stream)
            self._subagent_thinking_open = True
        print(chunk, end="", file=self._stream)

    def on_subagent_thinking_end(self) -> None:
        if self._subagent_thinking_open:
            print("\n[end subagent thinking]", file=self._stream)
            self._subagent_thinking_open = False

    def on_subagent_token_budget(
        self,
        *,
        estimated_tokens: int,
        max_context_tokens: int | None = None,
    ) -> None:
        if max_context_tokens is None:
            print(f"\n[subagent tokens] estimated={estimated_tokens}", file=self._stream)
            return

        usage_percent = (estimated_tokens / max_context_tokens) * 100
        print(
            f"\n[subagent tokens] estimated={estimated_tokens} / {max_context_tokens} ({usage_percent:.1f}%)",
            file=self._stream,
        )

    def on_subagent_compaction_needed(
        self,
        estimated_tokens: int,
        max_context_tokens: int,
        compact_threshold: float,
        compact_target: float,
    ) -> None:
        usage_percent = (estimated_tokens / max_context_tokens) * 100
        threshold_percent = compact_threshold * 100
        target_percent = compact_target * 100
        print(
            f"\n[subagent compact] usage={usage_percent:.1f}% threshold={threshold_percent:.1f}% target={target_percent:.1f}%",
            file=self._stream,
        )

    def on_subagent_result(self, result: str) -> None:
        if result:
            print("\n[subagent result]", file=self._stream)
            print(result, file=self._stream)
            print("[end subagent result]", file=self._stream)
