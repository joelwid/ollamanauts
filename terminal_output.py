from __future__ import annotations

import json

from tool_orchestrator import ToolResult


def print_help() -> None:
    print("Commands:")
    print("  /help   Show available commands")
    print("  /tools  List registered tools")
    print("  /clear  Reset conversation history")
    print("  /exit   Quit the agent")


def print_tool_result(result: ToolResult) -> None:
    status = "ok" if result.ok else "error"
    print(f"\n[{status}] {result.name}({result.arguments})")
    if result.name != "execute_script":
        return

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        print("[tool stdout]")
        print(result.content)
        print("[end tool stdout]")
        return

    stdout = payload.get("stdout", "")
    if stdout:
        print("[tool stdout]")
        print(stdout, end="" if stdout.endswith("\n") else "\n")
        print("[end tool stdout]")


_thinking_open = False


def print_thinking_chunk(chunk: str) -> None:
    global _thinking_open
    if not _thinking_open:
        print("\n[thinking]")
        _thinking_open = True
    print(chunk, end="")


def finish_thinking() -> None:
    global _thinking_open
    if _thinking_open:
        print("\n[end thinking]")
        _thinking_open = False
