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
_subagent_thinking_open = False


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


def print_subagent_start(task: str) -> None:
    print(f"\n[subagent] {task}")


def print_subagent_tool_result(result: ToolResult) -> None:
    status = "ok" if result.ok else "error"
    print(f"\n[subagent {status}] {result.name}({result.arguments})")
    if result.name != "execute_script":
        return

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        print("[subagent tool stdout]")
        print(result.content)
        print("[end subagent tool stdout]")
        return

    stdout = payload.get("stdout", "")
    if stdout:
        print("[subagent tool stdout]")
        print(stdout, end="" if stdout.endswith("\n") else "\n")
        print("[end subagent tool stdout]")


def print_subagent_thinking_chunk(chunk: str) -> None:
    global _subagent_thinking_open
    if not _subagent_thinking_open:
        print("\n[subagent thinking]")
        _subagent_thinking_open = True
    print(chunk, end="")


def finish_subagent_thinking() -> None:
    global _subagent_thinking_open
    if _subagent_thinking_open:
        print("\n[end subagent thinking]")
        _subagent_thinking_open = False


def print_subagent_result(result: str) -> None:
    if result:
        print("\n[subagent result]")
        print(result)
        print("[end subagent result]")
