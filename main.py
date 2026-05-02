from __future__ import annotations

import argparse
import json
import os

import ollama

from agent import Agent
from tool_orchestrator import ToolOrchestrator
from tool_orchestrator import ToolResult
from tools import DEFAULT_TOOLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a terminal Ollama agent.")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "gemma4:31b"),
        help="Ollama model name. Defaults to $OLLAMA_MODEL or gemma4:31b.",
    )
    parser.add_argument(
        "--think",
        choices=["off", "low", "medium", "high"],
        default=os.environ.get("OLLAMA_THINK", "medium"),
        help="Thinking mode for models that support it. Defaults to $OLLAMA_THINK or medium.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    think_mode = None if args.think == "off" else args.think
    agent = Agent(
        model=args.model,
        orchestrator=ToolOrchestrator(DEFAULT_TOOLS),
        think_mode=think_mode,
    )

    print(f"Model: {args.model}")
    print(f"Thinking: {args.think}")
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
            for tool_name in agent.describe_tools():
                print(f"- {tool_name}")
            continue
        if user_input == "/clear":
            agent.reset()
            print("Conversation cleared.")
            continue

        try:
            reply = agent.run_turn(
                user_input,
                on_tool_result=print_tool_result,
                on_thinking_chunk=print_thinking_chunk,
                on_thinking_end=finish_thinking,
            )
            print(f"\n{reply}")
        except ollama.ResponseError as exc:
            finish_thinking()
            print(f"\nOllama API error: {exc}")
        except ollama.RequestError as exc:
            finish_thinking()
            print(f"\nConnection error: {exc}")


if __name__ == "__main__":
    main()
