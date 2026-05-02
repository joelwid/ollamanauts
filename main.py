from __future__ import annotations

import argparse
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
        default=os.environ.get("OLLAMA_MODEL", "gemma4:latest"),
        help="Ollama model name. Defaults to $OLLAMA_MODEL or gemma4:latest.",
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


def main() -> None:
    args = parse_args()
    agent = Agent(model=args.model, orchestrator=ToolOrchestrator(DEFAULT_TOOLS))

    print(f"Model: {args.model}")
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
            reply = agent.run_turn(user_input, on_tool_result=print_tool_result)
            print(f"\n{reply}")
        except ollama.ResponseError as exc:
            print(f"\nOllama API error: {exc}")
        except ollama.RequestError as exc:
            print(f"\nConnection error: {exc}")


if __name__ == "__main__":
    main()
