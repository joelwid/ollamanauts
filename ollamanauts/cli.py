from __future__ import annotations

import argparse
import os

from .agent import InteractiveAgent


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable extra runtime traces (subagent/tool/thinking events).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    think_mode = None if args.think == "off" else args.think
    agent = InteractiveAgent(model=args.model, think_mode=think_mode, verbose=args.verbose)

    print(f"Model: {args.model}")
    print(f"Thinking: {args.think}")
    agent.interact()


if __name__ == "__main__":
    main()
