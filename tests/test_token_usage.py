from __future__ import annotations

import io
import sys
import types
import unittest


def _install_ollama_stub() -> None:
    if "ollama" in sys.modules:
        return

    class _ToolCall:
        @staticmethod
        def model_validate(value: object) -> object:
            return value

    ollama_stub = types.SimpleNamespace(
        chat=lambda **_: (),
        Message=types.SimpleNamespace(ToolCall=_ToolCall),
    )
    sys.modules["ollama"] = ollama_stub


_install_ollama_stub()

from ollamanauts.token_usage import estimate_message_tokens
from ollamanauts.token_usage import estimate_messages_tokens
from ollamanauts.verbose_output import VerbosePrinter


class _ChunkMessage:
    def __init__(self) -> None:
        self.role = "assistant"
        self.content = "done"
        self.thinking = ""
        self.tool_calls = []


class _Chunk:
    def __init__(self) -> None:
        self.message = _ChunkMessage()


class TokenUsageTests(unittest.TestCase):
    def test_estimate_messages_tokens_reports_count_and_tokens(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello world"},
        ]

        estimate = estimate_messages_tokens(messages)

        self.assertEqual(estimate.message_count, 2)
        self.assertGreater(estimate.estimated_tokens, 0)

    def test_estimate_message_tokens_includes_tool_calls_and_thinking(self) -> None:
        message = {
            "role": "assistant",
            "content": "Result",
            "thinking": "Internal chain",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"id": "1"}}}],
        }

        tokens = estimate_message_tokens(message)

        self.assertGreater(tokens, 4)


    def test_base_agent_token_callback_uses_keyword_arguments(self) -> None:
        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        captured: dict[str, int | None] = {}

        def on_token_budget(*, estimated_tokens: int, max_context_tokens: int | None = None) -> None:
            captured["estimated_tokens"] = estimated_tokens
            captured["max_context_tokens"] = max_context_tokens

        with unittest.mock.patch("ollamanauts.agent.ollama.chat", return_value=[_Chunk()]):
            agent = BaseAgent(
                model="dummy",
                orchestrator=ToolOrchestrator([]),
                system_prompt="system",
                max_context_tokens=1024,
                on_token_budget=on_token_budget,
            )
            result = agent.run_turn("hello")

        self.assertEqual(result, "done")
        self.assertIn("estimated_tokens", captured)
        self.assertEqual(captured["max_context_tokens"], 1024)

    def test_verbose_printer_token_budget_output(self) -> None:
        stream = io.StringIO()
        printer = VerbosePrinter(stream=stream)

        printer.on_token_budget(estimated_tokens=240, max_context_tokens=400)

        output = stream.getvalue()
        self.assertIn("[tokens]", output)
        self.assertIn("240", output)
        self.assertIn("400", output)


if __name__ == "__main__":
    unittest.main()
