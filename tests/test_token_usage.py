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
from ollamanauts.token_usage import should_compact
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

    def test_should_compact_threshold_behavior(self) -> None:
        self.assertFalse(
            should_compact(estimated_tokens=200, max_context_tokens=400, compact_threshold=0.75)
        )
        self.assertTrue(
            should_compact(estimated_tokens=300, max_context_tokens=400, compact_threshold=0.75)
        )

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

    def test_compaction_needed_callback_triggers_at_threshold(self) -> None:
        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        compaction_called: dict[str, float] = {}

        def on_compaction_needed(
            estimated_tokens: int,
            max_context_tokens: int,
            compact_threshold: float,
            compact_target: float,
        ) -> None:
            compaction_called["estimated_tokens"] = float(estimated_tokens)
            compaction_called["max_context_tokens"] = float(max_context_tokens)
            compaction_called["compact_threshold"] = compact_threshold
            compaction_called["compact_target"] = compact_target

        with unittest.mock.patch("ollamanauts.agent.ollama.chat", return_value=[_Chunk()]):
            agent = BaseAgent(
                model="dummy",
                orchestrator=ToolOrchestrator([]),
                system_prompt="system",
                max_context_tokens=10,
                compact_threshold=0.1,
                compact_target=0.5,
                on_compaction_needed=on_compaction_needed,
            )
            agent.run_turn("hello")

        self.assertEqual(compaction_called["max_context_tokens"], 10.0)
        self.assertEqual(compaction_called["compact_threshold"], 0.1)
        self.assertEqual(compaction_called["compact_target"], 0.5)

    def test_base_agent_defaults_compaction_model_to_model(self) -> None:
        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        agent = BaseAgent(
            model="gemma4:31b",
            orchestrator=ToolOrchestrator([]),
            system_prompt="system",
        )

        self.assertEqual(agent.compaction_model, "gemma4:31b")

    def test_base_agent_keeps_explicit_compaction_model(self) -> None:
        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        agent = BaseAgent(
            model="gemma4:31b",
            orchestrator=ToolOrchestrator([]),
            system_prompt="system",
            compaction_model="gemma4:27b",
        )

        self.assertEqual(agent.compaction_model, "gemma4:27b")

    def test_compaction_replaces_older_context_with_summary(self) -> None:
        from types import SimpleNamespace

        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        def fake_chat(**kwargs: object) -> object:
            if kwargs.get("stream") is False:
                return SimpleNamespace(message=SimpleNamespace(content="summarized context"))
            return [_Chunk()]

        with unittest.mock.patch("ollamanauts.agent.ollama.chat", side_effect=fake_chat):
            agent = BaseAgent(
                model="dummy",
                orchestrator=ToolOrchestrator([]),
                system_prompt="system",
                max_context_tokens=10,
                compact_threshold=0.1,
                compact_target=0.1,
                compaction_preserve_last_n_turns=1,
            )
            agent.messages.extend([
                {"role": "user", "content": "old context A"},
                {"role": "assistant", "content": "old context B"},
            ])
            agent.run_turn("hello world this is a long prompt")

        summary_messages = [m for m in agent.messages if "Conversation summary" in m.get("content", "")]
        self.assertTrue(summary_messages)

    def test_expand_tail_keeps_tool_call_result_boundary_intact(self) -> None:
        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        agent = BaseAgent(
            model="dummy",
            orchestrator=ToolOrchestrator([]),
            system_prompt="system",
        )
        non_system = [
            {"role": "user", "content": "old context"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_name": "lookup", "content": "result"},
            {"role": "assistant", "content": "follow-up"},
        ]
        kept_tail = non_system[-2:]

        expanded = agent._expand_tail_for_tool_integrity(non_system=non_system, kept_tail=kept_tail)

        self.assertEqual(expanded[0].get("role"), "assistant")
        self.assertTrue(expanded[0].get("tool_calls"))
        self.assertEqual(expanded[1].get("role"), "tool")

    def test_collect_unresolved_tool_indices(self) -> None:
        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        agent = BaseAgent(
            model="dummy",
            orchestrator=ToolOrchestrator([]),
            system_prompt="system",
        )
        non_system = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_name": "lookup", "content": "result"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
            {"role": "assistant", "content": "normal reply"},
        ]

        protected = agent._collect_unresolved_tool_indices(non_system)

        self.assertEqual(protected, {1, 2, 3})

    def test_collect_unresolved_tool_indices_keeps_assistant_follow_up(self) -> None:
        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        agent = BaseAgent(
            model="dummy",
            orchestrator=ToolOrchestrator([]),
            system_prompt="system",
        )
        non_system = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_name": "lookup", "content": "result"},
            {"role": "assistant", "content": "tool interpretation"},
            {"role": "user", "content": "next turn"},
        ]

        protected = agent._collect_unresolved_tool_indices(non_system)

        self.assertEqual(protected, {0, 1, 2})

    def test_collect_unresolved_tool_indices_handles_multi_episode_chain(self) -> None:
        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        agent = BaseAgent(
            model="dummy",
            orchestrator=ToolOrchestrator([]),
            system_prompt="system",
        )
        non_system = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_name": "lookup", "content": "result-a"},
            {"role": "assistant", "content": "analysis-a"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
            {"role": "tool", "tool_name": "search", "content": "result-b"},
            {"role": "assistant", "content": "analysis-b"},
        ]

        protected = agent._collect_unresolved_tool_indices(non_system)

        self.assertEqual(protected, {0, 1, 2, 3, 4, 5})

    def test_summary_prompt_uses_structured_memory_sections(self) -> None:
        from types import SimpleNamespace

        from ollamanauts.agent import BaseAgent
        from ollamanauts.tool_orchestrator import ToolOrchestrator

        captured_prompt: dict[str, str] = {}

        def fake_chat(**kwargs: object) -> object:
            messages = kwargs.get("messages")
            if isinstance(messages, list) and messages:
                captured_prompt["prompt"] = messages[0]["content"]
            return SimpleNamespace(message=SimpleNamespace(content="structured summary"))

        with unittest.mock.patch("ollamanauts.agent.ollama.chat", side_effect=fake_chat):
            agent = BaseAgent(
                model="dummy",
                orchestrator=ToolOrchestrator([]),
                system_prompt="system",
            )
            agent._summarize_messages([{"role": "user", "content": "hello"}])

        prompt = captured_prompt["prompt"]
        self.assertIn("## Facts", prompt)
        self.assertIn("## Decisions", prompt)
        self.assertIn("## Constraints", prompt)
        self.assertIn("## Open Questions", prompt)
        self.assertIn("## Pending Actions", prompt)

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
