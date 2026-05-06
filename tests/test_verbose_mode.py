from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch


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


class VerboseModeTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_ollama_stub()

    def test_run_wires_verbose_callbacks(self) -> None:
        from ollamanauts import Agent

        with patch("ollamanauts.agent.BaseAgent") as mock_base_agent:
            base_agent_instance = mock_base_agent.return_value
            base_agent_instance.run_turn.return_value = "ok"

            agent = Agent(enable_subagents=False, verbose=True)
            result = agent.run("hello")

        self.assertEqual(result, "ok")
        self.assertTrue(agent._verbose)
        run_kwargs = base_agent_instance.run_turn.call_args.kwargs
        self.assertIsNotNone(run_kwargs["on_tool_result"])
        self.assertIsNotNone(run_kwargs["on_thinking_chunk"])
        self.assertIsNotNone(run_kwargs["on_thinking_end"])

    def test_run_fans_out_user_and_verbose_callbacks(self) -> None:
        from ollamanauts import Agent
        from ollamanauts.tool_orchestrator import ToolResult

        on_tool_result = MagicMock()
        on_thinking_chunk = MagicMock()
        on_thinking_end = MagicMock()

        with patch("ollamanauts.agent.BaseAgent") as mock_base_agent:
            base_agent_instance = mock_base_agent.return_value

            def _trigger_callbacks(*_: object, **kwargs: object) -> str:
                kwargs["on_tool_result"](ToolResult(name="lookup", arguments={}, content="x"))
                kwargs["on_thinking_chunk"]("chain")
                kwargs["on_thinking_end"]()
                return "ok"

            base_agent_instance.run_turn.side_effect = _trigger_callbacks

            agent = Agent(enable_subagents=False, verbose=True)
            result = agent.run(
                "hello",
                on_tool_result=on_tool_result,
                on_thinking_chunk=on_thinking_chunk,
                on_thinking_end=on_thinking_end,
            )

        self.assertEqual(result, "ok")
        on_tool_result.assert_called_once()
        on_thinking_chunk.assert_called_once_with("chain")
        on_thinking_end.assert_called_once()

    def test_verbose_subagent_hooks_are_attached(self) -> None:
        from ollamanauts import Agent

        fake_deploy_tool = lambda task: task
        with patch("ollamanauts.tools.make_deploy_subagent_tool", return_value=fake_deploy_tool) as mock_make:
            Agent(verbose=True, enable_subagents=True)

        kwargs = mock_make.call_args.kwargs
        self.assertIsNotNone(kwargs["on_start"])
        self.assertIsNotNone(kwargs["on_tool_result"])
        self.assertIsNotNone(kwargs["on_thinking_chunk"])
        self.assertIsNotNone(kwargs["on_thinking_end"])
        self.assertIsNotNone(kwargs["on_result"])


if __name__ == "__main__":
    unittest.main()
