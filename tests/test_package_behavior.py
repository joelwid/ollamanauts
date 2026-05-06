from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch


class PackageBehaviorTests(unittest.TestCase):
    def test_package_imports_expose_public_api(self) -> None:
        package = importlib.import_module("ollamanauts")

        expected_exports = {
            "Agent",
            "BaseAgent",
            "InteractiveAgent",
            "SubAgent",
            "ToolOrchestrator",
            "ToolResult",
        }

        self.assertEqual(set(package.__all__), expected_exports)
        for name in expected_exports:
            self.assertTrue(hasattr(package, name))

    def test_prompt_files_load_from_package_resources(self) -> None:
        agent_module = importlib.import_module("ollamanauts.agent")

        interactive_prompt = agent_module.load_prompt("interactive_agent.md")
        subagent_prompt = agent_module.load_prompt("subagent.md")

        self.assertEqual(interactive_prompt, agent_module.INTERACTIVE_AGENT_PROMPT)
        self.assertEqual(subagent_prompt, agent_module.SUBAGENT_PROMPT)
        self.assertTrue(interactive_prompt.strip())
        self.assertTrue(subagent_prompt.strip())

    def test_system_prompt_replaces_default_prompt(self) -> None:
        from ollamanauts import Agent

        custom_prompt = "You are a JSON-only assistant."
        agent = Agent(system_prompt=custom_prompt, enable_subagents=False)

        self.assertEqual(agent._agent.system_prompt, custom_prompt)

    def test_extra_instructions_append_to_default_prompt(self) -> None:
        from ollamanauts import Agent
        from ollamanauts.agent import INTERACTIVE_AGENT_PROMPT

        extra_instructions = "Always answer in JSON."
        agent = Agent(
            extra_instructions=extra_instructions,
            enable_subagents=False,
        )

        self.assertEqual(
            agent._agent.system_prompt,
            f"{INTERACTIVE_AGENT_PROMPT}\n\nAdditional user instructions:\n{extra_instructions}",
        )

    def test_no_dangerous_default_tools_registered(self) -> None:
        from ollamanauts import Agent

        agent = Agent(enable_subagents=False)

        self.assertEqual(agent.describe_tools(), [])

    def test_user_supplied_tools_are_registered(self) -> None:
        from ollamanauts import Agent

        def lookup_customer(customer_id: str) -> dict[str, str]:
            return {"id": customer_id, "status": "active"}

        agent = Agent(
            tools=[lookup_customer],
            enable_subagents=False,
        )

        self.assertEqual(agent.describe_tools(), ["lookup_customer"])

    def test_subagent_tool_registered_only_on_main_agent(self) -> None:
        from ollamanauts import Agent
        from ollamanauts.tools.subagents import make_deploy_subagent_tool

        agent = Agent(enable_subagents=True)
        self.assertEqual(agent.describe_tools(), ["deploy_subagent"])

        with patch("ollamanauts.tools.subagents.SubAgent") as mock_subagent:
            mock_subagent.return_value.run.return_value = "done"
            nested_deploy_tool = make_deploy_subagent_tool(
                model="gemma4:31b",
                think_mode="medium",
            )
            deploy_subagent = make_deploy_subagent_tool(
                model="gemma4:31b",
                think_mode="medium",
                tools=[nested_deploy_tool],
            )

            result = deploy_subagent("Summarize the open questions.")

        self.assertEqual(result, "done")
        _, kwargs = mock_subagent.call_args
        self.assertEqual(kwargs["orchestrator"].tool_names(), [])


if __name__ == "__main__":
    unittest.main()
