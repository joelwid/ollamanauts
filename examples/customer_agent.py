from ollamanauts import Agent

from examples.customer_tools import explain_plan
from examples.customer_tools import lookup_customer


if __name__ == "__main__":
    agent = Agent(
        model="gemma4:31b",
        extra_instructions=(
            "You are a support assistant. Use the provided tools for customer"
            " and plan questions. If data is missing, say so plainly."
        ),
        tools=[lookup_customer, explain_plan],
        verbose=True,
        enable_subagents=False,
    )
    response = agent.run(
        "Customer CUST-100 wants to know whether their plan includes priority support."
    )
    print(response)
