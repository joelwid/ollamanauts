CUSTOMERS = {
    "CUST-100": {"name": "Ada Lovelace", "status": "active", "plan": "pro"},
    "CUST-200": {"name": "Grace Hopper", "status": "trial", "plan": "starter"},
}


def lookup_customer(customer_id: str) -> dict[str, str]:
    """Return one customer record by ID."""
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        return {"id": customer_id, "found": "false"}
    return {"id": customer_id, "found": "true", **customer}


def explain_plan(plan: str) -> dict[str, str]:
    """Explain the features of a named plan."""
    descriptions = {
        "starter": "Starter includes basic support and a small usage quota.",
        "pro": "Pro includes priority support and a higher usage quota.",
    }
    return {"plan": plan, "description": descriptions.get(plan, "Unknown plan")}
