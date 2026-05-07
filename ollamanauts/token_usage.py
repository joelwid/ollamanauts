from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenUsageEstimate:
    message_count: int
    estimated_tokens: int


@dataclass(frozen=True)
class TokenBudgetReport:
    estimated_tokens: int
    max_context_tokens: int

    @property
    def usage_ratio(self) -> float:
        return self.estimated_tokens / self.max_context_tokens


def estimate_text_tokens(text: str) -> int:
    """Estimate token count for plain text using a conservative heuristic."""
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate token count for one chat message payload."""
    role_cost = 4
    payload_cost = 0

    for key in ("content", "thinking", "tool_name"):
        value = message.get(key)
        if isinstance(value, str):
            payload_cost += estimate_text_tokens(value)

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            payload_cost += estimate_text_tokens(str(tool_call))

    return role_cost + payload_cost


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> TokenUsageEstimate:
    estimated_tokens = sum(estimate_message_tokens(message) for message in messages)
    return TokenUsageEstimate(message_count=len(messages), estimated_tokens=estimated_tokens)


def build_token_budget_report(
    *,
    messages: list[dict[str, Any]],
    max_context_tokens: int,
) -> TokenBudgetReport:
    usage = estimate_messages_tokens(messages)
    return TokenBudgetReport(
        estimated_tokens=usage.estimated_tokens,
        max_context_tokens=max_context_tokens,
    )


def should_compact(*, estimated_tokens: int, max_context_tokens: int | None, compact_threshold: float) -> bool:
    if max_context_tokens is None or max_context_tokens <= 0:
        return False
    return (estimated_tokens / max_context_tokens) >= compact_threshold
