"""LLM router adapter interface (pluggable, no live LLM calls)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMRouter(Protocol):
    """Pluggable router interface.

    Implementations classify a user query into an intent string. The shipped
    DeterministicRouter implements this protocol with hand-written rules;
    a real LLM adapter would implement it by calling a model.
    """

    def classify(self, query: str) -> str:
        """Return an intent label such as 'balance', 'policy', or 'request'."""
        ...
