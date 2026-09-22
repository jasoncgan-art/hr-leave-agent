"""OpenAI-backed router adapter — STUB, not a real integration.

This class exists to show where a real LLM would plug in (it implements the
LLMRouter protocol). It deliberately performs no network calls, invents no
model outputs, and raises clear errors instead of faking an integration.

To implement for real: set the OPENAI_API_KEY environment variable and fill in
classify() with an actual chat-completions call that maps the model output to
one of the intent labels ('balance', 'policy', 'request', 'out_of_scope').
"""

from __future__ import annotations

import os


class OpenAIAdapter:
    """Stub adapter. Honest by construction: raises instead of faking."""

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. This adapter is an unimplemented "
                "stub: set the environment variable and implement classify() "
                "before use."
            )
        # Even with a key present, there is no real implementation here.
        raise NotImplementedError(
            "OpenAIAdapter.classify() is not implemented in this demo. "
            "Wire up a real API call here; no model outputs are fabricated."
        )

    def classify(self, query: str) -> str:  # pragma: no cover - never reached
        raise NotImplementedError("See __init__: stub adapter is not usable.")
