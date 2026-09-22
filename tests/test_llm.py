"""LLM adapter interface tests."""

import pytest

from src.agent.router import DeterministicRouter
from src.llm.base import LLMRouter
from src.llm.openai_adapter import OpenAIAdapter


def test_deterministic_router_implements_protocol():
    assert isinstance(DeterministicRouter(), LLMRouter)
    r = DeterministicRouter(employee_names=["Ava Chen"])
    assert r.classify("What is Ava Chen's PTO balance?") == "balance"


def test_openai_adapter_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIAdapter()


def test_openai_adapter_stub_even_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key")
    with pytest.raises(NotImplementedError, match="not implemented"):
        OpenAIAdapter()
