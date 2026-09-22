"""LLM adapter package: pluggable router interface + OpenAI stub."""

from .base import LLMRouter
from .openai_adapter import OpenAIAdapter

__all__ = ["LLMRouter", "OpenAIAdapter"]
