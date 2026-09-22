"""FastAPI service exposing the leave-triage agent."""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.agent.router import Agent  # noqa: E402

app = FastAPI(title="hr-leave-agent", version="0.1.0")
agent = Agent(data_dir=os.path.join(REPO_ROOT, "data"),
              tickets_dir=os.path.join(REPO_ROOT, "tickets"))


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    doc: str
    section: str
    score: float


class ChatResponse(BaseModel):
    intent: str | None
    answer: str
    citations: list[Citation]
    tools_used: list[str]
    abstained: bool
    redacted_input: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "router": "deterministic", "llm": "none (simulated)"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    resp = agent.handle(req.query)
    return {
        "intent": resp["intent"],
        "answer": resp["answer"],
        "citations": resp["citations"],
        "tools_used": resp["tools_used"],
        "abstained": resp["abstained"],
        "redacted_input": resp["redacted_input"],
    }
