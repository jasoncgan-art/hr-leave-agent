"""Service + eval-harness tests."""

import json
import os

from fastapi.testclient import TestClient

from service.app import app

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "eval", "golden_tasks.json")


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["router"] == "deterministic"


def test_chat_balance():
    client = TestClient(app)
    r = client.post("/chat", json={"query": "What is Sofia Rossi's PTO balance?"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "balance"
    assert "22.5" in body["answer"]


def test_golden_tasks_schema():
    with open(GOLDEN, encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]
    assert len(tasks) >= 15
    types = {t["type"] for t in tasks}
    assert {"balance", "policy", "abstain", "pii", "out_of_scope", "request"} <= types
    for t in tasks:
        assert {"id", "query", "type", "expected"} <= set(t)
        assert "intent" in t["expected"]
