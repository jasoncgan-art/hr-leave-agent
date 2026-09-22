"""Guardrail tests: PII redaction, abstention, out-of-scope fallback."""

import os

from src.agent.router import Agent
from src.guardrails.pii import redact

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _agent(tmp_path):
    return Agent(data_dir=DATA, tickets_dir=str(tmp_path))


def test_redact_ssn():
    out, found = redact("My SSN is 123-45-6789, thanks.")
    assert found is True
    assert "123-45-6789" not in out
    assert "[REDACTED]" in out


def test_redact_email_and_phone():
    out, found = redact("Reach me at jane.doe@example.com or (555) 123-4567.")
    assert found is True
    assert "jane.doe@example.com" not in out
    assert "555" not in out


def test_redact_clean_text():
    out, found = redact("How much PTO do I accrue per year?")
    assert found is False
    assert out == "How much PTO do I accrue per year?"


def test_abstention_on_unknown_topic(tmp_path):
    resp = _agent(tmp_path).handle("What is the sabbatical policy?")
    assert resp["abstained"] is True
    assert resp["tools_used"] == []


def test_out_of_scope_fallback_has_no_tools(tmp_path):
    resp = _agent(tmp_path).handle("What is the 401k match?")
    assert resp["intent"] == "out_of_scope"
    assert resp["abstained"] is False
    assert resp["tools_used"] == []
    assert "leave-triage" in resp["answer"]


def test_pii_in_input_is_flagged_and_redacted_in_output(tmp_path):
    resp = _agent(tmp_path).handle(
        "My SSN is 123-45-6789. How many PTO days does Ava Chen have?"
    )
    assert resp["redacted_input"] is True
    assert "123-45-6789" not in resp["answer"]
    assert "[REDACTED]" in resp["answer"]
    assert resp["facts"]["value"] == 12.5
