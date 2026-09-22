"""Tool tests: retrieval, roster/balance, ticket store."""

import json
import os

from src.agent.tools import (LeaveRoster, PolicyRetriever, TicketStore,
                             load_policy_sections)

POLICIES = os.path.join(os.path.dirname(__file__), "..", "data", "policies")
ROSTER = os.path.join(os.path.dirname(__file__), "..", "data", "roster.csv")


def test_sections_loaded():
    secs = load_policy_sections(POLICIES)
    assert len(secs) == 26  # 4+4+3+4+3+2+2+4 across the 8 docs
    ids = {s.section_id for s in secs}
    assert {"PTO-2", "PAR-2", "CO-2", "REQ-1"} <= ids


def test_policy_lookup_cites_accrual_section():
    retr = PolicyRetriever(load_policy_sections(POLICIES))
    hits = retr.search("How much PTO do I accrue per year with 3 years of service?")
    top_sec, top_score = hits[0]
    assert top_sec.section_id == "PTO-2"
    assert top_score > 0.15


def test_policy_lookup_weak_for_unknown_topic():
    retr = PolicyRetriever(load_policy_sections(POLICIES))
    hits = retr.search("What is the sabbatical policy?")
    top_sec, top_score = hits[0]
    matched = retr.matched_content_tokens("What is the sabbatical policy?", top_sec)
    assert top_score < 0.15 or len(matched) < 2


def test_balance_values():
    roster = LeaveRoster(ROSTER)
    ava = roster.find("Ava Chen")
    assert roster.balance(ava, "pto")["value"] == 12.5
    diego = roster.find("EMP-1004")
    assert diego["name"] == "Diego Alvarez"
    assert roster.balance(diego, "pto")["value"] == 0.0
    assert roster.balance(diego, "parental")["unit"] == "weeks"
    assert roster.find("Nobody Real") is None


def test_ticket_creation_writes_file(tmp_path):
    store = TicketStore(str(tmp_path))
    roster = LeaveRoster(ROSTER)
    marcus = roster.find("Marcus Bell")
    t1 = store.create(marcus, "pto", "5 days starting Monday")
    t2 = store.create(marcus, "sick", "tomorrow")
    assert t1["ticket_id"] == "TCK-0001"
    assert t2["ticket_id"] == "TCK-0002"
    with open(tmp_path / "ticket-0001.json", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["employee_id"] == "EMP-1002"
    assert saved["status"] == "open"
