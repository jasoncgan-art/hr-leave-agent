"""Deterministic intent router + the agent's handle() orchestration.

This module SIMULATES an agentic AI system with pure rules: intent patterns,
tool dispatch, and guardrail checks. It is not an LLM and makes no claim to be
one. A real LLM can be plugged in via the LLMRouter protocol in src/llm/.
"""

from __future__ import annotations

import os
import re

from ..guardrails import (
    ABSTAIN_MESSAGE,
    CLARIFY_EMPLOYEE,
    CLARIFY_LEAVE_TYPE,
    OUT_OF_SCOPE_MESSAGE,
    UNKNOWN_EMPLOYEE_MESSAGE,
)
from ..guardrails.pii import redact
from .tools import (
    EMP_ID_RE,
    LEAVE_TYPE_RE,
    LEAVE_TYPES,
    LeaveRoster,
    PolicyRetriever,
    TicketStore,
    load_policy_sections,
)

# --- intent patterns ---------------------------------------------------------

INTENT_BALANCE = "balance"
INTENT_POLICY = "policy"
INTENT_REQUEST = "request"
INTENT_OUT_OF_SCOPE = "out_of_scope"
INTENT_UNCLEAR = "unclear"

BALANCE_RES = [
    r"\bbalance\b",
    r"\bremaining\b",
    r"\bhow many (days|weeks|hours)\b",
    r"\bdays? (do|does) .* have\b",
    r"\bleft\b",
]
REQUEST_RES = [
    r"\brequest\b",
    r"\bfile\b",
    r"\bsubmit\b",
    r"\bcreate\b.*\bticket\b",
    r"\btake\b.*\boff\b",
    r"\bapply for\b",
    r"\bbook\b.*\b(leave|pto|vacation)\b",
]
POLICY_RES = [
    r"\bpolicy\b",
    r"\beligible\b",
    r"\beligibility\b",
    r"\baccrue\b",
    r"\baccrual\b",
    r"\bcarry ?over\b",
    r"\bcarryover\b",
    r"\bhow does\b",
    r"\bhow much\b",
    r"\bhow many\b",
    r"\bwhat (is|are)\b",
    r"\bcan i\b",
    r"\bam i\b",
    r"\bdo(es)? .* (offer|have)\b",
]

# Topics the demo explicitly does NOT cover -> safe fallback, never tools.
OUT_OF_SCOPE_RES = [
    r"\b401\s?k\b",
    r"\bretirement\b",
    r"\bhealth insurance\b",
    r"\bdental\b",
    r"\bvision\b",
    r"\bsalary\b",
    r"\bbonus\b",
    r"\bpaycheck\b",
    r"\bpayroll\b",
    r"\bparking\b",
    r"\bdress code\b",
    r"\bpromotion\b",
    r"\bperformance review\b",
]

SCORE_THRESHOLD = 0.13
MIN_MATCHED_CONTENT_TOKENS = 2


class DeterministicRouter:
    """Rule/intent-based router implementing the LLMRouter protocol.

    Design rule: balance and request are *operational* intents — they act on a
    specific employee's record, so they require an identified employee (an
    EMP-#### id or a known roster name) in the query. Without one, "how many
    days ..." style questions are treated as policy questions (accrual rates,
    entitlements), which the retrieval layer then answers or abstains on.
    """

    def __init__(self, employee_names: list[str] | None = None):
        self.employee_names = [n.lower() for n in (employee_names or [])]

    def _has_employee(self, q: str) -> bool:
        if EMP_ID_RE.search(q):
            return True
        return any(name in q for name in self.employee_names)

    @staticmethod
    def _matches(q: str, patterns: list[str]) -> bool:
        return any(re.search(p, q) for p in patterns)

    def classify(self, query: str) -> str:
        q = query.lower()
        if self._matches(q, OUT_OF_SCOPE_RES):
            return INTENT_OUT_OF_SCOPE
        has_employee = self._has_employee(q)
        is_request = self._matches(q, REQUEST_RES)
        is_balance = self._matches(q, BALANCE_RES)
        is_policy = self._matches(q, POLICY_RES)
        if has_employee:
            if is_request:
                return INTENT_REQUEST
            if is_balance:
                return INTENT_BALANCE
            if is_policy:
                return INTENT_POLICY
            return INTENT_UNCLEAR
        # No identified employee: only explicit ticket-filing language counts
        # as a request; otherwise policy wording wins over balance wording.
        if (
            is_request
            and _leave_type(query) is not None
            and self._matches(q, [r"\bticket\b", r"\bfile\b", r"\bsubmit\b",
                                  r"\bbook\b"])
        ):
            return INTENT_REQUEST
        if is_policy:
            return INTENT_POLICY
        return INTENT_UNCLEAR


def _leave_type(query: str) -> str | None:
    m = LEAVE_TYPE_RE.search(query)
    if not m:
        return None
    key = m.group(1).lower()
    return {"vacation": "pto"}.get(key, key)


class Agent:
    """Orchestrates router -> tools -> guardrails for one query."""

    def __init__(self, data_dir: str | None = None, tickets_dir: str | None = None):
        base = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "data"
        )
        base = os.path.abspath(base)
        policies_dir = os.path.join(base, "policies")
        roster_csv = os.path.join(base, "roster.csv")
        self.retriever = PolicyRetriever(load_policy_sections(policies_dir))
        self.roster = LeaveRoster(roster_csv)
        self.tickets = TicketStore(
            tickets_dir or os.path.join(os.path.dirname(base), "tickets")
        )
        self.router = DeterministicRouter(
            employee_names=[e["name"] for e in self.roster.employees]
        )

    # -- response builder -----------------------------------------------------
    @staticmethod
    def _response(**kw) -> dict:
        base = {
            "intent": None,
            "answer": "",
            "citations": [],
            "tools_used": [],
            "facts": {},
            "abstained": False,
            "redacted_input": False,
            "redacted_output": False,
        }
        base.update(kw)
        return base

    def handle(self, query: str) -> dict:
        clean_query, pii_in = redact(query)
        intent = self.router.classify(clean_query)

        if intent == INTENT_OUT_OF_SCOPE:
            return self._final(
                self._response(
                    intent=intent,
                    answer=OUT_OF_SCOPE_MESSAGE,
                ),
                pii_in,
            )

        employee = self.roster.find(clean_query)

        if intent == INTENT_BALANCE:
            return self._final(self._handle_balance(clean_query, employee), pii_in)
        if intent == INTENT_REQUEST:
            return self._final(self._handle_request(clean_query, employee), pii_in)
        if intent == INTENT_POLICY:
            return self._final(self._handle_policy(clean_query), pii_in)
        # INTENT_UNCLEAR -> safe abstention
        return self._final(
            self._response(intent=intent, answer=ABSTAIN_MESSAGE, abstained=True),
            pii_in,
        )

    # -- intent handlers ------------------------------------------------------
    def _handle_balance(self, query: str, employee: dict | None) -> dict:
        leave_type = _leave_type(query)
        if employee is None:
            return self._response(
                intent=INTENT_BALANCE, answer=UNKNOWN_EMPLOYEE_MESSAGE
            )
        if leave_type is None:
            return self._response(
                intent=INTENT_BALANCE, answer=CLARIFY_LEAVE_TYPE
            )
        bal = self.roster.balance(employee, leave_type)
        answer = (
            f"{bal['name']} ({bal['employee_id']}) has "
            f"{bal['value']:g} {bal['unit']} of {bal['label']} remaining."
        )
        return self._response(
            intent=INTENT_BALANCE,
            answer=answer,
            tools_used=["leave_balance"],
            facts={
                "employee_id": bal["employee_id"],
                "name": bal["name"],
                "leave_type": bal["leave_type"],
                "value": bal["value"],
                "unit": bal["unit"],
            },
        )

    def _handle_policy(self, query: str) -> dict:
        hits = self.retriever.search(query, top_k=3)
        top_sec, top_score = hits[0]
        matched = self.retriever.matched_content_tokens(query, top_sec)
        if top_score < SCORE_THRESHOLD or len(matched) < MIN_MATCHED_CONTENT_TOKENS:
            return self._response(
                intent=INTENT_POLICY, answer=ABSTAIN_MESSAGE, abstained=True,
                facts={"top_score": round(top_score, 4),
                       "matched_tokens": sorted(matched)},
            )
        citations = [
            {"doc": sec.doc, "section": sec.section_id, "score": round(score, 4)}
            for sec, score in hits
            if score >= SCORE_THRESHOLD
        ]
        cited = "; ".join(
            f"[{c['section']}] ({c['doc']})" for c in citations
        )
        snippet = top_sec.text[:420].rsplit(" ", 1)[0]
        answer = (
            f"According to Acme Corp's leave policies ({cited}): {snippet}..."
        )
        return self._response(
            intent=INTENT_POLICY,
            answer=answer,
            citations=citations,
            tools_used=["policy_lookup"],
            facts={"top_score": round(top_score, 4)},
        )

    def _handle_request(self, query: str, employee: dict | None) -> dict:
        leave_type = _leave_type(query)
        if employee is None:
            return self._response(
                intent=INTENT_REQUEST, answer=CLARIFY_EMPLOYEE
            )
        if leave_type is None:
            return self._response(
                intent=INTENT_REQUEST, answer=CLARIFY_LEAVE_TYPE
            )
        ticket = self.tickets.create(employee, leave_type, query)
        answer = (
            f"Leave request ticket {ticket['ticket_id']} created for "
            f"{employee['name']} ({employee['employee_id']}): {leave_type} leave. "
            f"Details saved. Per policy [REQ-1], planned PTO requests need "
            f"14 days' advance notice and receive a decision within 2 "
            f"business days."
        )
        return self._response(
            intent=INTENT_REQUEST,
            answer=answer,
            citations=[{"doc": "08_request_process.md", "section": "REQ-1",
                        "score": 1.0}],
            tools_used=["create_ticket"],
            facts={"ticket_id": ticket["ticket_id"],
                   "employee_id": employee["employee_id"],
                   "leave_type": leave_type},
        )

    # -- output guardrails ----------------------------------------------------
    @staticmethod
    def _final(resp: dict, pii_in: bool) -> dict:
        answer, pii_out = redact(resp["answer"])
        if pii_in and "[REDACTED]" not in answer:
            answer = (
                "Note: I detected and redacted personal identifiers "
                "([REDACTED]) in your message before processing. " + answer
            )
        resp["answer"] = answer
        resp["redacted_input"] = pii_in
        resp["redacted_output"] = pii_out
        return resp
