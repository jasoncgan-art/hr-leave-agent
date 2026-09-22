"""Tool implementations: policy_lookup, leave_balance, create_ticket.

All retrieval is a small pure-Python TF-IDF ranker (no ML model, no network).
Everything here is deterministic given the same data/ directory.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass

SECTION_RE = re.compile(r"\[Section:\s*([A-Za-z0-9\-]+)\]")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "do", "does", "did", "what",
    "whats", "how", "how", "many", "much", "my", "i", "me", "you", "your",
    "it", "its", "this", "that", "these", "those", "at", "by", "from",
    "as", "about", "can", "could", "would", "should", "will", "have",
    "has", "had", "if", "not", "no", "so", "but", "we", "they", "he",
    "she", "who", "whom", "whose", "which", "when", "where", "why",
    # Corpus-specific noise: the company name appears in every policy doc and
    # carries no discriminating power for ranking.
    "acme", "corp",
}

# Generic tokens that carry no discriminating power for abstention checks.
GENERIC_TOKENS = {"policy", "policies", "leave", "leaves", "acme", "corp"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in GENERIC_TOKENS}


@dataclass
class PolicySection:
    doc: str
    section_id: str
    text: str


def load_policy_sections(policies_dir: str) -> list[PolicySection]:
    """Chunk every policy doc into its [Section: X] blocks."""
    sections: list[PolicySection] = []
    for fname in sorted(os.listdir(policies_dir)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(policies_dir, fname), encoding="utf-8") as f:
            raw = f.read()
        matches = list(SECTION_RE.finditer(raw))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body = raw[start:end].strip()
            # Drop any markdown heading lines that leaked in.
            body = "\n".join(
                line for line in body.splitlines()
                if not line.strip().startswith("#")
            ).strip()
            sections.append(
                PolicySection(doc=fname, section_id=m.group(1), text=body)
            )
    return sections


class PolicyRetriever:
    """Keyword TF-IDF ranker over policy sections (stdlib only)."""

    def __init__(self, sections: list[PolicySection]):
        self.sections = sections
        self._idf: dict[str, float] = {}
        self._doc_vecs: list[dict[str, float]] = []
        self._fit()

    def _fit(self) -> None:
        n = len(self.sections)
        df: dict[str, int] = {}
        tfs: list[dict[str, float]] = []
        for sec in self.sections:
            toks = tokenize(sec.text)
            tf: dict[str, float] = {}
            for t in toks:
                tf[t] = tf.get(t, 0.0) + 1.0
            for t in tf:
                df[t] = df.get(t, 0) + 1
            tfs.append(tf)
        for t, d in df.items():
            self._idf[t] = math.log((n + 1) / (d + 1)) + 1.0
        for tf in tfs:
            # Sublinear term frequency (1 + log tf): dampens the advantage of
            # sections that repeat a common word many times, so sections
            # matching the query's *distinctive* terms rank higher.
            self._doc_vecs.append(
                {t: (1.0 + math.log(c)) * self._idf[t] for t, c in tf.items()}
            )

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        num = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return num / (na * nb)

    def search(self, query: str, top_k: int = 3) -> list[tuple[PolicySection, float]]:
        toks = tokenize(query)
        qtf: dict[str, float] = {}
        for t in toks:
            qtf[t] = qtf.get(t, 0.0) + 1.0
        total = sum(qtf.values()) or 1.0
        qvec = {
            t: (c / total) * self._idf.get(t, math.log(len(self.sections) + 1) + 1.0)
            for t, c in qtf.items()
        }
        scored = [
            (sec, self._cosine(qvec, dvec))
            for sec, dvec in zip(self.sections, self._doc_vecs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def matched_content_tokens(query: str, section: PolicySection) -> set[str]:
        q = content_tokens(query)
        sec = set(tokenize(section.text))
        return q & sec


LEAVE_TYPES = {
    "pto": ("pto_balance_days", "PTO", "days"),
    "vacation": ("pto_balance_days", "PTO", "days"),
    "sick": ("sick_balance_days", "sick leave", "days"),
    "parental": ("parental_weeks_remaining", "parental leave", "weeks"),
    "maternity": ("parental_weeks_remaining", "parental leave", "weeks"),
    "paternity": ("parental_weeks_remaining", "parental leave", "weeks"),
}

LEAVE_TYPE_RE = re.compile(
    r"\b(pto|vacation|sick|parental|maternity|paternity)\b", re.IGNORECASE
)
EMP_ID_RE = re.compile(r"\bEMP-\d{4}\b", re.IGNORECASE)


class LeaveRoster:
    """Synthetic employee roster + leave-balance calculator."""

    def __init__(self, roster_csv: str):
        self.employees: list[dict] = []
        with open(roster_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["tenure_years"] = float(row["tenure_years"])
                row["pto_balance_days"] = float(row["pto_balance_days"])
                row["sick_balance_days"] = float(row["sick_balance_days"])
                row["parental_weeks_remaining"] = int(
                    float(row["parental_weeks_remaining"])
                )
                self.employees.append(row)

    def find(self, query: str) -> dict | None:
        """Find an employee by EMP-#### id or (case-insensitive) name match."""
        m = EMP_ID_RE.search(query)
        if m:
            wanted = m.group(0).upper()
            for e in self.employees:
                if e["employee_id"].upper() == wanted:
                    return e
        q = query.lower()
        for e in self.employees:
            if e["name"].lower() in q:
                return e
        return None

    def balance(self, employee: dict, leave_type: str) -> dict:
        col, label, unit = LEAVE_TYPES[leave_type]
        return {
            "employee_id": employee["employee_id"],
            "name": employee["name"],
            "leave_type": leave_type,
            "label": label,
            "unit": unit,
            "value": employee[col],
            "tenure_years": employee["tenure_years"],
        }


class TicketStore:
    """Writes leave-request tickets as JSON files (deterministic ids)."""

    def __init__(self, tickets_dir: str):
        self.tickets_dir = tickets_dir
        os.makedirs(tickets_dir, exist_ok=True)

    def _next_id(self) -> str:
        nums = []
        for fname in os.listdir(self.tickets_dir):
            m = re.fullmatch(r"ticket-(\d+)\.json", fname)
            if m:
                nums.append(int(m.group(1)))
        return f"TCK-{max(nums, default=0) + 1:04d}"

    def create(self, employee: dict, leave_type: str, details: str) -> dict:
        ticket_id = self._next_id()
        num = int(ticket_id.split("-")[1])
        ticket = {
            "ticket_id": ticket_id,
            "employee_id": employee["employee_id"],
            "name": employee["name"],
            "leave_type": leave_type,
            "details": details.strip()[:500],
            "status": "open",
        }
        path = os.path.join(self.tickets_dir, f"ticket-{num:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ticket, f, indent=2)
        return ticket
