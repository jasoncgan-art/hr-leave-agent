"""PII redaction guardrail (regex-based, stdlib only)."""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_PATTERNS = [
    # US Social Security Number: 123-45-6789
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Email address
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Phone numbers: (555) 123-4567, 555-123-4567, 555.123.4567, +1 555 123 4567
    re.compile(
        r"(?:\+?1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}\b"
    ),
]


def redact(text: str) -> tuple[str, bool]:
    """Replace SSN-like, email, and phone patterns with [REDACTED].

    Returns (redacted_text, found_any).
    """
    found = False
    out = text
    for pat in _PATTERNS:
        out, n = pat.subn(REDACTED, out)
        if n:
            found = True
    return out, found
