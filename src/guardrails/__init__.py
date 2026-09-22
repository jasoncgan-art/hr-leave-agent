"""Guardrail messages: abstention, out-of-scope, and clarification fallbacks."""

OUT_OF_SCOPE_MESSAGE = (
    "I'm a leave-triage assistant and can only help with leave balances, "
    "Acme Corp leave policies, and filing leave-request tickets. I can't help "
    "with that topic — please contact HR for anything outside leave."
)

ABSTAIN_MESSAGE = (
    "I don't have enough reliable information in Acme Corp's leave policies "
    "to answer that confidently, so I'd rather not guess. Please contact HR "
    "directly and they'll help you out."
)

UNKNOWN_EMPLOYEE_MESSAGE = (
    "I couldn't find that employee in the roster. Please include a valid "
    "employee ID (e.g. EMP-1001) or the employee's full name."
)

CLARIFY_EMPLOYEE = (
    "To file a leave request I need to know who it's for. Please include "
    "the employee ID (e.g. EMP-1001) or the employee's full name."
)

CLARIFY_LEAVE_TYPE = (
    "Which type of leave did you mean — PTO, sick leave, or parental leave?"
)
