"""Router intent-classification tests."""

from src.agent.router import DeterministicRouter

NAMES = ["Ava Chen", "Marcus Bell", "Diego Alvarez", "Sofia Rossi", "Priya Nair"]


def _router():
    return DeterministicRouter(employee_names=NAMES)


def test_balance_intent():
    r = _router()
    assert r.classify("How many PTO days does Ava Chen have left?") == "balance"
    assert r.classify("What is Marcus Bell's sick leave balance?") == "balance"


def test_policy_intent():
    r = _router()
    assert r.classify("How much PTO do I accrue per year?") == "policy"
    assert r.classify("Am I eligible for parental leave?") == "policy"
    # no named employee -> policy wording wins over "how many days"
    assert r.classify(
        "How many days of bereavement leave does Acme give for immediate family?"
    ) == "policy"


def test_request_intent():
    r = _router()
    assert r.classify(
        "I want to request 5 days of PTO next week. I'm Marcus Bell."
    ) == "request"
    assert r.classify("File a sick leave ticket for Diego Alvarez.") == "request"


def test_out_of_scope_intent():
    r = _router()
    assert r.classify("What is the 401k match?") == "out_of_scope"
    assert r.classify("How do I enroll in health insurance?") == "out_of_scope"
    # out-of-scope wins even when balance-ish words appear
    assert r.classify("What is my 401k balance?") == "out_of_scope"


def test_unclear_intent():
    r = _router()
    assert r.classify("hello there") == "unclear"
