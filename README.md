# HR Leave-Triage Agent (synthetic demo)

An agentic-style AI system that answers employee **leave** questions for a
fictional company ("Acme Corp") using three tools — policy lookup, a
leave-balance calculator, and leave-ticket creation — wrapped in guardrails
(PII redaction, weak-evidence abstention, out-of-scope fallback) and measured
by an evaluation harness.

> **Honest statement:** there is no LLM in this repo. The "agent" is a
> **deterministic simulation**: a rule/intent-based router over synthetic
> policy documents, with a pluggable `LLMRouter` protocol so a real model can
> replace the rules later. The bundled `OpenAIAdapter` is an explicit stub —
> it raises instead of faking an integration, and no model outputs are ever
> invented. All people, balances, and policies are synthetic.

## Architecture

```
                    ┌─────────────┐
  user query ──────▶│  FastAPI    │  POST /chat  →  ChatResponse (JSON)
                    │  service/   │  GET  /health
                    └──────┬──────┘
                           ▼
              ┌────────────────────────┐
              │  Agent.handle()        │  src/agent/router.py
              │  1. PII redact input   │
              │  2. route → intent     │◀── DeterministicRouter
              │  3. dispatch tools     │    (implements LLMRouter protocol)
              │  4. guardrail checks   │
              │  5. PII redact output  │
              └────┬───────┬───────┬───┘
                   ▼       ▼       ▼
            ┌────────┐ ┌────────┐ ┌──────────────┐
            │ policy │ │ leave  │ │    create    │
            │ lookup │ │balance │ │    ticket    │
            │ TF-IDF │ │ roster │ │ tickets/*.json
            │ 26 §   │ │  CSV   │ └──────────────┘
            └────────┘ └────────┘
                   │
        ┌──────────┴───────────┐
        ▼                      ▼
  src/guardrails/        src/llm/
  pii.py (regex          base.py: LLMRouter protocol
  redact SSN/email/      openai_adapter.py: stub —
  phone), abstention     raises unless OPENAI_API_KEY
  on weak evidence       is set (and even then is
  (score < 0.13 or        unimplemented by design)
  <2 content tokens
  matched), out-of-
  scope safe fallback
```

Data flow for a policy question: query → redacted → intent=`policy` →
`policy_lookup` ranks the 26 `[Section: X]` chunks with sublinear-TF-IDF
cosine → if top score < 0.13 **or** fewer than 2 distinctive query terms match
the top section, the agent **abstains** rather than guessing → otherwise the
answer quotes the top sections with `[Section]` citations.

Routing rule of thumb: `balance` and `request` are *operational* intents that
act on one employee's record, so they require an identified employee
(`EMP-####` or a roster name). Without one, "how many days…" questions are
treated as policy questions about accrual/entitlement.

## Measured results (real, from `eval/results.json`)

`python eval/run_eval.py` → 18-task golden set (4 balance, 5 policy with
expected cited sections, 3 abstention traps, 2 PII traps, 2 out-of-scope,
2 ticket requests). Generated 2026-09-22T22:41:15Z.

| Metric | Value |
|---|---|
| Task success rate | **1.000** (18/18) |
| Citation accuracy | **1.000** (6/6 policy/PII tasks cite every expected section) |
| Abstention precision | **1.000** (3 abstentions, all correct) |
| Abstention recall | **1.000** (all 3 traps abstained on) |
| Mean latency | **0.298 ms** |
| p95 latency | **1.303 ms** |
| pytest | **22 passed** |

Latency is sub-millisecond because everything is local and deterministic —
there is no model inference. The abstention traps are topics genuinely absent
from the policy docs (sabbatical, pet bereavement, jury duty); the two PII
tasks confirm SSNs/emails are redacted to `[REDACTED]` before processing and
never leak into answers.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned; CPU-only

pytest -q                         # 22 tests
python eval/run_eval.py           # 18-task eval → eval/results.json
uvicorn service.app:app --port 8000
curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "How many PTO days does Ava Chen have left?"}'
# → {"intent":"balance","answer":"Ava Chen (EMP-1001) has 12.5 days of PTO remaining.",...}
```

Or `make test`, `make eval`, `make serve`. Docker: `docker build -t
hr-leave-agent . && docker run -p 8000:8000 hr-leave-agent`.

## Repo layout

- `data/policies/` — 8 synthetic Acme Corp leave-policy docs (26 cited sections)
- `data/roster.csv` — 5 fake employees with leave balances
- `src/agent/` — `DeterministicRouter`, tool dispatch, `Agent.handle()`
- `src/agent/tools.py` — TF-IDF `policy_lookup`, `leave_balance`, `create_ticket`
- `src/guardrails/` — PII redaction, abstention/out-of-scope messages
- `src/llm/` — `LLMRouter` protocol + honest `OpenAIAdapter` stub
- `eval/` — `golden_tasks.json` (18 tasks), `run_eval.py`, `results.json`
- `service/app.py` — FastAPI `/chat` + `/health`
- `tests/` — 22 pytest tests
- `tickets/` — runtime output of `create_ticket` (git-ignored)

## Limitations

- **No real NLU.** Intent classification is regex + roster-name matching; it
  will misroute unusual phrasings, sarcasm, multi-intent queries ("check my
  balance and file a ticket"), and non-English input.
- **Keyword retrieval, not semantic search.** TF-IDF can't handle synonyms
  ("time off" vs "PTO") or paraphrase; section citations are only as good as
  lexical overlap.
- **Regex PII redaction is a demo control**, not a compliance guarantee — it
  misses non-US formats and can't catch PII expressed in prose.
- **Ticket store is flat JSON files**, not a real system of record; no auth,
  no concurrency control.
- **Policies are fictional** and deliberately incomplete (no sabbatical, jury
  duty, etc.) — the abstention behavior is a feature of the demo, not legal
  coverage.
- To use a real LLM: implement `LLMRouter.classify()` against an actual API
  (set `OPENAI_API_KEY`), then re-run the eval harness — expect the numbers
  above to change, and the latency row to stop being sub-millisecond.

## Environment

Python 3.12, CPU-only, no GPU, no network calls at runtime. The only
documented secret name is the `OPENAI_API_KEY` environment variable referenced
by the stub adapter; no real credentials exist anywhere in this repo.
