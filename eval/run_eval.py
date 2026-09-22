"""Evaluation harness for the HR leave-triage agent.

Runs the golden task set through Agent.handle(), checks intent / facts /
citations / abstention / PII behavior per task, and writes aggregate metrics
(success rate, citation accuracy, abstention precision & recall, latency) to
eval/results.json.

Usage:
    python eval/run_eval.py [--tasks eval/golden_tasks.json] [--out eval/results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.agent.router import Agent  # noqa: E402


def check_task(task: dict, resp: dict, tickets_dir: str) -> list[str]:
    """Return a list of failure descriptions (empty = task passed)."""
    failures: list[str] = []
    exp = task["expected"]

    if resp["intent"] != exp["intent"]:
        failures.append(
            f"intent: expected {exp['intent']!r}, got {resp['intent']!r}"
        )

    for k, v in exp.get("facts", {}).items():
        if resp["facts"].get(k) != v:
            failures.append(
                f"fact {k}: expected {v!r}, got {resp['facts'].get(k)!r}"
            )

    if "citations" in exp:
        got = {c["section"] for c in resp["citations"]}
        missing = [c for c in exp["citations"] if c not in got]
        if missing:
            failures.append(f"citations missing expected sections: {missing}")

    if exp.get("should_abstain", False):
        if not resp["abstained"]:
            failures.append("expected abstention but agent answered")
    elif resp["abstained"]:
        failures.append("unexpected abstention")

    raw_pii = exp.get("pii_trap")
    if raw_pii:
        if raw_pii in resp["answer"]:
            failures.append("PII leaked into answer")
        if "[REDACTED]" not in resp["answer"]:
            failures.append("no [REDACTED] marker in answer")
        if not resp["redacted_input"]:
            failures.append("input was not flagged as redacted")

    if exp.get("expects_ticket"):
        if "create_ticket" not in resp["tools_used"]:
            failures.append("create_ticket tool was not used")
        ticket_id = resp["facts"].get("ticket_id")
        if not ticket_id:
            failures.append("no ticket_id in response facts")
        else:
            num = ticket_id.split("-")[1]
            path = os.path.join(tickets_dir, f"ticket-{num}.json")
            if not os.path.exists(path):
                failures.append(f"ticket file missing: {path}")

    return failures


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


def run(tasks_path: str) -> dict:
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]

    tickets_dir = tempfile.mkdtemp(prefix="eval-tickets-")
    agent = Agent(
        data_dir=os.path.join(REPO_ROOT, "data"), tickets_dir=tickets_dir
    )

    results = []
    latencies = []
    for task in tasks:
        t0 = time.perf_counter()
        resp = agent.handle(task["query"])
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        failures = check_task(task, resp, tickets_dir)
        results.append(
            {
                "id": task["id"],
                "type": task["type"],
                "passed": not failures,
                "failures": failures,
                "intent": resp["intent"],
                "abstained": resp["abstained"],
                "latency_ms": round(dt_ms, 3),
            }
        )

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    # Citation accuracy: fraction of citation-bearing tasks whose expected
    # sections were all present.
    cit_tasks = [t for t in tasks if "citations" in t["expected"]]
    cit_ok = sum(
        1
        for t, r in zip(tasks, results)
        if "citations" in t["expected"] and r["passed"]
    )
    # (passed implies citations matched, since citation mismatch is a failure)

    # Abstention precision / recall.
    actual_pos = {t["id"] for t in tasks if t["expected"].get("should_abstain")}
    pred_pos = {r["id"] for r in results if r["abstained"]}
    tp = len(actual_pos & pred_pos)
    precision = tp / len(pred_pos) if pred_pos else 1.0
    recall = tp / len(actual_pos) if actual_pos else 1.0

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_tasks": total,
        "n_passed": passed,
        "success_rate": round(passed / total, 4) if total else 0.0,
        "citation_accuracy": round(cit_ok / len(cit_tasks), 4) if cit_tasks else 0.0,
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 3)
        if latencies
        else 0.0,
        "p95_latency_ms": round(percentile(latencies, 95), 3),
        "tasks": results,
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(REPO_ROOT, "eval",
                                                    "golden_tasks.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "eval",
                                                  "results.json"))
    args = ap.parse_args()
    metrics = run(args.tasks)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"tasks={metrics['n_tasks']} passed={metrics['n_passed']} "
          f"success_rate={metrics['success_rate']}")
    print(f"citation_accuracy={metrics['citation_accuracy']} "
          f"abstention_p={metrics['abstention_precision']} "
          f"abstention_r={metrics['abstention_recall']}")
    print(f"mean_latency_ms={metrics['mean_latency_ms']} "
          f"p95_latency_ms={metrics['p95_latency_ms']}")
    print(f"wrote {args.out}")
    for r in metrics["tasks"]:
        if not r["passed"]:
            print(f"  FAIL {r['id']}: {r['failures']}")


if __name__ == "__main__":
    main()
