"""
CLI entry point.

Usage:
  python -m src.cli --samples            # run all 5 sample_questions.json cases
  python -m src.cli --question "..."     # run a single ad-hoc question
  python -m src.cli --samples --out outputs/sample_run_outputs.json
"""
from __future__ import annotations
import argparse
import json
import time

from src.graph import build_graph
from src.data_loader import load_sample_questions
from src.state import AgentState


def run_one(graph, question: str, question_id: str | None = None, demo_force_fail: bool = False) -> dict:
    init_state: AgentState = {
        "question": question,
        "question_id": question_id,
        "retrieved": [],
        "revision_count": 0,
        "warnings": [],
        "logs": [],
        "_demo_force_verification_failure": demo_force_fail,
    }
    t0 = time.time()
    result = graph.invoke(init_state)
    elapsed = time.time() - t0

    print("=" * 80)
    print(f"Q ({question_id or 'ad-hoc'}): {question}")
    print("-" * 80)
    print("EXECUTION LOG:")
    for line in result["logs"]:
        print(f"  - {line}")
    print("-" * 80)
    print("FINAL OUTPUT (schema-shaped):")
    print(json.dumps(result["final_output"], indent=2))
    print(f"(total time: {elapsed:.2f}s)")

    return {
        "question_id": question_id,
        "question": question,
        "logs": result["logs"],
        "final_output": result["final_output"],
        "elapsed_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true", help="run all sample_questions.json cases")
    parser.add_argument("--question", type=str, help="run a single ad-hoc question")
    parser.add_argument("--out", type=str, default=None, help="write JSON results to this path")
    parser.add_argument(
        "--demo-verification-failure",
        action="store_true",
        help="deterministically force the first draft to fail verification (test case 5 demo)",
    )
    args = parser.parse_args()

    graph = build_graph()
    results = []

    if args.samples:
        for q in load_sample_questions():
            results.append(run_one(graph, q["question"], q["question_id"], args.demo_verification_failure))
    elif args.question:
        results.append(run_one(graph, args.question, demo_force_fail=args.demo_verification_failure))
    else:
        parser.print_help()
        return

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)} result(s) to {args.out}")


if __name__ == "__main__":
    main()
