"""Micro-benchmark: cascade routing latency + tier split.

Not a model benchmark (the mock is instant) — it measures the *routing overhead*
the cascade adds and reports the tier distribution and coverage on the golden
set. On real hardware the same script gives you the edge/cloud latency profile
you size the deployment against.

Usage:
    python -m benchmarks.latency --n 2000
"""

import argparse
import statistics
import time

from functiongemma.cascade import Cascade, coverage_report
from pipelines.evaluate import load_golden


def main():
    ap = argparse.ArgumentParser(description="Benchmark cascade routing latency.")
    ap.add_argument("--n", type=int, default=2000, help="requests to route")
    ap.add_argument("--golden", default="data/eval/golden.jsonl")
    args = ap.parse_args()

    requests = [ex["request"] for ex in load_golden(args.golden)]
    cascade = Cascade()

    # Warm up, then time each route call.
    for r in requests[:10]:
        cascade.route(r)

    decisions, times = [], []
    for i in range(args.n):
        req = requests[i % len(requests)]
        t0 = time.perf_counter()
        decisions.append(cascade.route(req))
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    report = coverage_report(decisions)
    print(f"routed {args.n} requests")
    print(f"  p50 latency : {statistics.median(times):.3f} ms")
    print(f"  p95 latency : {times[int(0.95 * len(times)) - 1]:.3f} ms")
    print(f"  edge coverage : {report['coverage_edge']:.1%}")
    print(f"  escalation    : {report['escalation_rate']:.1%}")
    print(f"  abstain       : {report['abstain_rate']:.1%}")


if __name__ == "__main__":
    main()
