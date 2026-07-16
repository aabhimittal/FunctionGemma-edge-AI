"""Observability for the running model  —  structured logs + Prometheus metrics.

You cannot improve what you cannot see. In production the two things you always
want from an inference service are:

  * a **structured event log** (one JSON line per request) you can replay to
    find drift, mine hard cases, or build the next training set; and
  * **live counters/histograms** a scraper (Prometheus) can pull for dashboards
    and alerts — request count, tier split, latency, abstain rate.

Both are here in stdlib. ``Telemetry.observe(decision)`` takes a
``cascade.Decision`` so the same object that carries the answer also carries the
metrics — no duplicate bookkeeping. ``render_prometheus()`` emits the text
exposition format the ``/metrics`` endpoint serves.
"""

import json
import threading
import time


class Telemetry:
    """Thread-safe in-process metrics + optional JSONL event log."""

    def __init__(self, event_log_path=None, latency_buckets_ms=None):
        self._lock = threading.Lock()
        self.event_log_path = event_log_path
        self.buckets = latency_buckets_ms or [5, 10, 25, 50, 100, 250, 500, 1000]
        self.counters = {
            "requests_total": 0,
            "tier_edge_total": 0,
            "tier_cloud_total": 0,
            "tier_abstain_total": 0,
            "repairs_total": 0,
        }
        self.latency_sum_ms = 0.0
        self.latency_hist = {b: 0 for b in self.buckets}
        self.latency_inf = 0  # count above the largest bucket.

    def observe(self, decision):
        """Record one routed request (a ``cascade.Decision``)."""
        with self._lock:
            self.counters["requests_total"] += 1
            self.counters[f"tier_{decision.tier}_total"] += 1
            self.counters["repairs_total"] += len(decision.repairs)
            self.latency_sum_ms += decision.latency_ms
            self._observe_latency(decision.latency_ms)
        if self.event_log_path:
            self._append_event(decision)

    def _observe_latency(self, ms):
        for b in self.buckets:  # cumulative "<= bucket" histogram.
            if ms <= b:
                self.latency_hist[b] += 1
                return
        self.latency_inf += 1

    def _append_event(self, decision):
        event = decision.to_dict()
        event["ts"] = time.time()
        with open(self.event_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def snapshot(self):
        """Return a plain dict of the current metrics (for tests / debugging)."""
        with self._lock:
            n = self.counters["requests_total"] or 1
            return {
                **self.counters,
                "avg_latency_ms": round(self.latency_sum_ms / n, 3),
                "abstain_rate": round(
                    self.counters["tier_abstain_total"] / n, 4
                ),
            }

    def render_prometheus(self):
        """Emit metrics in Prometheus text exposition format."""
        with self._lock:
            lines = []
            for key, val in self.counters.items():
                lines.append(f"# TYPE functiongemma_{key} counter")
                lines.append(f"functiongemma_{key} {val}")
            # Latency histogram in the standard bucketed form.
            lines.append("# TYPE functiongemma_latency_ms histogram")
            cumulative = 0
            for b in self.buckets:
                cumulative += self.latency_hist[b]
                lines.append(
                    f'functiongemma_latency_ms_bucket{{le="{b}"}} {cumulative}'
                )
            cumulative += self.latency_inf
            lines.append(
                f'functiongemma_latency_ms_bucket{{le="+Inf"}} {cumulative}'
            )
            lines.append(f"functiongemma_latency_ms_sum {round(self.latency_sum_ms, 3)}")
            lines.append(
                f"functiongemma_latency_ms_count {self.counters['requests_total']}"
            )
            return "\n".join(lines) + "\n"
