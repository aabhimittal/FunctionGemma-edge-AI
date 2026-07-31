"""Drift & health monitoring  —  catching a silent model regression in the wild.

Telemetry counts things; a monitor *decides whether they are OK*. The most
dangerous production failure is not a crash — crashes page you. It is the model
quietly getting worse: a phrasing shift, a new locale, a tool renamed upstream,
and the abstain/escalation/repair rates creep up while every request still
returns 200. Nobody notices until users do.

Two complementary signals, both computed online from the same
``cascade.Decision`` stream, both dependency-free:

  * **rate thresholds** — rolling abstain / escalation / repair rates over a
    window, alerting when any crosses an operator-set ceiling. Simple, legible,
    and what most on-call runbooks actually key on.
  * **distribution drift (PSI)** — the Population Stability Index of the model's
    confidence distribution now versus a frozen baseline. PSI is the standard
    credit-risk / ML-monitoring metric for "has the input distribution moved?":
    < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift. It catches
    drift that hasn't yet shown up as a rate change — the earliest warning.

``report()`` returns data, and ``alerts()`` returns the subset that breached a
threshold, so this slots straight into a health endpoint or a cron check.
"""

import math
from collections import deque

_PSI_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]  # confidence buckets; last is inclusive.


def _histogram(values, bins=_PSI_BINS):
    """Fraction of ``values`` falling in each [bins[i], bins[i+1]) bucket."""
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
    total = len(values) or 1
    return [c / total for c in counts]


def psi(baseline, current, eps=1e-4):
    """Population Stability Index between two confidence samples.

    PSI = sum over bins of (curr - base) * ln(curr / base), with an epsilon
    floor on empty bins so the log stays finite.
    """
    b = _histogram(baseline)
    c = _histogram(current)
    total = 0.0
    for pb, pc in zip(b, c, strict=False):
        pb, pc = max(pb, eps), max(pc, eps)
        total += (pc - pb) * math.log(pc / pb)
    return round(total, 4)


class HealthMonitor:
    """Rolling health + drift over a stream of decisions."""

    def __init__(
        self,
        window=200,
        max_abstain_rate=0.35,
        max_escalation_rate=0.35,
        max_repair_rate=0.5,
        psi_alert=0.25,
    ):
        self.window = window
        self.max_abstain_rate = max_abstain_rate
        self.max_escalation_rate = max_escalation_rate
        self.max_repair_rate = max_repair_rate
        self.psi_alert = psi_alert
        self._tiers = deque(maxlen=window)
        self._repairs = deque(maxlen=window)
        self._confidence = deque(maxlen=window)
        self._baseline = None  # frozen confidence distribution.

    def set_baseline(self, decisions=None):
        """Freeze the reference confidence distribution (from a known-good run)."""
        if decisions is None:
            self._baseline = list(self._confidence)
        else:
            self._baseline = [d.confidence for d in decisions]
        return self

    def observe(self, decision):
        self._tiers.append(decision.tier)
        self._repairs.append(1 if decision.repairs else 0)
        self._confidence.append(decision.confidence)

    def _rate(self, seq, predicate):
        return round(sum(1 for x in seq if predicate(x)) / (len(seq) or 1), 4)

    def report(self):
        """Current rolling health, plus PSI vs baseline if one is set."""
        rep = {
            "n": len(self._tiers),
            "abstain_rate": self._rate(self._tiers, lambda t: t == "abstain"),
            "escalation_rate": self._rate(self._tiers, lambda t: t == "cloud"),
            "repair_rate": round(sum(self._repairs) / (len(self._repairs) or 1), 4),
        }
        if self._baseline:
            rep["psi"] = psi(self._baseline, list(self._confidence))
        return rep

    def alerts(self):
        """The subset of health signals that breached their threshold."""
        rep = self.report()
        out = []
        if rep["abstain_rate"] > self.max_abstain_rate:
            out.append(f"abstain_rate {rep['abstain_rate']} > {self.max_abstain_rate}")
        if rep["escalation_rate"] > self.max_escalation_rate:
            out.append(
                f"escalation_rate {rep['escalation_rate']} > {self.max_escalation_rate}"
            )
        if rep["repair_rate"] > self.max_repair_rate:
            out.append(f"repair_rate {rep['repair_rate']} > {self.max_repair_rate}")
        if "psi" in rep and rep["psi"] > self.psi_alert:
            out.append(f"confidence drift PSI {rep['psi']} > {self.psi_alert}")
        return out
