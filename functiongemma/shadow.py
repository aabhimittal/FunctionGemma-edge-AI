"""Shadow & canary evaluation  —  proving a new model on real traffic, safely.

The offline gate (``pipelines/evaluate.py``) says a candidate is good on the
golden set. The golden set is not production. The last mile of any model rollout
is the same everywhere in industry:

  1. **shadow** — run the candidate on real traffic *in parallel*, serve only the
     incumbent's answer, and measure how often they disagree;
  2. **canary** — if agreement and error rates look sane, serve the candidate to
     a small traffic slice; and
  3. **promote or roll back** — on the evidence, not on hope.

``ShadowRunner`` is step 1 and the decision rule for step 3. Its single hard
requirement is the one that makes shadowing safe at all: **the candidate can
never affect the user**. It runs behind a deadline, every exception is swallowed
into a counter, and the returned answer is always the incumbent's. A candidate
that crashes on every request degrades to "0% agreement, 100% errors" — a clear
*fail* verdict — and not to a single failed user request.

The verdict deliberately refuses to answer without enough traffic:
``insufficient_data`` is a real outcome, and shipping on five samples is how
teams talk themselves into a regression.
"""

from collections import Counter

from .reliability import guarded_call


def calls_agree(a, b):
    """Do two decisions represent the same action? (Both abstaining counts.)"""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.get("name") == b.get("name") and a.get("arguments") == b.get("arguments")


class ShadowRunner:
    """Run a candidate cascade alongside production without exposing it."""

    def __init__(self, production, candidate, timeout_s=1.0, max_divergences=50):
        self.production = production
        self.candidate = candidate
        self.timeout_s = timeout_s
        self.max_divergences = max_divergences
        self.n = 0
        self.agreements = 0
        self.candidate_errors = 0
        self.divergences = []  # bounded sample, for eyeballing what changed.
        self.tier_shift = Counter()  # (prod_tier -> cand_tier) transitions.

    def route(self, request):
        """Serve the production decision; score the candidate in the shadow."""
        decision = self.production.route(request)

        # Everything below is best-effort. A broken candidate must cost the user
        # nothing, so faults become counters, never exceptions.
        shadow, error = guarded_call(
            self.candidate.route, request, timeout_s=self.timeout_s
        )
        self.n += 1
        if error is not None or shadow is None:
            self.candidate_errors += 1
            return decision

        self.tier_shift[(decision.tier, shadow.tier)] += 1
        if calls_agree(decision.call, shadow.call):
            self.agreements += 1
        elif len(self.divergences) < self.max_divergences:
            self.divergences.append({
                "request": request,
                "production": decision.call,
                "candidate": shadow.call,
                "production_tier": decision.tier,
                "candidate_tier": shadow.tier,
            })
        return decision

    @property
    def agreement_rate(self):
        return round(self.agreements / self.n, 4) if self.n else 0.0

    @property
    def error_rate(self):
        return round(self.candidate_errors / self.n, 4) if self.n else 0.0

    def report(self):
        return {
            "n": self.n,
            "agreement_rate": self.agreement_rate,
            "candidate_error_rate": self.error_rate,
            "divergences": len(self.divergences),
            "tier_shift": {f"{a}->{b}": c for (a, b), c in self.tier_shift.items()},
        }

    def verdict(self, min_samples=30, min_agreement=0.9, max_error_rate=0.01):
        """``insufficient_data`` | ``pass`` | ``fail`` — the rollout decision."""
        if self.n < min_samples:
            return "insufficient_data"
        if self.error_rate > max_error_rate:
            return "fail"
        return "pass" if self.agreement_rate >= min_agreement else "fail"
