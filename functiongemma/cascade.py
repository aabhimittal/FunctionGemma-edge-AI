"""The Confidence-Gated Edge->Cloud Cascade  —  where the novel pieces meet.

This is the contribution of the repo. Function calling on the edge is a
*selective* problem: a 2B model resolves the easy 80-90% of requests instantly
and privately, and should hand the hard tail to a larger model instead of
guessing. The cascade wires that together:

    request
       |
   [ edge model ]  --raw text-->  constrain (schema)  -->  validate
       |                                                      |
   confidence.score  -->  Calibrator.accept(tau)? --- yes --> answer locally
       |                                                      (tier = "edge")
       no
       |
   [ cloud model ]  --raw text-->  constrain --> validate --> answer
                                                            (tier = "cloud")

Three industrial dials fall straight out of this and are all logged per call:
  * **coverage**  — fraction answered on-device (latency, privacy, cost win).
  * **escalation rate** — fraction sent to the cloud (the cost you pay).
  * **selective accuracy** — accuracy *on the accepted set*, the quality floor
    the Calibrator was fitted to hold.

Everything below is backend-agnostic: swap the mock for a real on-device Gemma
and a hosted large model and the routing logic is unchanged.
"""

import time
from dataclasses import dataclass, field

from .confidence import Calibrator, score
from .constrained import constrain
from .model import cloud_backend, mock_backend
from .parser import InvalidCall, validate
from .prompt import build_prompt
from .tools import TOOLS


@dataclass
class Decision:
    """The full record of one routed request — the unit we log and monitor."""

    request: str
    tier: str  # "edge" | "cloud" | "abstain"
    call: dict | None
    confidence: float
    repairs: list = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self):
        return {
            "request": self.request,
            "tier": self.tier,
            "call": self.call,
            "confidence": self.confidence,
            "repairs": self.repairs,
            "latency_ms": round(self.latency_ms, 3),
            "error": self.error,
        }


class Cascade:
    """Route a request through edge -> (maybe) cloud with a calibrated gate."""

    def __init__(
        self,
        edge_backend=mock_backend,
        cloud_backend=cloud_backend,
        calibrator=None,
        tools=TOOLS,
    ):
        self.edge_backend = edge_backend
        self.cloud_backend = cloud_backend
        self.calibrator = calibrator or Calibrator(tau=0.5)
        self.tools = tools

    def _run_tier(self, request, backend):
        """One tier: prompt -> backend -> constrain -> validate. Returns
        ``(call_or_None, repairs, error)``."""
        prompt = build_prompt(self.tools, request)
        raw = backend(prompt)
        call, repairs = constrain(raw, self.tools)
        if call is None:
            return None, repairs, "no valid call in output"
        try:
            return validate(call, self.tools), repairs, None
        except InvalidCall as exc:
            return None, repairs, str(exc)

    def route(self, request):
        """Route one request and return a :class:`Decision`."""
        start = time.perf_counter()

        edge_call, repairs, err = self._run_tier(request, self.edge_backend)
        conf = score(request, edge_call, tools=self.tools)

        # Gate: keep the edge answer only if it is valid AND confident enough.
        if edge_call is not None and self.calibrator.accept(conf):
            return Decision(
                request=request, tier="edge", call=edge_call, confidence=conf,
                repairs=repairs, latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Escalate to the larger model.
        cloud_call, cloud_repairs, cloud_err = self._run_tier(
            request, self.cloud_backend
        )
        latency = (time.perf_counter() - start) * 1000
        if cloud_call is not None:
            return Decision(
                request=request, tier="cloud", call=cloud_call,
                confidence=conf, repairs=repairs + cloud_repairs,
                latency_ms=latency,
            )

        # Neither tier produced a valid call -> honest abstention.
        return Decision(
            request=request, tier="abstain", call=None, confidence=conf,
            repairs=repairs + cloud_repairs, latency_ms=latency,
            error=cloud_err or err,
        )


def coverage_report(decisions):
    """Summarise a batch of decisions into the three headline MLops metrics."""
    n = len(decisions) or 1
    tiers = [d.tier for d in decisions]
    return {
        "total": len(decisions),
        "coverage_edge": round(tiers.count("edge") / n, 4),
        "escalation_rate": round(tiers.count("cloud") / n, 4),
        "abstain_rate": round(tiers.count("abstain") / n, 4),
        "avg_latency_ms": round(sum(d.latency_ms for d in decisions) / n, 3),
    }
