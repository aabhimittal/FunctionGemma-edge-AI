"""Multi-intent planning  —  parallel function calling for compound requests.

Real users batch intents: *"what's the weather in Paris and set a timer for 10
minutes"*. Modern function-calling models handle this by emitting an **array**
of calls (parallel function calling). A compact edge model trained on one-call
data cannot — but the *system* still can, by planning above the model:

    compound request
        |  split on coordination markers ("and", "then", ";")
        v
    clause 1 ── cascade ──> call 1        (each clause routed independently,
    clause 2 ── cascade ──> call 2         so easy clauses stay on the edge
        |                                  and hard ones may escalate)
        v
    Plan(steps=[Decision, Decision])

The dangerous edge case is the **false split**: "play Simon and Garfunkel" is
one intent, not two. The planner is conservative — a split is kept only if
*every* clause independently resolves to a valid call; otherwise it falls back
to routing the whole request as a single intent. A wrong merge costs one
escalation; a wrong split executes a hallucinated call. We pay the cheap cost.

With a real parallel-FC model you would skip the splitter and validate each
element of the emitted array with the same ``parse``/``constrain`` machinery —
the Plan shape below stays identical, which is the point.
"""

import re
from dataclasses import dataclass, field

from .cascade import Cascade

# Coordination markers that separate intents. Order matters: longer first.
_SPLIT_RE = re.compile(r"\s*(?:\band then\b|\bthen\b|\band\b|;|,\s*then)\s*", re.IGNORECASE)

MAX_STEPS = 5  # an edge assistant plan should be small; 50 steps is an attack.


@dataclass
class Plan:
    """An ordered set of routed calls answering one compound request."""

    request: str
    steps: list = field(default_factory=list)  # list[cascade.Decision]
    split: bool = False  # False -> treated as a single intent.

    @property
    def calls(self):
        """The validated calls, in order, skipping abstentions."""
        return [d.call for d in self.steps if d.call is not None]

    def to_dict(self):
        return {
            "request": self.request,
            "split": self.split,
            "steps": [d.to_dict() for d in self.steps],
        }


class Planner:
    """Split-if-safe planner over a :class:`Cascade`."""

    def __init__(self, cascade=None, max_steps=MAX_STEPS):
        self.cascade = cascade or Cascade()
        self.max_steps = max_steps

    def plan(self, request):
        """Return a :class:`Plan` for a possibly-compound request."""
        clauses = [c for c in _SPLIT_RE.split(request or "") if c.strip()]

        if len(clauses) > 1 and len(clauses) <= self.max_steps:
            decisions = [self.cascade.route(c.strip()) for c in clauses]
            # Keep the split ONLY if every clause resolved to a real call —
            # one abstaining clause means we probably cut a single intent in
            # half ("Simon and Garfunkel"), so retreat to the whole request.
            if all(d.call is not None for d in decisions):
                return Plan(request=request, steps=decisions, split=True)

        return Plan(request=request, steps=[self.cascade.route(request)], split=False)
