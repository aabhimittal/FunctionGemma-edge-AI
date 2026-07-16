"""Calibrated confidence & abstention  —  the first half of the novel idea.

A compact edge model is *right most of the time and wrong some of the time*.
The interesting question is not "is it right?" but "does it **know** when it is
likely wrong?" If it does, we can let it answer locally when confident and
escalate to a bigger model when not (see `functiongemma.cascade`).

Two pieces:

1. ``score(request, call)`` — a confidence signal in [0, 1].
   * With a *real* model you would use the token log-probabilities of the
     generated call (e.g. the margin between the top-1 and top-2 tool logits,
     or the mean token log-prob). Pass that in as ``model_logprob`` and it is
     used directly.
   * With the runnable mock we have no logits, so we derive a cheap proxy from
     lexical overlap between the request and the chosen tool. It behaves like a
     real confidence signal (high on clear requests, low on vague ones) which
     is all the cascade needs to demonstrate the mechanism.

2. ``Calibrator`` — a raw score is not a probability. We fit a single decision
   threshold ``tau`` on a labelled validation set using the standard
   *selective prediction* objective: **maximise coverage subject to a target
   selective accuracy**. This is the tried-and-tested part; the same recipe is
   used to calibrate abstention in production classifiers.
"""

import json
import math
import re

from .tools import TOOLS, get_tool

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return set(_WORD.findall(text.lower()))


def score(request, call, model_logprob=None, tools=TOOLS):
    """Return a confidence in [0, 1] for ``call`` given ``request``.

    If ``model_logprob`` (a real model's mean token log-prob, <= 0) is given it
    is squashed to [0, 1] and used directly. Otherwise a lexical-overlap proxy
    is computed so the mock pipeline still yields a meaningful signal.
    """
    if call is None or call.get("name") is None:
        return 0.0  # an abstention carries no confidence in a tool.

    if model_logprob is not None:
        # Map a log-prob (<= 0) to (0, 1] with a logistic; -0.7 nats ~= 0.5.
        return 1.0 / (1.0 + math.exp(-(model_logprob + 2.0)))

    tool = get_tool(call["name"], tools)
    if tool is None:
        return 0.0
    req = _tokens(request)
    # Signal words that point at this tool: its name parts + description words.
    signal = _tokens(tool["name"].replace("_", " ") + " " + tool["description"])
    # Also credit argument values that literally appear in the request.
    for value in call.get("arguments", {}).values():
        signal |= _tokens(str(value))
    if not req:
        return 0.0
    overlap = len(req & signal) / len(req)
    # Affine map of overlap into confidence: a floor so any matched tool starts
    # above zero, a slope that only *full* lexical coverage drives near 1.0.
    return round(min(1.0, 0.2 + 0.9 * overlap), 4)


class Calibrator:
    """Fits and stores the abstention threshold ``tau``.

    Fit on validation pairs of (confidence, was_correct). ``accept(conf)`` then
    answers whether the edge model may keep a prediction of that confidence.
    """

    def __init__(self, tau=0.5, target_accuracy=0.9):
        self.tau = tau
        self.target_accuracy = target_accuracy

    def fit(self, confidences, correct):
        """Pick the lowest ``tau`` whose accepted set still hits target accuracy.

        Lower tau => higher coverage (we answer more locally) but risk lower
        accuracy. We sweep candidate thresholds and keep the most permissive one
        that satisfies the accuracy target — maximising coverage under a quality
        floor, the classic risk-coverage trade-off.
        """
        pairs = sorted(zip(confidences, correct, strict=False), key=lambda p: p[0])
        candidates = sorted({c for c, _ in pairs} | {0.0, 1.0})
        best_tau = 1.0  # if nothing qualifies, abstain on everything.
        for cand in candidates:
            accepted = [ok for conf, ok in pairs if conf >= cand]
            if not accepted:
                continue
            acc = sum(accepted) / len(accepted)
            if acc >= self.target_accuracy:
                best_tau = cand
                break  # candidates ascending => first hit is most permissive.
        self.tau = round(best_tau, 4)
        return self

    def accept(self, confidence):
        """True if the edge model is confident enough to answer locally."""
        return confidence >= self.tau

    # --- persistence: a calibrator is a model artifact, so it round-trips ----

    def to_dict(self):
        return {"tau": self.tau, "target_accuracy": self.target_accuracy}

    @classmethod
    def from_dict(cls, d):
        return cls(tau=d["tau"], target_accuracy=d.get("target_accuracy", 0.9))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
