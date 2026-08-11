"""Multi-turn dialogue state  —  because assistants are conversations.

Everything so far treats a request as a self-contained event. Real assistant
traffic is not shaped that way:

    user: "set a timer"                -> which is missing a duration
    asst: "How many minutes?"
    user: "10 minutes"                 -> only meaningful given the last turn

    user: "what's the weather in Paris"
    user: "what about Tokyo?"          -> same tool, new entity

A compact edge model trained on single-turn data cannot resolve either one. The
*system* can, by keeping a small amount of dialogue state above the model —
which is exactly how production assistants do it, and it is state, not
intelligence, so it belongs here rather than in the checkpoint.

Two mechanisms, both deliberately conservative:

  * **slot filling** — when the model returns a well-formed but incomplete call
    (``Decision.partial``), we ask for the missing argument and hold the partial
    call. The next utterance fills that one slot, typed by the tool schema.
  * **follow-up resolution** — a short utterance that resolves to nothing on its
    own, right after a successful call, is treated as a new entity for the *same*
    tool ("what about Tokyo?").

The conservatism rule is the same one the planner uses: **if an utterance
resolves to a valid call on its own, it is a new intent** — it is never consumed
as a slot answer or a follow-up. A wrong "new intent" costs one clarification; a
wrong "slot fill" executes something the user did not ask for.

Safety properties this layer must hold, all of them tested:

  * dialogue state **expires** (``pending_ttl_s``) — walking away and coming back
    to say "yes" must not fire a stale side effect;
  * a confirmation is **bound to the exact call** it was issued for — "yes" can
    never authorise anything else, and a bare "yes" with nothing pending does
    nothing; and
  * history is **bounded** — an edge device cannot grow a conversation forever.

The real-model seam: with a checkpoint that accepts dialogue history you would
put prior turns in the prompt and delete the heuristics below. The state machine,
the TTL and the confirmation binding stay exactly as they are — those are policy,
not modelling.
"""

import re
import time
from collections import deque
from dataclasses import dataclass, field

from .executor import Agent
from .guard import screen_request
from .tools import get_tool

# Function words carrying no entity content, stripped when reading a follow-up.
_STOPWORDS = {
    "a", "about", "also", "an", "and", "any", "are", "as", "at", "be", "but",
    "by", "do", "does", "for", "from", "how", "i", "in", "is", "it", "its",
    "me", "my", "no", "now", "of", "ok", "okay", "on", "or", "please", "so",
    "that", "the", "then", "there", "this", "to", "too", "up", "us", "was",
    "we", "what", "when", "where", "which", "who", "why", "with", "yes", "you",
}

_AFFIRM = {
    "y", "ya", "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm",
    "confirmed", "do it", "send it", "go ahead", "please do", "affirmative",
}
_DECLINE = {
    "n", "no", "nope", "cancel", "stop", "don't", "dont", "never mind",
    "nevermind", "forget it", "abort",
}

MAX_FOLLOW_UP_TOKENS = 6  # a follow-up is short; a paragraph is a new intent.


@dataclass
class Pending:
    """Dialogue state carried between turns. Always has a deadline."""

    kind: str  # "slot" | "confirm"
    call: dict
    question: str
    created_at: float
    missing: str | None = None
    flags: list = field(default_factory=list)

    def to_dict(self):
        return {
            "kind": self.kind,
            "call": self.call,
            "question": self.question,
            "missing": self.missing,
        }


class Session:
    """One conversation with one user on one device."""

    def __init__(self, agent=None, max_turns=20, pending_ttl_s=120.0,
                 _clock=time.monotonic):
        self.agent = agent or Agent()
        self.pending_ttl_s = pending_ttl_s
        self._clock = _clock
        self.history = deque(maxlen=max_turns)  # bounded: no unbounded growth.
        self.pending = None
        self.last_call = None  # last successfully executed call, for follow-ups.

    # --- public API ---------------------------------------------------------

    def ask(self, utterance):
        """Process one conversational turn. Returns a report dict."""
        self._expire_pending()

        if self.pending is not None:
            report = self._continue_pending(utterance)
            if report is not None:
                return self._record(report)

        # A bare yes/no with nothing pending must never do anything.
        if self._intent(utterance) is not None and self.pending is None:
            return self._record({
                "utterance": utterance, "kind": "ignored", "steps": [],
                "reason": "no pending question to answer",
            })

        follow_up = self._as_follow_up(utterance)
        if follow_up is not None:
            entry = self.agent.execute_call(follow_up, screen_request(utterance))
            self._remember(entry)
            return self._record({
                "utterance": utterance, "kind": "follow_up", "steps": [entry],
            })

        return self._record(self._new_intent(utterance))

    def reset(self):
        """Drop dialogue state (new conversation), keeping the agent."""
        self.pending = None
        self.last_call = None
        self.history.clear()

    # --- turn handling ------------------------------------------------------

    def _new_intent(self, utterance):
        """Route a fresh request; hold a clarification if a slot is missing."""
        report = self.agent.handle(utterance)
        report["utterance"] = utterance
        report["kind"] = "new"

        for entry in report["steps"]:
            if entry.get("status") == "needs_confirmation":
                # Hold the exact call, so a later "yes" can only authorise this.
                self.pending = Pending(
                    kind="confirm", call=entry["call"],
                    question=f"Confirm {entry['call']['name']}?",
                    created_at=self._clock(), flags=report.get("flags", []),
                )
                report["question"] = self.pending.question
            self._remember(entry)

        # Only a single-intent turn can become a slot-filling clarification;
        # asking "which one did you mean?" across a compound request is a
        # different (and much easier to get wrong) interaction.
        if self.pending is None and not report.get("split"):
            question = self._clarification_for(report)
            if question is not None:
                report["kind"] = "clarify"
                report["question"] = question
        return report

    def _continue_pending(self, utterance):
        """Try to consume ``utterance`` as an answer. None => not an answer."""
        pending = self.pending
        intent = self._intent(utterance)

        if intent is False:  # explicit decline: drop the held call.
            self.pending = None
            return {"utterance": utterance, "kind": "declined", "steps": [],
                    "cancelled": pending.to_dict()}

        if pending.kind == "confirm":
            if intent is True:
                self.pending = None
                # Execute exactly the held call, with the original turn's flags.
                entry = self.agent.execute_call(
                    pending.call, pending.flags, confirm=True
                )
                self._remember(entry)
                return {"utterance": utterance, "kind": "confirmed",
                        "steps": [entry]}
            return None  # anything else is a new intent, not an approval.

        # Slot filling: a self-sufficient utterance is a new intent instead.
        if self._resolves_alone(utterance):
            self.pending = None
            return None
        value = self._extract_slot(pending.call["name"], pending.missing, utterance)
        if value is None:
            return {"utterance": utterance, "kind": "clarify",
                    "steps": [], "question": pending.question}

        self.pending = None
        call = {
            "name": pending.call["name"],
            "arguments": {**pending.call.get("arguments", {}), pending.missing: value},
        }
        entry = self.agent.execute_call(call, screen_request(utterance))
        self._remember(entry)
        return {"utterance": utterance, "kind": "slot_filled", "steps": [entry]}

    # --- helpers ------------------------------------------------------------

    def _expire_pending(self):
        if self.pending is None:
            return
        if self._clock() - self.pending.created_at >= self.pending_ttl_s:
            self.pending = None  # stale intent: never resurrect it silently.

    def _clarification_for(self, report):
        """If the turn failed only for a missing argument, hold and ask."""
        for entry in report["steps"]:
            if entry.get("status") != "abstained":
                continue
            partial = entry.get("partial")
            missing = entry.get("missing")
            if not partial or not missing:
                continue
            self.pending = Pending(
                kind="slot", call=partial, missing=missing,
                question=f"What {missing} for {partial['name']}?",
                created_at=self._clock(),
            )
            return self.pending.question
        return None

    def _resolves_alone(self, utterance):
        """Would this utterance produce a valid call by itself? (No execution.)"""
        return self.agent.planner.cascade.route(utterance).call is not None

    def _as_follow_up(self, utterance):
        """Rewrite a short trailing utterance onto the previous tool, or None."""
        if self.last_call is None:
            return None
        tokens = _content_tokens(utterance)
        if not tokens or len(utterance.split()) > MAX_FOLLOW_UP_TOKENS:
            return None
        if self._resolves_alone(utterance):
            return None  # a standalone intent is never a follow-up.
        tool = get_tool(self.last_call["name"])
        if tool is None:
            return None
        slot = next(
            (p for p, s in tool["parameters"].items() if s["type"] == "string"), None
        )
        if slot is None:
            return None
        return {
            "name": self.last_call["name"],
            "arguments": {**self.last_call["arguments"], slot: " ".join(tokens).title()},
        }

    def _extract_slot(self, tool_name, slot, utterance):
        """Read one typed value out of a slot answer. None if unreadable."""
        tool = get_tool(tool_name)
        spec = (tool or {}).get("parameters", {}).get(slot, {})
        if spec.get("type") == "integer":
            m = re.search(r"-?\d+", utterance)
            return int(m.group()) if m else None
        tokens = _content_tokens(utterance)
        return " ".join(tokens).title() if tokens else None

    def _intent(self, utterance):
        """True=affirm, False=decline, None=neither."""
        text = utterance.strip().lower().rstrip("!.")
        if text in _AFFIRM:
            return True
        if text in _DECLINE:
            return False
        return None

    def _remember(self, entry):
        if entry.get("status") == "executed" and entry.get("call"):
            self.last_call = entry["call"]

    def _record(self, report):
        self.history.append(report)
        return report


def _content_tokens(text):
    """Words carrying entity content: no stopwords, no punctuation."""
    words = re.findall(r"[\w'-]+", text or "", flags=re.UNICODE)
    return [w for w in words if w.lower() not in _STOPWORDS]
