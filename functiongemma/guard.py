"""Guardrails  —  the policy layer between a valid call and an executed call.

The parser (step 4) answers "is this call *well-formed*?". On a real device
that is not enough: a well-formed call can still be unsafe to run. Industry
deployments put a policy layer between validation and execution that answers
"is this call *allowed, sane and within budget* right now?". This module is
that layer, kept schema-driven and dependency-free:

  * **request screening** — cheap lexical heuristics that flag likely prompt
    injection ("ignore previous instructions", "reveal your system prompt")
    and oversized inputs *before* the model runs. A flag does not block the
    request; it downgrades trust so side-effecting tools need confirmation.
  * **argument sanitization** — strip control characters, collapse whitespace
    and cap string lengths, so a hostile or confused model cannot smuggle
    escape sequences or megabyte payloads into downstream systems.
  * **range constraints** — enforce ``minimum``/``maximum`` declared in the
    tool catalogue (e.g. a timer of -5 or 10^9 minutes is well-typed but
    nonsense; the schema, not the code, says what "sane" means).
  * **tool policy** — deny-listed tools never run; side-effecting tools
    (``side_effects``) require explicit confirmation when trust is low.
  * **rate limiting** — a per-tool sliding-window budget, because a looping
    agent that fires ``send_message`` 500×/minute is an incident, not a bug.

The output is a :class:`Verdict`, never an exception: policy decisions are
data the caller (executor, server, agent) acts on and logs.
"""

import re
import time
from dataclasses import dataclass, field

from .tools import TOOLS, get_tool

# Phrases that appear in the overwhelming majority of real prompt-injection
# attempts against tool-calling systems. Lexical matching is not a security
# boundary on its own — it is the cheap first tripwire that downgrades trust.
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (all |any )?(previous|prior|above)",
    r"(reveal|show|print|repeat) (your|the) (system )?prompt",
    r"you are now\b",
    r"developer mode",
    r"jailbreak",
    r"pretend (you are|to be)",
    r"\bnew instructions?:",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# ASCII/Unicode control characters (except \t \n) — never legitimate in a
# tool argument, frequently abused for log forging and terminal escapes.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u2028\u2029]")

MAX_REQUEST_CHARS = 2000  # a spoken/typed assistant request is short; 100 KB is an attack.
MAX_ARG_CHARS = 280


@dataclass
class Verdict:
    """The guard's decision about one call. Data, not an exception."""

    allowed: bool
    call: dict | None = None  # the call with *sanitized* arguments.
    reasons: list = field(default_factory=list)
    requires_confirmation: bool = False

    def to_dict(self):
        return {
            "allowed": self.allowed,
            "call": self.call,
            "reasons": self.reasons,
            "requires_confirmation": self.requires_confirmation,
        }


def screen_request(text, max_chars=MAX_REQUEST_CHARS):
    """Pre-model screening of the raw user request. Returns a list of flags.

    An empty list means nothing suspicious. Flags lower trust; they do not
    reject — the model may still abstain or the guard may still allow a
    harmless read-only call.
    """
    flags = []
    if not text or not text.strip():
        flags.append("empty request")
        return flags
    if len(text) > max_chars:
        flags.append(f"oversized request ({len(text)} > {max_chars} chars)")
    if _INJECTION_RE.search(text):
        flags.append("possible prompt injection")
    if _CONTROL_RE.search(text):
        flags.append("control characters in request")
    return flags


def sanitize_value(value, max_chars=MAX_ARG_CHARS):
    """Clean one string argument: strip controls, collapse space, cap length."""
    cleaned = _CONTROL_RE.sub("", value)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    truncated = len(cleaned) > max_chars
    return cleaned[:max_chars], truncated


class Guardrail:
    """Stateful policy checker: one instance per session/device.

    ``deny`` lists tools that must never execute. ``side_effects`` lists tools
    that change the world outside the device (messages, calendar writes) — when
    the request was flagged by :func:`screen_request`, those come back with
    ``requires_confirmation=True`` so a human stays in the loop.
    """

    def __init__(
        self,
        tools=TOOLS,
        deny=(),
        side_effects=("send_message", "create_calendar_event"),
        rate_per_minute=30,
        max_arg_chars=MAX_ARG_CHARS,
    ):
        self.tools = tools
        self.deny = set(deny)
        self.side_effects = set(side_effects)
        self.rate_per_minute = rate_per_minute
        self.max_arg_chars = max_arg_chars
        self._recent_calls = {}  # tool name -> [monotonic timestamps]

    def check(self, call, request_flags=()):
        """Full policy check of a *validated* call. Returns a :class:`Verdict`."""
        name = call["name"]
        if name in self.deny:
            return Verdict(False, reasons=[f"tool {name!r} is deny-listed"])

        if not self._within_rate(name):
            return Verdict(
                False,
                reasons=[f"rate limit: >{self.rate_per_minute} {name!r} calls/minute"],
            )

        tool = get_tool(name, self.tools)
        reasons = []
        args = {}
        for pname, value in call.get("arguments", {}).items():
            spec = tool["parameters"].get(pname, {})
            if isinstance(value, str):
                value, truncated = sanitize_value(value, self.max_arg_chars)
                if truncated:
                    reasons.append(f"truncated {pname!r} to {self.max_arg_chars} chars")
                if not value:
                    return Verdict(
                        False, reasons=[f"argument {pname!r} empty after sanitization"]
                    )
            if isinstance(value, int) and not isinstance(value, bool):
                lo, hi = spec.get("minimum"), spec.get("maximum")
                if lo is not None and value < lo:
                    return Verdict(
                        False, reasons=[f"{pname}={value} below minimum {lo}"]
                    )
                if hi is not None and value > hi:
                    return Verdict(
                        False, reasons=[f"{pname}={value} above maximum {hi}"]
                    )
            args[pname] = value

        needs_confirm = bool(request_flags) and name in self.side_effects
        if needs_confirm:
            reasons.append(
                f"low-trust request + side-effecting tool {name!r}: hold for confirmation"
            )
        return Verdict(
            True,
            call={"name": name, "arguments": args},
            reasons=reasons,
            requires_confirmation=needs_confirm,
        )

    def _within_rate(self, name, now=None):
        """Sliding 60 s window per tool; record the call only if admitted."""
        now = time.monotonic() if now is None else now
        window = [t for t in self._recent_calls.get(name, []) if now - t < 60.0]
        if len(window) >= self.rate_per_minute:
            self._recent_calls[name] = window
            return False
        window.append(now)
        self._recent_calls[name] = window
        return True
