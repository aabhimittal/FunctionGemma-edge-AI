"""PII redaction  —  because the event log is user data.

The telemetry event log is the project's most valuable asset (replay it to mine
hard cases and build the next training set) *and* its biggest liability: it
contains verbatim user requests and tool arguments — names, phone numbers,
addresses, card numbers. On an edge device that log may sync to the cloud or sit
in a backup. Logging it raw is how a function-calling feature becomes a privacy
incident.

The industrial control is **redaction at the boundary**: scrub PII the instant a
record is written, not "later, in a batch job". This module is a dependency-free
redactor for the common, high-confidence PII shapes (email, phone, credit card,
SSN, IBAN-ish, long digit runs). It is:

  * **conservative** — it only touches strings that clearly match a PII pattern,
    so ``get_weather(city="Paris")`` is left untouched;
  * **idempotent** — redacting an already-redacted string is a no-op, so it is
    safe to apply anywhere in the pipeline; and
  * **non-mutating** — it copies, so the live call the device executes keeps the
    real values while only the *logged* copy is scrubbed.

Regex PII detection is not a compliance guarantee — it is the pragmatic 95% that
stops the obvious leaks. Pair it with retention limits and access control.
"""

import re

# Order matters: card/SSN before the generic long-digit rule so they win.
_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")),
    ("DIGITS", re.compile(r"\b\d{6,}\b")),
]


class Redactor:
    """Replace PII spans with ``[TYPE]`` placeholders."""

    def __init__(self, patterns=_PATTERNS):
        self.patterns = patterns

    def redact_text(self, text):
        if not isinstance(text, str):
            return text
        for label, pattern in self.patterns:
            placeholder = f"[{label}]"
            # Skip work if the placeholder is already present for idempotency of
            # exact re-runs; the regexes themselves never match "[EMAIL]" etc.
            text = pattern.sub(placeholder, text)
        return text

    def redact_call(self, call):
        """Return a copy of a call with string argument values redacted."""
        if not call:
            return call
        args = {
            k: self.redact_text(v) if isinstance(v, str) else v
            for k, v in call.get("arguments", {}).items()
        }
        return {"name": call.get("name"), "arguments": args}

    def redact_event(self, event):
        """Redact a telemetry event dict (``request`` + ``call``) without mutating it."""
        out = dict(event)
        if "request" in out:
            out["request"] = self.redact_text(out["request"])
        if out.get("call"):
            out["call"] = self.redact_call(out["call"])
        return out


#: A ready-to-use default redactor.
default_redactor = Redactor()


def redact_text(text):
    return default_redactor.redact_text(text)
