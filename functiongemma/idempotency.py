"""Idempotency  —  making a retried side effect happen exactly once.

Everything upstream treats a request as a fresh event. Reality doesn't: the user
double-taps send, the network layer retries a request whose response was lost, an
agent loop re-emits the same call. For a read-only tool that is harmless. For
``send_message`` it means the message goes out **twice**, and "the assistant
texted my boss twice" is a real incident.

The industrial answer is an **idempotency key**: fingerprint the operation, and
if the same fingerprint is seen again inside a short window, return the first
result instead of performing the effect again. Payment APIs (Stripe et al.) work
exactly this way. Three rules this implementation follows, each of which is a bug
if you get it wrong:

  * **only successes are cached** — a failed send must stay retryable, otherwise
    a transient error becomes permanent;
  * **the key is order-independent** — ``{"to": "Sam", "body": "hi"}`` and
    ``{"body": "hi", "to": "Sam"}`` are the same operation; and
  * **entries expire** — the same message *should* be sendable again later;
    suppression is for retries, not for the rest of time.

Read-only tools are deliberately *not* suppressed: asking for the weather twice
should return fresh weather.
"""

import hashlib
import json
import threading
import time


def fingerprint(call):
    """A stable, order-independent hash of a call's identity."""
    canonical = json.dumps(
        {"name": call.get("name"), "arguments": call.get("arguments", {})},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


class IdempotencyCache:
    """Short-TTL memory of successful side effects, keyed by call fingerprint."""

    def __init__(self, ttl_s=60.0, max_entries=256, _clock=time.monotonic):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = _clock
        self._lock = threading.Lock()
        self._entries = {}  # key -> (stored_at, outcome)

    def get(self, call):
        """Return the remembered outcome for ``call``, or None."""
        key = fingerprint(call)
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, outcome = entry
            if now - stored_at >= self.ttl_s:
                del self._entries[key]  # expired: the effect may happen again.
                return None
            return outcome

    def put(self, call, outcome):
        """Remember a *successful* outcome. Failures are never cached."""
        if not outcome.get("ok"):
            return
        key = fingerprint(call)
        now = self._clock()
        with self._lock:
            self._evict(now)
            self._entries[key] = (now, outcome)

    def _evict(self, now):
        """Drop expired entries, then the oldest if still over capacity."""
        for key in [k for k, (t, _) in self._entries.items() if now - t >= self.ttl_s]:
            del self._entries[key]
        while len(self._entries) >= self.max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k][0])
            del self._entries[oldest]
