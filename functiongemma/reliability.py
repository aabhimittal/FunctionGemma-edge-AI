"""Reliability primitives  —  because the cloud tier is a network call.

The cascade's second tier is a *remote* model. On an edge device the network is
the least reliable component in the whole system: it times out, it 503s, it
vanishes in a tunnel. Two failure modes must be handled or the assistant hangs:

  * a single call that hangs or raises — handled by running it under a deadline
    and turning any exception into a value (``guarded_call``); and
  * a dependency that is *down* — handled by a **circuit breaker** so we stop
    hammering it, fail fast, and recover automatically when it comes back.

A breaker has three states: **closed** (calls flow), **open** (calls short-
circuit immediately after too many failures), and **half-open** (after a cool-
down, one trial call decides whether to close again). This is the standard
Hystrix/resilience4j pattern, in ~40 lines of stdlib. The cascade uses it to
degrade gracefully: when the cloud tier is open, escalation is skipped and the
request abstains locally instead of blocking on a dead socket.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout


class CircuitOpen(Exception):
    """Raised when a call is short-circuited because the breaker is open."""


class CircuitBreaker:
    """Trip after ``failure_threshold`` failures; retry after ``reset_timeout``."""

    def __init__(self, failure_threshold=3, reset_timeout_s=30.0, _clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = _clock
        self._failures = 0
        self._opened_at = None
        self.state = "closed"

    def allow(self):
        """Return True if a call may proceed now (and move open -> half-open)."""
        if self.state == "open":
            if self._clock() - self._opened_at >= self.reset_timeout_s:
                self.state = "half-open"  # let exactly one trial through.
                return True
            return False
        return True

    def record_success(self):
        self._failures = 0
        self._opened_at = None
        self.state = "closed"

    def record_failure(self):
        self._failures += 1
        # A failed trial in half-open, or crossing the threshold, opens it.
        if self.state == "half-open" or self._failures >= self.failure_threshold:
            self.state = "open"
            self._opened_at = self._clock()

    def call(self, fn, *args, **kwargs):
        """Invoke ``fn`` through the breaker. Raises :class:`CircuitOpen` if open."""
        if not self.allow():
            raise CircuitOpen("circuit is open")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


def guarded_call(fn, *args, timeout_s=None, **kwargs):
    """Run ``fn`` with an optional deadline. Returns ``(result, error)``.

    Never raises: a timeout or exception comes back as the ``error`` string so
    callers stay branch-free. With no timeout it is a plain try/except; with one
    the call runs in a worker thread so a hung dependency cannot pin the caller.
    """
    if timeout_s is None:
        try:
            return fn(*args, **kwargs), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s), None
    except FutureTimeout:
        return None, f"timeout after {timeout_s}s"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
