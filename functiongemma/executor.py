"""Safe execution  —  where a validated, allowed call finally touches code.

The model *chooses*; this module *executes* — and execution on a device has its
own failure modes the model never sees: a tool that hangs (network), a tool
that raises, a tool whose output is attacker-controlled. Industry rules baked
in here:

  * **registry, not dispatch-by-eval** — only functions explicitly registered
    in ``IMPLEMENTATIONS`` can run. There is no path from model text to code.
  * **timeouts** — every call runs in a worker thread with a deadline. An edge
    assistant that hangs is worse than one that errors.
  * **structured results** — success and failure are the same shape
    (``{"ok", "tool", "result"|"error", "latency_ms"}``) so telemetry,
    retries and the agent loop never special-case.
  * **tool output is data, not instructions** — results are returned verbatim
    to the caller and never re-entered into a prompt as trusted text. A
    calendar title saying "ignore previous instructions" must stay a string.

``Agent`` composes the full production path for one request:

    screen -> plan (multi-intent cascade) -> guard -> execute -> report
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from .guard import Guardrail, screen_request
from .planner import Planner

DEFAULT_TIMEOUT_S = 2.0

# The parser's message for an incomplete call; the name of the absent argument
# is what a dialogue layer needs in order to ask a useful question.
_MISSING_RE = re.compile(r"missing argument:\s*(\w+)")


def _missing_argument(error):
    """Extract the absent argument name from a validation error, or None."""
    m = _MISSING_RE.search(error or "")
    return m.group(1) if m else None


# --- reference tool implementations (deterministic, offline) ----------------
# Stand-ins for the device services a real assistant would bind to. They are
# deterministic so tests and demos are reproducible.

def _get_weather(city):
    return {"city": city, "temp_c": 18 + len(city) % 10, "sky": "partly cloudy"}


def _set_timer(minutes):
    return {"timer_id": f"t-{minutes}", "expires_in_s": minutes * 60}


def _send_message(to, body):
    return {"delivered_to": to, "chars": len(body)}


def _create_calendar_event(title, hour):
    return {"event_id": f"ev-{hour:02d}", "title": title, "hour": hour}


def _play_music(query):
    return {"now_playing": query}


IMPLEMENTATIONS = {
    "get_weather": _get_weather,
    "set_timer": _set_timer,
    "send_message": _send_message,
    "create_calendar_event": _create_calendar_event,
    "play_music": _play_music,
}


def execute(call, implementations=IMPLEMENTATIONS, timeout_s=DEFAULT_TIMEOUT_S):
    """Run one validated+allowed call. Always returns a structured result."""
    name = call["name"]
    impl = implementations.get(name)
    start = time.perf_counter()

    def _fail(error):
        return {
            "ok": False, "tool": name, "error": error,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        }

    if impl is None:
        return _fail(f"no implementation registered for {name!r}")

    # No context manager: `with` would join the worker on exit, so a hung tool
    # would stall us past the deadline anyway. shutdown(wait=False) lets the
    # caller get its timeout answer now; the orphaned thread ends on its own.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(impl, **call.get("arguments", {}))
    try:
        result = future.result(timeout=timeout_s)
    except FutureTimeout:
        return _fail(f"timeout after {timeout_s}s")
    except Exception as exc:  # a tool bug must not crash the assistant.
        return _fail(f"{type(exc).__name__}: {exc}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return {
        "ok": True, "tool": name, "result": result,
        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
    }


class Agent:
    """The full request lifecycle: screen -> plan -> guard -> execute.

    ``handle`` returns a report dict with one entry per planned step; steps
    blocked by policy or held for confirmation are reported, not silently
    dropped — the caller (UI, server) decides how to surface them.
    """

    def __init__(self, planner=None, guardrail=None, implementations=IMPLEMENTATIONS,
                 timeout_s=DEFAULT_TIMEOUT_S, idempotency=None):
        self.planner = planner or Planner()
        self.guardrail = guardrail or Guardrail()
        self.implementations = implementations
        self.timeout_s = timeout_s
        # Optional IdempotencyCache: suppresses a *repeated* side effect (a
        # retry, a double-tap) inside a short window. See functiongemma.idempotency.
        self.idempotency = idempotency

    def execute_call(self, call, flags=(), confirm=False):
        """Guard, then execute, one already-validated call.

        Returns the step entry dict (``status`` plus ``call``/``outcome``/
        ``reasons``). This is the single per-call path: :meth:`handle` uses it for
        every planned step, and the multi-turn session layer
        (``functiongemma.session``) uses it for a call it assembled across turns,
        so policy and idempotency can never be bypassed by a different entry point.
        """
        entry = {}
        verdict = self.guardrail.check(call, request_flags=flags)
        entry["call"] = verdict.call or call
        if not verdict.allowed:
            entry.update(status="blocked", reasons=verdict.reasons)
            return entry
        if verdict.requires_confirmation and not confirm:
            entry.update(status="needs_confirmation", reasons=verdict.reasons)
            return entry

        outcome, duplicate = self._execute_once(verdict.call)
        entry.update(
            status="executed" if outcome["ok"] else "failed",
            outcome=outcome,
        )
        if duplicate:
            # Surfaced, never hidden: the caller should know the effect was
            # suppressed rather than performed a second time.
            entry["duplicate"] = True
        if verdict.reasons:
            entry["reasons"] = verdict.reasons
        return entry

    def _execute_once(self, call):
        """Execute, suppressing a repeated side effect. Returns (outcome, duplicate)."""
        side_effecting = call["name"] in self.guardrail.side_effects
        if self.idempotency is not None and side_effecting:
            cached = self.idempotency.get(call)
            if cached is not None:
                return cached, True
            outcome = execute(call, self.implementations, self.timeout_s)
            self.idempotency.put(call, outcome)  # failures are not cached.
            return outcome, False
        return execute(call, self.implementations, self.timeout_s), False

    def handle(self, request, confirm=False):
        """Process one user request end to end.

        ``confirm=True`` means the human has already approved side effects for
        this request (e.g. tapped "send it"), so confirmation holds are lifted.
        """
        flags = screen_request(request)
        plan = self.planner.plan(request)
        steps = []

        for decision in plan.steps:
            entry = {"tier": decision.tier, "confidence": decision.confidence}
            if decision.call is None:
                entry.update(status="abstained", error=decision.error)
                # A *incomplete* call (right tool, missing a required argument)
                # is not executable, but it is not nothing either: surface it so
                # a dialogue layer can ask for the slot instead of dropping the
                # user's intent. See functiongemma.session.
                if decision.partial is not None:
                    entry["partial"] = decision.partial
                    missing = _missing_argument(decision.error)
                    if missing:
                        entry["missing"] = missing
                steps.append(entry)
                continue
            entry.update(self.execute_call(decision.call, flags, confirm))
            steps.append(entry)

        return {
            "request": request,
            "flags": flags,
            "split": plan.split,
            "steps": steps,
        }
