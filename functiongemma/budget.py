"""Adaptive escalation-budget controller  —  holding a cloud-cost target online.

The calibrator fits the abstention threshold ``tau`` *offline*, to hold a quality
floor. But the thing an operator actually pays for and must cap is the
**escalation rate** — every request that falls through to the cloud costs money,
latency and a privacy hop. In production that rate drifts: a new app, a new
phrasing, a time-of-day shift, and suddenly 40% of traffic is escalating against
a 20% budget.

This is a classic control problem, so we solve it with a controller, not a
retrain. ``EscalationBudgetController`` watches the live tier decisions and nudges
``tau`` to keep the escalation rate near a target:

    higher tau  => edge accepts fewer answers => MORE escalation
    lower  tau  => edge accepts more answers  => LESS escalation

So to reduce escalation we *lower* tau. A proportional update over a sliding
window does it, with the gain small enough to be stable and tau clamped to
[floor, 1]. The floor stops the controller from trading the whole quality bar
away to hit a cost number — cost control must not silently disable abstention.
"""

from collections import deque


class EscalationBudgetController:
    """Proportional controller that steers a calibrator's ``tau`` to a budget."""

    def __init__(
        self,
        calibrator,
        target_escalation=0.2,
        window=50,
        gain=0.3,
        tau_floor=0.2,
    ):
        self.calibrator = calibrator
        self.target = target_escalation
        self.gain = gain
        self.tau_floor = tau_floor
        self._window = deque(maxlen=window)

    @property
    def escalation_rate(self):
        """Current escalation rate over the sliding window."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def observe(self, decision):
        """Record one :class:`~functiongemma.cascade.Decision` and adjust tau.

        Only acts once the window has enough signal (>= 10 samples) so a couple
        of early escalations don't swing the threshold. Returns the (possibly
        unchanged) tau so callers can log the trajectory.
        """
        self._window.append(1 if decision.tier == "cloud" else 0)
        if len(self._window) < 10:
            return self.calibrator.tau

        error = self.escalation_rate - self.target
        # error > 0 => escalating too much => lower tau to accept more locally.
        new_tau = self.calibrator.tau - self.gain * error
        self.calibrator.tau = round(min(1.0, max(self.tau_floor, new_tau)), 4)
        return self.calibrator.tau
