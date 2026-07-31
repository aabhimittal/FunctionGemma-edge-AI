"""Tests for the adaptive escalation-budget controller."""

import unittest

from functiongemma.budget import EscalationBudgetController
from functiongemma.cascade import Decision
from functiongemma.confidence import Calibrator


def _decision(tier):
    return Decision(request="x", tier=tier, call=None, confidence=0.5)


class TestBudgetController(unittest.TestCase):
    def test_overspend_lowers_tau(self):
        cal = Calibrator(tau=0.8)
        ctrl = EscalationBudgetController(cal, target_escalation=0.2, gain=0.5)
        # Feed a stream that escalates far above the 20% target.
        for _ in range(40):
            ctrl.observe(_decision("cloud"))
        self.assertLess(cal.tau, 0.8)  # tau pushed down to accept more locally.

    def test_underspend_raises_tau(self):
        cal = Calibrator(tau=0.4)
        ctrl = EscalationBudgetController(cal, target_escalation=0.5, gain=0.5)
        for _ in range(40):
            ctrl.observe(_decision("edge"))  # 0% escalation, below 50% target.
        self.assertGreater(cal.tau, 0.4)

    def test_tau_never_below_floor(self):
        cal = Calibrator(tau=0.5)
        ctrl = EscalationBudgetController(cal, target_escalation=0.0, gain=1.0, tau_floor=0.3)
        for _ in range(200):
            ctrl.observe(_decision("cloud"))
        self.assertGreaterEqual(cal.tau, 0.3)

    def test_tau_never_above_one(self):
        cal = Calibrator(tau=0.9)
        ctrl = EscalationBudgetController(cal, target_escalation=1.0, gain=1.0)
        for _ in range(200):
            ctrl.observe(_decision("edge"))
        self.assertLessEqual(cal.tau, 1.0)

    def test_no_adjustment_before_warmup(self):
        cal = Calibrator(tau=0.6)
        ctrl = EscalationBudgetController(cal, target_escalation=0.2)
        for _ in range(5):  # fewer than the 10-sample warmup.
            ctrl.observe(_decision("cloud"))
        self.assertEqual(cal.tau, 0.6)

    def test_escalation_rate_tracks_window(self):
        cal = Calibrator(tau=0.5)
        ctrl = EscalationBudgetController(cal, window=10)
        for _ in range(10):
            ctrl.observe(_decision("cloud"))
        self.assertAlmostEqual(ctrl.escalation_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
