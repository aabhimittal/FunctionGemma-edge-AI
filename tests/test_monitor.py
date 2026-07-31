"""Tests for drift & health monitoring."""

import unittest

from functiongemma.cascade import Decision
from functiongemma.monitor import HealthMonitor, psi


def _d(tier="edge", confidence=0.8, repairs=None):
    return Decision(request="x", tier=tier, call={"name": "get_weather"},
                    confidence=confidence, repairs=repairs or [])


class TestPSI(unittest.TestCase):
    def test_identical_distributions_have_zero_psi(self):
        sample = [0.1, 0.3, 0.5, 0.7, 0.9] * 10
        self.assertLess(psi(sample, sample), 0.01)

    def test_shifted_distribution_has_high_psi(self):
        baseline = [0.9] * 100  # confident
        shifted = [0.1] * 100   # collapsed to low confidence
        self.assertGreater(psi(baseline, shifted), 0.25)


class TestHealthMonitor(unittest.TestCase):
    def test_healthy_stream_has_no_alerts(self):
        mon = HealthMonitor()
        for _ in range(100):
            mon.observe(_d("edge", 0.8))
        self.assertEqual(mon.alerts(), [])

    def test_abstain_flood_alerts(self):
        mon = HealthMonitor(max_abstain_rate=0.3)
        for _ in range(100):
            mon.observe(_d("abstain", 0.0))
        self.assertTrue(any("abstain_rate" in a for a in mon.alerts()))

    def test_escalation_flood_alerts(self):
        mon = HealthMonitor(max_escalation_rate=0.3)
        for _ in range(100):
            mon.observe(_d("cloud", 0.4))
        self.assertTrue(any("escalation_rate" in a for a in mon.alerts()))

    def test_confidence_drift_alerts_against_baseline(self):
        mon = HealthMonitor(psi_alert=0.25)
        healthy = [_d("edge", 0.9) for _ in range(100)]
        mon.set_baseline(healthy)
        for _ in range(100):  # confidence collapses relative to baseline.
            mon.observe(_d("edge", 0.1))
        self.assertTrue(any("drift" in a for a in mon.alerts()))

    def test_empty_monitor_does_not_crash(self):
        mon = HealthMonitor()
        self.assertEqual(mon.report()["n"], 0)
        self.assertEqual(mon.alerts(), [])


if __name__ == "__main__":
    unittest.main()
