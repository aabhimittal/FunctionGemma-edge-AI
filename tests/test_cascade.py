"""Tests for the confidence-gated edge->cloud cascade."""

import unittest

from functiongemma.cascade import Cascade, coverage_report
from functiongemma.confidence import Calibrator


class TestCascade(unittest.TestCase):
    def test_confident_request_stays_on_edge(self):
        cascade = Cascade(calibrator=Calibrator(tau=0.3))
        decision = cascade.route("what's the weather in Paris?")
        self.assertEqual(decision.tier, "edge")
        self.assertEqual(decision.call["name"], "get_weather")

    def test_high_threshold_forces_escalation(self):
        # tau=1.0 => nothing clears the edge gate, everything escalates.
        cascade = Cascade(calibrator=Calibrator(tau=1.0))
        decision = cascade.route("set a timer for 10 minutes")
        self.assertIn(decision.tier, ("cloud", "abstain"))

    def test_unanswerable_request_abstains(self):
        cascade = Cascade(calibrator=Calibrator(tau=0.3))
        decision = cascade.route("tell me a joke")
        self.assertEqual(decision.tier, "abstain")
        self.assertIsNone(decision.call)

    def test_decision_is_serializable(self):
        cascade = Cascade(calibrator=Calibrator(tau=0.3))
        d = cascade.route("play jazz").to_dict()
        self.assertIn("tier", d)
        self.assertIn("latency_ms", d)

    def test_coverage_report_sums_to_one(self):
        cascade = Cascade(calibrator=Calibrator(tau=0.3))
        decisions = [cascade.route(r) for r in
                     ["weather in Paris", "set a timer for 5 minutes", "tell me a joke"]]
        rep = coverage_report(decisions)
        total = rep["coverage_edge"] + rep["escalation_rate"] + rep["abstain_rate"]
        self.assertAlmostEqual(total, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
