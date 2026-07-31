"""Scale & reliability edge cases, exercised through the cascade.

Where test_industry_edge_cases.py covers the *runtime* layer (guard/plan/exec),
this suite covers the *scale & operations* layer: a crowded tool catalogue that
won't fit the context window, a cloud tier that is slow or down, cost-budget
drift, and confidence drift — the failures that show up once the thing is
actually deployed to many devices.
"""

import time
import unittest

from functiongemma.budget import EscalationBudgetController
from functiongemma.cascade import Cascade
from functiongemma.confidence import Calibrator
from functiongemma.reliability import CircuitBreaker
from functiongemma.retrieval import ToolRetriever
from functiongemma.tools import TOOLS


def _big_catalog(n=300):
    filler = [
        {"name": f"vendor_{i}_do", "description": f"Vendor {i} operation.",
         "parameters": {"x": {"type": "string", "description": "arg"}}}
        for i in range(n)
    ]
    return TOOLS + filler


class TestRetrievalCascade(unittest.TestCase):
    """Routing stays correct when the catalogue is far larger than the prompt."""

    def setUp(self):
        catalog = _big_catalog()
        self.cascade = Cascade(
            tools=catalog,
            calibrator=Calibrator(tau=0.3),
            retriever=ToolRetriever(catalog),
            top_k=8,
        )

    def test_correct_tool_from_300_tool_catalogue(self):
        d = self.cascade.route("what's the weather in Paris?")
        self.assertIsNotNone(d.call)
        self.assertEqual(d.call["name"], "get_weather")

    def test_off_topic_request_abstains_not_crashes(self):
        d = self.cascade.route("ponder the meaning of existence")
        self.assertEqual(d.tier, "abstain")


class TestCloudReliability(unittest.TestCase):
    """The cloud tier is a network call: it can raise, hang, or be down."""

    def test_raising_cloud_degrades_to_abstain(self):
        def dead_cloud(prompt):
            raise ConnectionError("no route to host")

        cascade = Cascade(cloud_backend=dead_cloud, calibrator=Calibrator(tau=1.0))
        # tau=1.0 forces escalation; cloud raises -> must abstain, not crash.
        d = cascade.route("set a timer for 10 minutes")
        self.assertEqual(d.tier, "abstain")
        self.assertIsNone(d.call)

    def test_circuit_breaker_opens_and_stops_calling_cloud(self):
        calls = {"n": 0}

        def dead_cloud(prompt):
            calls["n"] += 1
            raise ConnectionError("down")

        breaker = CircuitBreaker(failure_threshold=3)
        cascade = Cascade(cloud_backend=dead_cloud, calibrator=Calibrator(tau=1.0),
                          cloud_breaker=breaker)
        for _ in range(3):
            cascade.route("set a timer for 5 minutes")
        self.assertEqual(breaker.state, "open")
        calls_before = calls["n"]
        cascade.route("set a timer for 5 minutes")  # breaker open -> skip cloud
        self.assertEqual(calls["n"], calls_before)  # cloud was NOT called again

    def test_hanging_cloud_is_bounded_by_timeout(self):
        def slow_cloud(prompt):
            time.sleep(5)

        cascade = Cascade(cloud_backend=slow_cloud, calibrator=Calibrator(tau=1.0),
                          cloud_timeout_s=0.05)
        start = time.perf_counter()
        d = cascade.route("set a timer for 10 minutes")
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0)          # did not wait the full 5s
        self.assertEqual(d.tier, "abstain")

    def test_clean_abstention_does_not_trip_breaker(self):
        # An honest "no tool fits" from the cloud is not an outage.
        breaker = CircuitBreaker(failure_threshold=2)
        cascade = Cascade(calibrator=Calibrator(tau=1.0), cloud_breaker=breaker)
        for _ in range(5):
            cascade.route("tell me a joke")     # both tiers abstain, no fault
        self.assertEqual(breaker.state, "closed")


class TestBudgetIntegration(unittest.TestCase):
    """The controller holds a cost budget over a live cascade stream."""

    def test_controller_pulls_escalation_toward_target(self):
        requests = ["weather in Paris", "set a timer for 5 minutes",
                    "play jazz", "message Sam", "schedule a meeting at 3"]
        stream = [requests[i % len(requests)] for i in range(60)]
        target = 0.3

        # Baseline: a fixed, miserly threshold escalates heavily and never adapts.
        base = Cascade(calibrator=Calibrator(tau=0.95))
        base_esc = sum(base.route(r).tier == "cloud" for r in stream) / len(stream)

        # Controlled: same start, but the budget controller adapts tau online.
        cal = Calibrator(tau=0.95)
        cascade = Cascade(calibrator=cal)
        ctrl = EscalationBudgetController(cal, target_escalation=target, gain=0.5)
        ctrl_esc = 0
        for r in stream:
            d = cascade.route(r)
            ctrl.observe(d)
            ctrl_esc += d.tier == "cloud"
        ctrl_esc /= len(stream)

        # The controller both cuts escalation and lands closer to the budget.
        self.assertLess(ctrl_esc, base_esc)
        self.assertLess(abs(ctrl_esc - target), abs(base_esc - target))


if __name__ == "__main__":
    unittest.main()
