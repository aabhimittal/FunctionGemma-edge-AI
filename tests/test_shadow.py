"""Tests for the shadow / canary rollout harness."""

import time
import unittest

from functiongemma.cascade import Cascade, Decision
from functiongemma.shadow import ShadowRunner, calls_agree

REQUESTS = ["weather in Paris", "set a timer for 10 minutes", "play jazz",
            "message Sam", "tell me a joke"]


class _StubCascade:
    """Minimal stand-in with a scripted route()."""

    def __init__(self, fn):
        self.route = fn


class TestCallsAgree(unittest.TestCase):
    def test_identical_calls_agree(self):
        a = {"name": "set_timer", "arguments": {"minutes": 10}}
        self.assertTrue(calls_agree(a, dict(a)))

    def test_both_abstaining_agree(self):
        self.assertTrue(calls_agree(None, None))

    def test_abstain_vs_call_disagree(self):
        self.assertFalse(calls_agree(None, {"name": "set_timer", "arguments": {}}))

    def test_same_tool_different_arguments_disagree(self):
        a = {"name": "set_timer", "arguments": {"minutes": 10}}
        b = {"name": "set_timer", "arguments": {"minutes": 5}}
        self.assertFalse(calls_agree(a, b))


class TestShadowRunner(unittest.TestCase):
    def test_identical_candidate_agrees_fully(self):
        runner = ShadowRunner(Cascade(), Cascade())
        for i in range(40):
            runner.route(REQUESTS[i % len(REQUESTS)])
        self.assertEqual(runner.agreement_rate, 1.0)
        self.assertEqual(runner.verdict(), "pass")

    def test_production_answer_is_served_even_if_candidate_crashes(self):
        def exploding(request):
            raise RuntimeError("candidate is broken")

        prod = Cascade()
        runner = ShadowRunner(prod, _StubCascade(exploding))
        decision = runner.route("weather in Paris")
        # The user still gets the right answer...
        self.assertEqual(decision.call["name"], "get_weather")
        # ...and the breakage is recorded, not raised.
        self.assertEqual(runner.error_rate, 1.0)

    def test_crashing_candidate_fails_the_verdict(self):
        def exploding(request):
            raise RuntimeError("boom")

        runner = ShadowRunner(Cascade(), _StubCascade(exploding))
        for i in range(40):
            runner.route(REQUESTS[i % len(REQUESTS)])
        self.assertEqual(runner.verdict(), "fail")

    def test_hanging_candidate_is_bounded_by_timeout(self):
        def slow(request):
            time.sleep(5)

        runner = ShadowRunner(Cascade(), _StubCascade(slow), timeout_s=0.05)
        start = time.perf_counter()
        runner.route("weather in Paris")
        self.assertLess(time.perf_counter() - start, 2.0)
        self.assertEqual(runner.candidate_errors, 1)

    def test_divergences_are_recorded_and_bounded(self):
        def always_timer(request):
            return Decision(request=request, tier="edge",
                            call={"name": "set_timer", "arguments": {"minutes": 1}},
                            confidence=1.0)

        runner = ShadowRunner(Cascade(), _StubCascade(always_timer),
                              max_divergences=3)
        for _ in range(40):
            runner.route("weather in Paris")
        self.assertEqual(len(runner.divergences), 3)      # sample stays bounded
        self.assertLess(runner.agreement_rate, 1.0)
        self.assertEqual(runner.verdict(), "fail")

    def test_verdict_refuses_without_enough_traffic(self):
        runner = ShadowRunner(Cascade(), Cascade())
        for _ in range(5):
            runner.route("weather in Paris")
        self.assertEqual(runner.verdict(min_samples=30), "insufficient_data")

    def test_report_records_tier_shift(self):
        runner = ShadowRunner(Cascade(), Cascade())
        for i in range(10):
            runner.route(REQUESTS[i % len(REQUESTS)])
        report = runner.report()
        self.assertEqual(report["n"], 10)
        self.assertTrue(any("->" in k for k in report["tier_shift"]))

    def test_empty_runner_reports_safely(self):
        runner = ShadowRunner(Cascade(), Cascade())
        self.assertEqual(runner.report()["n"], 0)
        self.assertEqual(runner.verdict(), "insufficient_data")


if __name__ == "__main__":
    unittest.main()
