"""Tests for reliability primitives: circuit breaker + guarded_call."""

import time
import unittest

from functiongemma.reliability import CircuitBreaker, CircuitOpen, guarded_call


class TestGuardedCall(unittest.TestCase):
    def test_success_returns_result(self):
        self.assertEqual(guarded_call(lambda x: x + 1, 41), (42, None))

    def test_exception_becomes_error(self):
        def boom():
            raise ValueError("nope")

        result, error = guarded_call(boom)
        self.assertIsNone(result)
        self.assertIn("ValueError", error)

    def test_timeout_is_caught(self):
        result, error = guarded_call(lambda: time.sleep(2), timeout_s=0.05)
        self.assertIsNone(result)
        self.assertIn("timeout", error)


class TestCircuitBreaker(unittest.TestCase):
    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        self.assertEqual(cb.state, "open")
        self.assertFalse(cb.allow())

    def test_success_resets(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_success()
        self.assertEqual(cb.state, "closed")

    def test_half_open_after_cooldown_then_recovers(self):
        clock = [1000.0]
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=30, _clock=lambda: clock[0])
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.allow())          # open
        clock[0] += 31                        # cooldown elapses
        self.assertTrue(cb.allow())           # half-open trial admitted
        self.assertEqual(cb.state, "half-open")
        cb.record_success()
        self.assertEqual(cb.state, "closed")  # recovered

    def test_half_open_failure_reopens(self):
        clock = [0.0]
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=10, _clock=lambda: clock[0])
        cb.record_failure()                   # open
        clock[0] += 11
        self.assertTrue(cb.allow())           # half-open
        cb.record_failure()                   # trial fails
        self.assertEqual(cb.state, "open")

    def test_call_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        with self.assertRaises(CircuitOpen):
            cb.call(lambda: 1)


if __name__ == "__main__":
    unittest.main()
