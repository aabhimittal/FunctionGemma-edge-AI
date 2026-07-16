"""Tests for confidence scoring and the abstention Calibrator."""

import unittest

from functiongemma.confidence import Calibrator, score


class TestScore(unittest.TestCase):
    def test_clear_request_is_confident(self):
        call = {"name": "get_weather", "arguments": {"city": "Paris"}}
        self.assertGreater(score("what's the weather in Paris?", call), 0.5)

    def test_abstention_has_zero_confidence(self):
        self.assertEqual(score("tell me a joke", {"name": None}), 0.0)
        self.assertEqual(score("tell me a joke", None), 0.0)

    def test_logprob_path_is_used_when_given(self):
        call = {"name": "set_timer", "arguments": {"minutes": 5}}
        high = score("x", call, model_logprob=-0.1)
        low = score("x", call, model_logprob=-5.0)
        self.assertGreater(high, low)


class TestCalibrator(unittest.TestCase):
    def test_fit_picks_threshold_meeting_target(self):
        # Low-confidence items are wrong, high-confidence ones are right.
        confs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        correct = [0, 0, 0, 1, 1, 1]
        cal = Calibrator(target_accuracy=1.0).fit(confs, correct)
        # Must reject the wrong (<=0.3) items and accept the right ones.
        self.assertTrue(cal.accept(0.7))
        self.assertFalse(cal.accept(0.3))

    def test_roundtrip_serialization(self):
        cal = Calibrator(tau=0.42, target_accuracy=0.95)
        restored = Calibrator.from_dict(cal.to_dict())
        self.assertEqual(restored.tau, 0.42)
        self.assertEqual(restored.target_accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()
