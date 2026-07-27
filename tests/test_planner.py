"""Unit tests for the multi-intent planner."""

import unittest

from functiongemma.planner import Planner


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = Planner()

    def test_single_intent_is_not_split(self):
        plan = self.planner.plan("What's the weather in Paris?")
        self.assertFalse(plan.split)
        self.assertEqual(len(plan.calls), 1)
        self.assertEqual(plan.calls[0]["name"], "get_weather")

    def test_compound_request_splits_into_two_calls(self):
        plan = self.planner.plan(
            "What's the weather in Paris and set a timer for 10 minutes"
        )
        self.assertTrue(plan.split)
        self.assertEqual([c["name"] for c in plan.calls], ["get_weather", "set_timer"])
        self.assertEqual(plan.calls[0]["arguments"]["city"], "Paris")
        self.assertEqual(plan.calls[1]["arguments"]["minutes"], 10)

    def test_then_separator(self):
        plan = self.planner.plan("Set a timer for 3 minutes then play Radiohead")
        self.assertTrue(plan.split)
        self.assertEqual([c["name"] for c in plan.calls], ["set_timer", "play_music"])

    def test_false_split_falls_back_to_single_intent(self):
        # "and" inside one intent: a naive splitter would emit a bogus call
        # for the clause "Garfunkel". The planner must retreat to one intent.
        plan = self.planner.plan("Play Simon and Garfunkel")
        self.assertFalse(plan.split)
        self.assertEqual(len(plan.calls), 1)
        self.assertEqual(plan.calls[0]["name"], "play_music")

    def test_three_intents(self):
        plan = self.planner.plan(
            "weather in Tokyo and set a timer for 5 minutes and play jazz"
        )
        self.assertTrue(plan.split)
        self.assertEqual(len(plan.calls), 3)

    def test_too_many_clauses_not_split(self):
        request = " and ".join(["set a timer for 2 minutes"] * 10)
        plan = self.planner.plan(request)
        self.assertFalse(plan.split)

    def test_empty_request_yields_abstention(self):
        plan = self.planner.plan("")
        self.assertFalse(plan.split)
        self.assertEqual(plan.calls, [])


if __name__ == "__main__":
    unittest.main()
