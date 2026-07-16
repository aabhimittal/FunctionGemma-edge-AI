"""Minimal tests: run with `python -m pytest` (or `python -m unittest`)."""

import unittest

from functiongemma import InvalidCall, parse, run


class TestPipeline(unittest.TestCase):
    def test_weather(self):
        call = run("What's the weather in Paris?")
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["arguments"]["city"], "Paris")

    def test_timer(self):
        call = run("Set a timer for 10 minutes")
        self.assertEqual(call["name"], "set_timer")
        self.assertEqual(call["arguments"]["minutes"], 10)

    def test_abstain(self):
        # No matching tool -> the model returns null -> parser rejects it.
        with self.assertRaises(InvalidCall):
            run("Tell me a joke")

    def test_rejects_unknown_tool(self):
        with self.assertRaises(InvalidCall):
            parse('{"name": "launch_rocket", "arguments": {}}')

    def test_rejects_bad_type(self):
        with self.assertRaises(InvalidCall):
            parse('{"name": "set_timer", "arguments": {"minutes": "soon"}}')


if __name__ == "__main__":
    unittest.main()
