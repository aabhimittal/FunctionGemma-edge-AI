"""Tests for schema-constrained decoding (repair/projection)."""

import unittest

from functiongemma.constrained import constrain
from functiongemma.tools import TOOLS


class TestConstrain(unittest.TestCase):
    def test_passes_valid_call_unchanged(self):
        raw = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        call, repairs = constrain(raw, TOOLS)
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(repairs, [])

    def test_coerces_integer_string(self):
        raw = '{"name": "set_timer", "arguments": {"minutes": "10"}}'
        call, repairs = constrain(raw, TOOLS)
        self.assertEqual(call["arguments"]["minutes"], 10)
        self.assertTrue(repairs)

    def test_fixes_misspelled_tool_name(self):
        raw = '{"name": "get_weathr", "arguments": {"city": "Paris"}}'
        call, repairs = constrain(raw, TOOLS)
        self.assertEqual(call["name"], "get_weather")
        self.assertTrue(any("get_weather" in r for r in repairs))

    def test_drops_unknown_argument(self):
        raw = '{"name": "get_weather", "arguments": {"city": "Paris", "units": "F"}}'
        call, repairs = constrain(raw, TOOLS)
        self.assertNotIn("units", call["arguments"])
        self.assertTrue(any("units" in r for r in repairs))

    def test_ignores_surrounding_prose(self):
        raw = 'Sure! {"name": "play_music", "arguments": {"query": "jazz"}} done.'
        call, _ = constrain(raw, TOOLS)
        self.assertEqual(call["name"], "play_music")

    def test_returns_none_on_abstention(self):
        call, _ = constrain('{"name": null, "arguments": {}}', TOOLS)
        self.assertIsNone(call)

    def test_returns_none_on_garbage(self):
        call, _ = constrain("no json here", TOOLS)
        self.assertIsNone(call)


if __name__ == "__main__":
    unittest.main()
