"""Tests for large-catalogue tool retrieval (BM25)."""

import unittest

from functiongemma.prompt import build_prompt
from functiongemma.retrieval import ToolRetriever
from functiongemma.tools import TOOLS


def _big_catalog(n=200):
    """The real tools plus n filler tools, to simulate a crowded device."""
    filler = [
        {
            "name": f"app_{i}_action",
            "description": f"Perform action {i} in third-party app number {i}.",
            "parameters": {"value": {"type": "string", "description": "an argument"}},
        }
        for i in range(n)
    ]
    return TOOLS + filler


class TestToolRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = ToolRetriever(_big_catalog())

    def test_retrieves_relevant_tool_from_crowd(self):
        picked = [t["name"] for t in self.retriever.select("weather in Paris", k=5)]
        self.assertIn("get_weather", picked)

    def test_top_k_is_respected(self):
        self.assertEqual(len(self.retriever.select("set a timer", k=3)), 3)

    def test_narrows_prompt_size(self):
        full = build_prompt(_big_catalog(), "set a timer for 10 minutes")
        narrowed = build_prompt(
            self.retriever.select("set a timer for 10 minutes", k=5),
            "set a timer for 10 minutes",
        )
        self.assertLess(len(narrowed), len(full) / 5)

    def test_deterministic(self):
        a = [t["name"] for t in self.retriever.select("play jazz", k=5)]
        b = [t["name"] for t in self.retriever.select("play jazz", k=5)]
        self.assertEqual(a, b)

    def test_off_topic_request_still_returns_tools(self):
        # No lexical match -> returns first k rather than an empty prompt.
        picked = self.retriever.select("qwerty zxcvb", k=4)
        self.assertEqual(len(picked), 4)

    def test_k_larger_than_catalogue_returns_all(self):
        r = ToolRetriever(TOOLS)
        self.assertEqual(len(r.select("anything", k=999)), len(TOOLS))

    def test_rare_term_beats_common_term(self):
        # "timer" is discriminative; BM25 should rank set_timer top for it.
        top = self.retriever.select("timer", k=1)[0]["name"]
        self.assertEqual(top, "set_timer")


if __name__ == "__main__":
    unittest.main()
