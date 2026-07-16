"""Tests for the offline MLops pipeline steps (data gen, eval, config)."""

import json
import os
import tempfile
import unittest

from functiongemma.config import Config, load_config
from pipelines.evaluate import evaluate
from pipelines.generate_data import generate


class TestDataGen(unittest.TestCase):
    def test_deterministic_and_valid(self):
        a = list(generate(50, seed=1))
        b = list(generate(50, seed=1))
        self.assertEqual(a, b)  # same seed -> same data.
        for ex in a:
            completion = json.loads(ex["completion"])
            self.assertIn("name", completion)  # every row is a valid target.

    def test_contains_negatives(self):
        rows = list(generate(200, seed=0, negative_ratio=0.5))
        names = [json.loads(r["completion"])["name"] for r in rows]
        self.assertIn(None, names)  # abstention examples are present.


class TestEvaluate(unittest.TestCase):
    def test_mock_clears_gate(self):
        metrics, pairs = evaluate()
        self.assertGreaterEqual(metrics["tool_accuracy"], 0.8)
        self.assertEqual(len(pairs), metrics["n"])


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(Config().port, 8000)

    def test_env_override(self):
        os.environ["FG_PORT"] = "9999"
        try:
            self.assertEqual(load_config().port, 9999)
        finally:
            del os.environ["FG_PORT"]

    def test_file_override(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"tau": 0.77}, fh)
            path = fh.name
        try:
            self.assertEqual(load_config(path).tau, 0.77)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
