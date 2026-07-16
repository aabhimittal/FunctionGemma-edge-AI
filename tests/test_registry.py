"""Tests for the file-backed model registry and the telemetry recorder."""

import tempfile
import unittest

from functiongemma.cascade import Cascade
from functiongemma.confidence import Calibrator
from functiongemma.registry import ModelRegistry
from functiongemma.telemetry import Telemetry


class TestRegistry(unittest.TestCase):
    def test_register_and_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelRegistry(tmp)
            v1 = reg.register("m", metrics={"tool_accuracy": 0.9})
            reg.promote(v1, "production")
            self.assertEqual(reg.current("production")["version"], v1)

    def test_promotion_archives_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelRegistry(tmp)
            v1 = reg.register("m", metrics={"tool_accuracy": 0.8})
            v2 = reg.register("m", metrics={"tool_accuracy": 0.95})
            reg.promote(v1, "production")
            reg.promote(v2, "production")
            self.assertEqual(reg.current("production")["version"], v2)
            self.assertEqual(reg.get(v1)["stage"], "archived")

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelRegistry(tmp)
            v1 = reg.register("m", metrics={"tool_accuracy": 0.9})
            reg.promote(v1, "production")
            # New instance reads the same manifest from disk.
            self.assertEqual(ModelRegistry(tmp).current("production")["version"], v1)


class TestTelemetry(unittest.TestCase):
    def test_observe_updates_counters_and_prometheus(self):
        tel = Telemetry()
        cascade = Cascade(calibrator=Calibrator(tau=0.3))
        for r in ["weather in Paris", "tell me a joke"]:
            tel.observe(cascade.route(r))
        snap = tel.snapshot()
        self.assertEqual(snap["requests_total"], 2)
        self.assertIn("functiongemma_requests_total", tel.render_prometheus())


if __name__ == "__main__":
    unittest.main()
