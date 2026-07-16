"""Tests for the FastAPI serving layer. Skipped if serving deps aren't installed."""

import unittest

try:
    from fastapi.testclient import TestClient

    from serving.app import app

    _CLIENT = TestClient(app)
    _HAVE_SERVING = True
except ImportError:  # pragma: no cover - serving extras not installed
    _HAVE_SERVING = False


@unittest.skipUnless(_HAVE_SERVING, "serving extras (fastapi) not installed")
class TestServing(unittest.TestCase):
    def test_healthz(self):
        resp = _CLIENT.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_call_returns_edge_decision(self):
        resp = _CLIENT.post("/v1/call", json={"request": "weather in Paris"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn(body["tier"], ("edge", "cloud", "abstain"))
        if body["call"]:
            self.assertIn("name", body["call"])

    def test_metrics_exposition(self):
        _CLIENT.post("/v1/call", json={"request": "set a timer for 5 minutes"})
        resp = _CLIENT.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("functiongemma_requests_total", resp.text)


if __name__ == "__main__":
    unittest.main()
