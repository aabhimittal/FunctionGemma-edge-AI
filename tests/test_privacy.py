"""Tests for PII redaction of telemetry."""

import json
import os
import tempfile
import unittest

from functiongemma.cascade import Decision
from functiongemma.privacy import Redactor, default_redactor
from functiongemma.telemetry import Telemetry


class TestRedactor(unittest.TestCase):
    def setUp(self):
        self.r = Redactor()

    def test_email_redacted(self):
        self.assertEqual(self.r.redact_text("mail me at bob@acme.com now"),
                         "mail me at [EMAIL] now")

    def test_phone_redacted(self):
        out = self.r.redact_text("call +1 415 555 2671 please")
        self.assertIn("[PHONE]", out)
        self.assertNotIn("2671", out)

    def test_credit_card_redacted(self):
        out = self.r.redact_text("card 4111 1111 1111 1111")
        self.assertIn("[CARD]", out)

    def test_non_pii_is_untouched(self):
        self.assertEqual(self.r.redact_text("weather in Paris"), "weather in Paris")

    def test_idempotent(self):
        once = self.r.redact_text("bob@acme.com")
        twice = self.r.redact_text(once)
        self.assertEqual(once, twice)

    def test_redact_call_does_not_mutate_original(self):
        call = {"name": "send_message", "arguments": {"to": "bob@acme.com", "body": "hi"}}
        red = self.r.redact_call(call)
        self.assertEqual(call["arguments"]["to"], "bob@acme.com")  # original intact
        self.assertEqual(red["arguments"]["to"], "[EMAIL]")

    def test_integer_arguments_survive(self):
        call = {"name": "set_timer", "arguments": {"minutes": 10}}
        self.assertEqual(self.r.redact_call(call)["arguments"]["minutes"], 10)


class TestTelemetryRedaction(unittest.TestCase):
    def test_event_log_is_scrubbed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            tel = Telemetry(event_log_path=path, redactor=default_redactor)
            decision = Decision(
                request="text bob@acme.com my card 4111 1111 1111 1111",
                tier="edge",
                call={"name": "send_message",
                      "arguments": {"to": "bob@acme.com", "body": "hi"}},
                confidence=0.9,
            )
            tel.observe(decision)
            with open(path, encoding="utf-8") as fh:
                logged = json.loads(fh.readline())
        self.assertNotIn("bob@acme.com", json.dumps(logged))
        self.assertNotIn("4111", json.dumps(logged))
        self.assertEqual(logged["call"]["arguments"]["to"], "[EMAIL]")


if __name__ == "__main__":
    unittest.main()
