"""Tests for exactly-once side effects."""

import unittest

from functiongemma.executor import IMPLEMENTATIONS, Agent
from functiongemma.idempotency import IdempotencyCache, fingerprint

SEND = {"name": "send_message", "arguments": {"to": "Sam", "body": "hi"}}
OK = {"ok": True, "tool": "send_message", "result": {"delivered_to": "Sam"}}


class TestFingerprint(unittest.TestCase):
    def test_argument_order_does_not_matter(self):
        a = {"name": "send_message", "arguments": {"to": "Sam", "body": "hi"}}
        b = {"name": "send_message", "arguments": {"body": "hi", "to": "Sam"}}
        self.assertEqual(fingerprint(a), fingerprint(b))

    def test_different_arguments_differ(self):
        other = {"name": "send_message", "arguments": {"to": "Alex", "body": "hi"}}
        self.assertNotEqual(fingerprint(SEND), fingerprint(other))

    def test_unicode_is_stable(self):
        call = {"name": "send_message", "arguments": {"to": "Zürich", "body": "héllo"}}
        self.assertEqual(fingerprint(call), fingerprint(dict(call)))


class TestCache(unittest.TestCase):
    def test_hit_within_window(self):
        cache = IdempotencyCache(ttl_s=60)
        cache.put(SEND, OK)
        self.assertEqual(cache.get(SEND), OK)

    def test_miss_for_different_call(self):
        cache = IdempotencyCache(ttl_s=60)
        cache.put(SEND, OK)
        other = {"name": "send_message", "arguments": {"to": "Alex", "body": "hi"}}
        self.assertIsNone(cache.get(other))

    def test_entry_expires(self):
        clock = [0.0]
        cache = IdempotencyCache(ttl_s=10, _clock=lambda: clock[0])
        cache.put(SEND, OK)
        clock[0] += 11
        self.assertIsNone(cache.get(SEND))  # the same message is sendable again.

    def test_failures_are_not_cached(self):
        cache = IdempotencyCache(ttl_s=60)
        cache.put(SEND, {"ok": False, "tool": "send_message", "error": "boom"})
        self.assertIsNone(cache.get(SEND))  # a failed send must stay retryable.

    def test_capacity_is_bounded(self):
        cache = IdempotencyCache(ttl_s=600, max_entries=8)
        for i in range(50):
            cache.put({"name": "send_message", "arguments": {"to": f"u{i}"}}, OK)
        self.assertLessEqual(len(cache._entries), 8)


class TestAgentIntegration(unittest.TestCase):
    """The effect must happen once even when the request repeats."""

    def _counting_agent(self, cache):
        sent = []
        impls = dict(IMPLEMENTATIONS)
        impls["send_message"] = lambda to, body: sent.append(to) or {"to": to}
        impls["get_weather"] = lambda city: {"city": city, "n": len(sent)}
        return Agent(implementations=impls, idempotency=cache), sent

    def test_double_tap_sends_once(self):
        agent, sent = self._counting_agent(IdempotencyCache(ttl_s=60))
        first = agent.handle("send a message to Sam")
        second = agent.handle("send a message to Sam")
        self.assertEqual(len(sent), 1)                       # effect once
        self.assertEqual(second["steps"][0]["status"], "executed")
        self.assertTrue(second["steps"][0]["duplicate"])     # and it says so
        self.assertNotIn("duplicate", first["steps"][0])

    def test_read_only_tool_is_not_suppressed(self):
        agent, _ = self._counting_agent(IdempotencyCache(ttl_s=60))
        for _ in range(3):
            report = agent.handle("weather in Paris")
        self.assertNotIn("duplicate", report["steps"][0])  # weather stays fresh

    def test_distinct_recipients_both_send(self):
        agent, sent = self._counting_agent(IdempotencyCache(ttl_s=60))
        agent.handle("send a message to Sam")
        agent.handle("send a message to Alex")
        self.assertEqual(len(sent), 2)

    def test_without_cache_the_effect_repeats(self):
        # Confirms the suppression is the cache's doing, not an accident.
        agent, sent = self._counting_agent(None)
        agent.handle("send a message to Sam")
        agent.handle("send a message to Sam")
        self.assertEqual(len(sent), 2)

    def test_failed_send_is_retried(self):
        calls = {"n": 0}

        def flaky(to, body):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("radio off")
            return {"to": to}

        impls = dict(IMPLEMENTATIONS)
        impls["send_message"] = flaky
        agent = Agent(implementations=impls, idempotency=IdempotencyCache(ttl_s=60))
        first = agent.handle("send a message to Sam")
        second = agent.handle("send a message to Sam")
        self.assertEqual(first["steps"][0]["status"], "failed")
        self.assertEqual(second["steps"][0]["status"], "executed")
        self.assertEqual(calls["n"], 2)  # the retry really re-ran


if __name__ == "__main__":
    unittest.main()
