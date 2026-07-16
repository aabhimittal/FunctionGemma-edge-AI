"""Step 3 — The model backend.

A real FunctionGemma is a ~2B parameter Gemma checkpoint, fine-tuned so that
given the prompt from step 2 it emits a JSON function call. It is small enough
to run *on the edge* (a phone or laptop) with no cloud round-trip.

Downloading a 2B model would make this repo heavy and slow, so we ship a tiny
RULE-BASED stand-in that produces the *same JSON output contract*. This lets the
whole pipeline run instantly. `generate()` is the single seam where you would
swap in the real model — see `real_backend()` below for how.

A backend is any callable ``str -> str``: prompt in, raw completion out. That
one-line contract is what lets the cascade (functiongemma.cascade) treat the
edge mock, a real on-device Gemma, and a large cloud model interchangeably.
"""

import json
import re


def mock_backend(prompt):
    """A minimal deterministic 'model': keyword rules -> JSON call.

    It only reads the 'User: ...' line and maps intent to a tool call. This is
    NOT machine learning; it just mimics what the trained model would output so
    the surrounding code (prompt building, parsing, dispatch) is fully runnable.
    """
    request = prompt.split("User:")[-1].split("JSON:")[0].strip().lower()

    if "weather" in request:
        city = _guess_value(request, after=("in", "for")) or "London"
        return json.dumps({"name": "get_weather", "arguments": {"city": city.title()}})

    if "timer" in request or "remind" in request:
        minutes = _first_int(request) or 5
        return json.dumps({"name": "set_timer", "arguments": {"minutes": minutes}})

    if "message" in request or "text " in request or request.startswith("send"):
        to = _guess_value(request, after=("to",)) or "Mom"
        return json.dumps(
            {"name": "send_message", "arguments": {"to": to.title(), "body": "Hi!"}}
        )

    if "meeting" in request or "calendar" in request or "schedule" in request:
        hour = _first_int(request) or 9
        return json.dumps(
            {"name": "create_calendar_event",
             "arguments": {"title": "Meeting", "hour": hour}}
        )

    if "play" in request or "song" in request or "music" in request:
        query = _guess_value(request, after=("play", "by")) or "music"
        return json.dumps({"name": "play_music", "arguments": {"query": query}})

    # No confident match: the model is allowed to abstain.
    return json.dumps({"name": None, "arguments": {}})


def cloud_backend(prompt):
    """Stand-in for a *large* fallback model (the top of the cascade).

    In production this would be a hosted 27B+ model reached over the network.
    Here it is the same rule engine with a wider net and sensible defaults, so
    the cascade in `functiongemma.cascade` has a second tier to escalate to.
    The point of the demo is the *routing*, not the fallback's cleverness.
    """
    raw = mock_backend(prompt)
    call = json.loads(raw)
    if call["name"] is not None:
        return raw
    # A bigger model guesses rather than abstains on the leftover cases.
    request = prompt.split("User:")[-1].split("JSON:")[0].strip().lower()
    if "joke" in request or "translate" in request or "?" in request:
        # Still no tool fits -> a good large model also abstains. Honesty wins.
        return raw
    return raw


def real_backend(prompt, model_id="google/functiongemma-2b"):
    """How you'd plug in the actual model (kept as reference, not run in demo).

    Requires: pip install transformers torch, and a real checkpoint. For the
    confidence-gated cascade you would also return token log-probabilities so
    `functiongemma.confidence` can gate on the true model margin instead of a
    heuristic — see that module's docstring.
    """
    from transformers import pipeline  # imported lazily on purpose

    pipe = pipeline("text-generation", model=model_id)
    out = pipe(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
    return out[len(prompt):]  # keep only the newly generated JSON


def generate(prompt, backend=mock_backend):
    """Run a prompt through a backend. Default is the runnable mock."""
    return backend(prompt)


# --- tiny helpers for the mock (not part of the real model) -----------------

def _first_int(text):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def _guess_value(text, after):
    words = text.replace("?", "").split()
    for key in after:
        if key in words:
            i = words.index(key)
            if i + 1 < len(words):
                return words[i + 1]
    return None
