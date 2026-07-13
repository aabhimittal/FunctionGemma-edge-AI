"""Step 3 — The model backend.

A real FunctionGemma is a ~2B parameter Gemma checkpoint, fine-tuned so that
given the prompt from step 2 it emits a JSON function call. It is small enough
to run *on the edge* (a phone or laptop) with no cloud round-trip.

Downloading a 2B model would make this repo heavy and slow, so we ship a tiny
RULE-BASED stand-in that produces the *same JSON output contract*. This lets the
whole pipeline run instantly. `generate()` is the single seam where you would
swap in the real model — see `real_backend()` below for how.
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
        # naive city grab: last capitalized-ish word in the original request
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

    # No confident match: the model is allowed to abstain.
    return json.dumps({"name": None, "arguments": {}})


def real_backend(prompt, model_id="google/functiongemma-2b"):
    """How you'd plug in the actual model (kept as reference, not run in demo).

    Requires: pip install transformers torch, and a real checkpoint.
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
