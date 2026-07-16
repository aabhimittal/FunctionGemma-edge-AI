"""Synthetic training-data generator  —  the data-engineering step.

You cannot fine-tune a function-calling specialist without function-calling
data, and hand-labelling thousands of (request -> call) pairs is expensive. The
industrial shortcut is *templated synthesis with slot filling*: enumerate
natural phrasings per tool, fill typed slots from small value banks, and pair
each utterance with its exact target call. Cheap, fully labelled, and — because
the label is generated *with* the utterance — perfectly clean.

We also mint **hard negatives**: chit-chat with no matching tool, whose target
is an abstention (``{"name": null}``). Training on these is what teaches the
model to abstain instead of forcing a wrong call — the behaviour the cascade
relies on. A real pipeline would additionally distil from a large teacher model;
the seam for that is noted in ``pipelines/train.py``.

Output: JSONL of ``{"request", "completion"}`` where completion is the exact
JSON the model should learn to emit. Deterministic given ``--seed``.

Usage:
    python -m pipelines.generate_data --n 500 --out data/train.jsonl
"""

import argparse
import json
import random

# Phrase templates per tool; {slot} placeholders are filled from the banks below.
TEMPLATES = {
    "get_weather": [
        "what's the weather in {city}?",
        "tell me the weather in {city}",
        "is it raining in {city}?",
        "weather for {city}",
    ],
    "set_timer": [
        "set a timer for {minutes} minutes",
        "remind me in {minutes} minutes",
        "start a {minutes} minute timer",
    ],
    "send_message": [
        "send a message to {to}",
        "text {to}",
        "message {to} that I'll be late",
    ],
    "create_calendar_event": [
        "schedule a meeting at {hour}",
        "put a meeting on my calendar at {hour}",
        "book time at {hour}",
    ],
    "play_music": [
        "play {query}",
        "put on some {query}",
        "play {query} for me",
    ],
}

NEGATIVES = [
    "tell me a joke",
    "what is the meaning of life?",
    "how are you today?",
    "do you like pizza?",
    "sing me a song about the sea",
    "explain quantum physics",
]

BANKS = {
    "city": ["Paris", "Tokyo", "Berlin", "Cairo", "Lima", "Oslo"],
    "minutes": [1, 3, 5, 10, 15, 20, 30, 45, 60],
    "to": ["Alex", "Sam", "Mom", "Dad", "Priya", "Chen"],
    "hour": list(range(1, 24)),
    "query": ["jazz", "radiohead", "lo-fi", "the beatles", "classical"],
}


def _fill(slot, rng):
    return rng.choice(BANKS[slot])


def _example(tool, template, rng):
    """Build one (request, completion) pair from a template."""
    slots = {s for s in BANKS if "{" + s + "}" in template}
    values = {s: _fill(s, rng) for s in slots}
    request = template.format(**values)
    # The completion is the exact call, with slot values typed correctly.
    arguments = {s: values[s] for s in slots}
    completion = {"name": tool, "arguments": arguments}
    return {"request": request, "completion": json.dumps(completion)}


def generate(n, seed=0, negative_ratio=0.2):
    """Yield ``n`` synthetic examples with a fraction of hard negatives."""
    rng = random.Random(seed)
    tools = list(TEMPLATES)
    for _ in range(n):
        if rng.random() < negative_ratio:
            request = rng.choice(NEGATIVES)
            yield {
                "request": request,
                "completion": json.dumps({"name": None, "arguments": {}}),
            }
        else:
            tool = rng.choice(tools)
            template = rng.choice(TEMPLATES[tool])
            yield _example(tool, template, rng)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic FC training data.")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--negative-ratio", type=float, default=0.2)
    ap.add_argument("--out", default="data/train.jsonl")
    args = ap.parse_args()

    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in generate(args.n, args.seed, args.negative_ratio):
            fh.write(json.dumps(ex) + "\n")
    print(f"wrote {args.n} examples -> {args.out}")


if __name__ == "__main__":
    main()
