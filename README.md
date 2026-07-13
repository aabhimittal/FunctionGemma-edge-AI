# FunctionGemma-edge-AI

**How Google AI built a compact function-calling specialist for edge workloads — explained with the smallest possible runnable example.**

FunctionGemma is the idea of taking a *small* Gemma model (~2B params) and fine-tuning it to do exactly one job well: read a user request plus a list of available tools, and output **which function to call with which arguments**. Because it is small, it runs *on the edge* — a phone, a laptop, an IoT device — with no cloud round-trip.

This repo strips that idea down to a ~150-line pipeline you can read in five minutes and run with zero downloads.

```
tools  ->  prompt  ->  model  ->  parse  ->  call
```

---

## Why "compact" and "edge"?

| General chat LLM | FunctionGemma |
| --- | --- |
| Huge, runs in the cloud | ~2B params, runs on-device |
| Writes free-form prose | Emits one strict JSON call |
| Round-trip latency + privacy cost | Instant, local, private |
| Good at everything | *Specialist* at tool selection |

The trick is **specialization**: you don't need a 70B model to decide "the user said 'set a timer for 10 minutes' → call `set_timer(minutes=10)`". A small model, fine-tuned only on function-calling data, does it accurately and cheaply.

---

## The 5 steps (each is one small file)

### Step 1 — Describe the tools  ·  [`functiongemma/tools.py`](functiongemma/tools.py)
The model can't call anything it hasn't been told about. We give it a small catalogue of tools, each with a name, a description, and typed parameters.

### Step 2 — Build the prompt  ·  [`functiongemma/prompt.py`](functiongemma/prompt.py)
We compress the tools + the user's request into one tight prompt and set an **output contract**: *reply with one JSON object, nothing else*. A small context window forces this discipline.

### Step 3 — Run the model  ·  [`functiongemma/model.py`](functiongemma/model.py)
The real FunctionGemma turns that prompt into JSON. To keep the repo instant, we ship a tiny **rule-based mock** that produces the same JSON contract. `real_backend()` shows the one-line swap to a real Gemma checkpoint via 🤗 Transformers.

### Step 4 — Parse & validate  ·  [`functiongemma/parser.py`](functiongemma/parser.py)
On the edge there is no big server to clean up messy output. We extract the JSON, reject unknown tools, and type-check every argument. A model that isn't confident is allowed to **abstain** rather than guess.

### Step 5 — Call it  ·  [`demo.py`](demo.py)
Once you have a validated `{"name", "arguments"}`, your app dispatches it to real code. This demo just prints it.

---

## Run it

```bash
python demo.py                          # see the whole pipeline on 4 examples
python -m unittest discover -s tests    # run the tests
```

Sample output:

```
User: What's the weather in Paris?
Call:
{
  "name": "get_weather",
  "arguments": { "city": "Paris" }
}
...
User: Tell me a joke
No valid call: model abstained: no tool selected
```

## Plug in the real model

Swap the mock for an actual Gemma checkpoint — everything else stays identical:

```python
from functiongemma import run
from functiongemma.model import real_backend   # needs: pip install transformers torch

call = run("Set a timer for 10 minutes", backend=real_backend)
```

---

## What to take away

1. **Specialist > generalist** for narrow tasks — a fine-tuned 2B model beats a giant one on cost, speed, and privacy for tool selection.
2. **A strict output contract** (JSON only) makes a small model reliable.
3. **Validation is part of the model**, not an afterthought — reject bad calls, allow abstaining.
4. The model *chooses*; **your code executes**. That boundary is the whole design.

MIT licensed. Educational — the "model" here is a deterministic stand-in, not a trained network.
