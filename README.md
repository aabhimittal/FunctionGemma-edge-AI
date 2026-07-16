# FunctionGemma-edge-AI

**How Google AI built a compact function-calling specialist for edge workloads —
rebuilt as a minimal, runnable project with a real MLops loop around it.**

FunctionGemma is the idea of taking a *small* Gemma model (~2B params) and
fine-tuning it to do exactly one job well: read a user request plus a list of
available tools, and output **which function to call with which arguments**.
Because it is small, it runs *on the edge* — a phone, a laptop, an IoT device —
with no cloud round-trip.

This repo does two things at once:

1. **Explains the idea** with a ~150-line core pipeline you can read in five
   minutes and run with zero downloads.
2. **Turns it into a system** — a novel *confidence-gated edge→cloud cascade* on
   top, and a tried-and-tested *MLops loop* (data → train → eval-gate → registry
   → quantize → serve → monitor) around it.

```
tools ─▶ prompt ─▶ model ─▶ parse ─▶ [constrain ─▶ confidence gate ─▶ cascade] ─▶ call
└──────────────── minimal core ─────────────┘ └──────────── novel serving layer ────────────┘
```

> The "model" here is a deterministic mock so everything runs instantly and in
> CI. `real_backend()` shows the one-line swap to an actual Gemma checkpoint —
> **nothing else in the pipeline changes.** That separation is the whole point.

---

## Quickstart

```bash
git clone <this repo> && cd FunctionGemma-edge-AI

python demo.py                        # core pipeline + cascade, no installs
pip install -r requirements-dev.txt   # for tests + serving
python -m pytest -q

make pipeline                         # full offline MLops loop
make serve                            # FastAPI edge server at :8000/docs
```

Sample demo output:

```
[   edge] conf=0.80  'Set a timer for 10 minutes' -> set_timer
[   edge] conf=0.65  "What's the weather in Paris?" -> get_weather
[abstain] conf=0.00  'Tell me a joke' -> —
coverage report: {'coverage_edge': 0.83, 'escalation_rate': 0.0, 'abstain_rate': 0.17, ...}
```

---

## Why "compact" and "edge"?

| General chat LLM | FunctionGemma |
| --- | --- |
| Huge, runs in the cloud | ~2B params, runs on-device |
| Writes free-form prose | Emits one strict JSON call |
| Round-trip latency + privacy cost | Instant, local, private |
| Good at everything | *Specialist* at tool selection |

You don't need a 70B model to decide *"set a timer for 10 minutes" →
`set_timer(minutes=10)`*. A small model, fine-tuned only on function-calling
data, does it accurately and cheaply. The trick is **specialization**.

---

## Part 1 — The minimal core (5 files)

Each step is one small, dependency-free file.

| # | Step | File | What it does |
|---|------|------|--------------|
| 1 | Describe the tools | [`functiongemma/tools.py`](functiongemma/tools.py) | a typed catalogue the model may call, loaded from [`data/tools_catalog.json`](data/tools_catalog.json) |
| 2 | Build the prompt | [`functiongemma/prompt.py`](functiongemma/prompt.py) | compress tools + request into one prompt with a strict *JSON-only* output contract |
| 3 | Run the model | [`functiongemma/model.py`](functiongemma/model.py) | the backend seam: mock now, real Gemma with a one-line swap |
| 4 | Parse & validate | [`functiongemma/parser.py`](functiongemma/parser.py) | extract JSON, reject unknown tools, type-check args, allow abstaining |
| 5 | Call it | [`demo.py`](demo.py) | dispatch the validated `{name, arguments}` to real code |

```python
from functiongemma import run
run("What's the weather in Paris?")
# -> {'name': 'get_weather', 'arguments': {'city': 'Paris'}}
```

### Plug in the real model

```python
from functiongemma import run
from functiongemma.model import real_backend      # pip install transformers torch
run("Set a timer for 10 minutes", backend=real_backend)
```

---

## Part 2 — The novel bit: a confidence-gated edge→cloud cascade

A compact model is *right most of the time and wrong on a hard tail*. The
interesting question is not "is it right?" but **"does it know when it isn't?"**
If it does, it can answer the easy 80–90% locally and hand the rest to a bigger
model — instead of guessing. Three ideas make that work:

**1. Schema-constrained decoding** · [`constrained.py`](functiongemma/constrained.py)
Projects raw model text onto the tool grammar: fuzzy-fixes a misspelled tool
name, coerces `"10" → 10`, drops hallucinated arguments — and leaves genuine
failures for the parser to reject. Reduces the reject rate without hiding errors.

**2. Calibrated abstention** · [`confidence.py`](functiongemma/confidence.py)
A confidence signal (real: token log-probs; mock: a lexical proxy) plus a
`Calibrator` that fits a single threshold `τ` on a validation set using the
standard *selective-prediction* objective — **maximise coverage subject to a
target selective accuracy**.

**3. The cascade** · [`cascade.py`](functiongemma/cascade.py)
Edge model tries first; if confidence ≥ τ it answers locally, else it escalates
to a larger model. Every decision logs the three dials you actually operate on:
**coverage** (answered on-device), **escalation rate** (cost), and **selective
accuracy** (quality floor).

```python
from functiongemma import Cascade, Calibrator
cascade = Cascade(calibrator=Calibrator(tau=0.55))
d = cascade.route("set a timer for 10 minutes")
d.tier, d.call, d.confidence        # ('edge', {'name': 'set_timer', ...}, 0.8)
```

---

## Part 3 — The MLops loop (industrial, tried-and-tested)

```
generate_data ─▶ train(LoRA) ─▶ evaluate(gate) ─▶ promote ─▶ registry ─▶ quantize ─▶ serve
      ▲                              │                                                  │
      └──────── hard cases ◀── telemetry event log ◀───────────────────────────────────┘
```

| Stage | File | Role |
|-------|------|------|
| Data | [`pipelines/generate_data.py`](pipelines/generate_data.py) | synthetic FC data + hard negatives (teaches abstention) |
| Train | [`pipelines/train.py`](pipelines/train.py) | LoRA fine-tune of Gemma (lazy deps; `--dry-run` needs none) |
| Evaluate | [`pipelines/evaluate.py`](pipelines/evaluate.py) | tool/arg/exact-match + abstention metrics; fits the calibrator |
| Promote | [`pipelines/promote.py`](pipelines/promote.py) | the quality **gate** — production only if it clears the bar |
| Registry | [`functiongemma/registry.py`](functiongemma/registry.py) | immutable versions + promotable stage pointers (rollback) |
| Quantize | [`pipelines/quantize.py`](pipelines/quantize.py) | int4 (GGUF) / int8 (ONNX) edge export + footprint budget |
| Serve | [`serving/app.py`](serving/app.py) | FastAPI `/v1/call`, `/healthz`, Prometheus `/metrics` |
| Monitor | [`functiongemma/telemetry.py`](functiongemma/telemetry.py) | structured event log + live counters/latency histogram |

Full walkthrough: [`docs/mlops.md`](docs/mlops.md) · design: [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`model_card.md`](model_card.md).

### The serving API

```bash
make serve
curl -s localhost:8000/v1/call -H 'content-type: application/json' \
     -d '{"request":"weather in Tokyo"}'
# {"tier":"edge","call":{"name":"get_weather","arguments":{"city":"Tokyo"}},"confidence":0.65,...}
curl -s localhost:8000/metrics        # Prometheus exposition
```

`docker compose up` brings up the server plus a Prometheus that scrapes it.

---

## Project layout

```
functiongemma/     core (tools·prompt·model·parser) + novel (constrained·confidence·cascade)
                   + mlops helpers (registry·telemetry·config)
pipelines/         generate_data · train · evaluate · quantize · promote
serving/           FastAPI app + schemas
data/              tools_catalog.json · eval/golden.jsonl
configs/           default.yaml · train_lora.yaml · prometheus.yml
tests/             one focused test file per module
benchmarks/        cascade routing latency
.github/workflows/ ci.yml (lint·test·eval-gate) · release.yml (container)
```

---

## What to take away

1. **Specialist > generalist** for narrow tasks — a fine-tuned 2B model beats a
   giant one on cost, speed, and privacy for tool selection.
2. **A strict output contract + constrained decoding** make a small model reliable.
3. **Calibrated abstention** turns "sometimes wrong" into "knows when to defer".
4. **A cascade** buys generalist accuracy at specialist cost.
5. **The model chooses; your code executes** — and an MLops loop keeps the model
   that reaches production one that cleared the gate.

MIT licensed. Educational — the "model" is a deterministic stand-in, not a
trained network; every other layer is production-shaped.
