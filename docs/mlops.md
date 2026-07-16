# The MLops loop

This project is small, but it wires together the same lifecycle a production
edge model goes through. Everything here runs offline with `make pipeline`.

```
generate_data  ->  train  ->  evaluate  ->  promote  ->  (quantize)  ->  serve
      ^                          |                                         |
      |                          v                                         v
      +-------- hard cases <-- telemetry event log <----------------------+
```

## 1. Data — `pipelines/generate_data.py`
Templated synthesis with slot filling produces clean, fully-labelled
`(request -> call)` pairs, plus **hard negatives** (chit-chat whose target is an
abstention). The negatives are what teach the model to abstain instead of
guessing. For higher quality, generate the `completion` field with a large
teacher model and distil — the downstream code is unchanged.

## 2. Train — `pipelines/train.py`
LoRA fine-tune of a small Gemma base. Only a few million low-rank parameters are
adapted, so the specialist is cheap to train and ships as a few MB of adapter
weights. Runs behind lazy imports; `--dry-run` validates data and prints the
plan with no dependencies. Real training needs `requirements-ml.txt` and a GPU.

## 3. Evaluate — `pipelines/evaluate.py`
Scores a backend on the golden set and reports the metrics that matter for tool
calling: **tool accuracy**, **argument accuracy**, **exact match**, and
**abstention precision/recall**. The same run fits the abstention
`Calibrator`, so the shipped threshold is the one that was measured.

## 4. Promote — `pipelines/promote.py` + `functiongemma/registry.py`
The single quality gate. A candidate is registered as an immutable version and
only moved to the **production** pointer if it clears `min_tool_accuracy`.
Non-zero exit on failure, so CI can block a bad model. Rollback = re-point the
stage at a previous version.

## 5. Quantize — `pipelines/quantize.py`
Edge export to int4 (GGUF, phones/laptops) or int8 (ONNX, cross-platform). The
footprint budget table makes the size/latency trade-off explicit before you
spend a GPU on conversion.

## 6. Serve — `serving/app.py` + `functiongemma/telemetry.py`
FastAPI endpoint wrapping the cascade, with `/healthz` and a Prometheus
`/metrics` endpoint. Every request is written to a structured JSONL event log.

## 7. Close the loop
Replay the event log to mine escalated/abstained requests; feed them back into
step 1. Production experience becomes next iteration's training data.

## Running it

```bash
make pipeline     # data -> eval -> gate -> promote (offline, no model download)
make serve        # start the FastAPI edge server
make docker       # build the serving container
```

## What is real vs illustrative

| Real / production-shaped | Illustrative stand-in |
| --- | --- |
| Pipeline structure, gate, registry, telemetry, serving API | The "model" (a deterministic mock, not a trained net) |
| LoRA/quantize scripts (industry-standard libraries) | Footprint numbers (order-of-magnitude, not benchmarked) |
| Calibration objective (selective prediction) | Confidence proxy (lexical overlap in place of logits) |

Swap the mock backend for a real Gemma checkpoint and none of the surrounding
structure changes — that separation is the design.
