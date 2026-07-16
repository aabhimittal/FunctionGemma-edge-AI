# Model Card — FunctionGemma-edge

> This card documents the *intended* model and pipeline. The checkpoint shipped
> in this repository is a deterministic mock backend for education and testing,
> not a trained network — see "Limitations".

## Model details
- **Purpose:** a compact function-calling specialist that maps a user request +
  a tool catalogue to a single validated JSON call, or abstains.
- **Intended base:** a small Gemma checkpoint (~2B) fine-tuned with LoRA.
- **Serving form:** int4 (GGUF) or int8 (ONNX) for on-device inference.
- **Deployment:** the confidence-gated edge→cloud cascade in `functiongemma.cascade`.

## Intended use
- On-device tool selection for assistants (timers, messages, weather, calendar,
  media) where latency, privacy, and offline operation matter.
- **Out of scope:** free-form chat, reasoning, code generation, or any tool not
  in the catalogue. The model is a *specialist*, not a general assistant.

## Inputs and outputs
- **Input:** natural-language request + the tool catalogue (`data/tools_catalog.json`).
- **Output:** `{"name": <tool>, "arguments": {...}}`, or an abstention
  (`name: null`) when no tool applies or confidence is below the calibrated
  threshold.

## Evaluation
- Metrics (`pipelines/evaluate.py`): tool accuracy, argument accuracy, exact
  match, abstention precision/recall.
- Promotion gate: a candidate reaches production only if tool accuracy ≥
  `min_tool_accuracy` (`configs/default.yaml`).
- Abstention is calibrated with a selective-prediction objective: maximise
  coverage subject to a target selective accuracy.

## Safety & reliability design
- **Constrained decoding** projects raw output onto the tool schema, so unknown
  tools and mistyped arguments are repaired or rejected rather than executed.
- **The model chooses; your code executes.** The library never runs a tool — it
  returns a validated call for the host application to dispatch.
- **Calibrated abstention + cascade** let the edge model defer instead of
  guessing on low-confidence inputs.

## Limitations
- The shipped backend is a rule-based mock; accuracy numbers from the mock are
  not representative of a trained model.
- The confidence signal in the mock is a lexical proxy; a real deployment should
  gate on model log-probabilities (`confidence.score(..., model_logprob=...)`).
- Footprint/latency figures in `pipelines/quantize.py` are illustrative.

## Ethical considerations
- Keep the tool catalogue minimal and audited: the model can only ever emit a
  call to a tool you have explicitly listed.
- Log responsibly — the telemetry event log contains user requests; treat it as
  sensitive and apply your own retention/PII policy.
