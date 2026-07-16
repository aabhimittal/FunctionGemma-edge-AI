# Architecture

FunctionGemma-edge-AI is deliberately split into three layers you can read
independently: a **minimal core**, a **novel serving layer**, and an
**industrial MLops loop** around them. Each box below maps to one small module.

## The three layers

```mermaid
flowchart TB
    subgraph CORE["Core pipeline (minimal, stdlib)"]
        T["tools.py<br/>tool catalogue"] --> P["prompt.py<br/>build prompt"]
        P --> M["model.py<br/>backend (mock / Gemma)"]
        M --> PA["parser.py<br/>validate call"]
    end

    subgraph NOVEL["Novel layer: confidence-gated cascade"]
        C["constrained.py<br/>schema repair"]
        CF["confidence.py<br/>score + Calibrator"]
        CA["cascade.py<br/>edge → cloud routing"]
    end

    subgraph MLOPS["MLops loop"]
        GD["generate_data.py"] --> TR["train.py (LoRA)"]
        TR --> EV["evaluate.py (gate)"]
        EV --> PR["promote.py"]
        PR --> RG["registry.py"]
        QZ["quantize.py<br/>edge export"]
        SV["serving/app.py<br/>FastAPI"]
        TM["telemetry.py<br/>logs + /metrics"]
    end

    PA --> C --> CF --> CA
    CA --> SV
    SV --> TM
    RG --> SV
    TM -. hard cases .-> GD
```

## Request path at serving time

```mermaid
sequenceDiagram
    participant U as Client
    participant S as FastAPI /v1/call
    participant E as Edge model (2B)
    participant K as Constrained decode
    participant G as Calibrated gate
    participant Cl as Cloud model
    U->>S: "set a timer for 10 minutes"
    S->>E: prompt
    E-->>K: raw JSON-ish text
    K-->>G: repaired, in-schema call + confidence
    alt confidence >= tau
        G-->>S: answer locally (tier=edge)
    else confidence < tau
        G->>Cl: escalate
        Cl-->>S: answer (tier=cloud)
    end
    S-->>U: {tier, call, confidence, latency_ms}
    Note over S: every request recorded via telemetry
```

## Why this shape

| Concern | Where it lives | Design choice |
| --- | --- | --- |
| What can be called | `tools.py` + `data/tools_catalog.json` | one JSON contract shared by data, eval, serving |
| Making a small model reliable | `prompt.py`, `constrained.py` | strict output contract + decode-time schema projection |
| Knowing when to defer | `confidence.py`, `cascade.py` | calibrated abstention + edge→cloud cascade |
| Shipping safely | `evaluate.py`, `promote.py`, `registry.py` | eval gate guards the production pointer |
| Fitting on-device | `quantize.py` | int4/int8 export, footprint budget |
| Operating it | `serving/app.py`, `telemetry.py` | health, Prometheus metrics, event log |

## The feedback loop

The event log written by `telemetry.py` is not just for dashboards: replaying it
surfaces the requests that escalated or abstained — exactly the hard cases worth
adding to the next `generate_data.py` run (or labelling with a teacher model for
distillation). That closes the loop from **production → data → training →
evaluation → production**, which is the whole point of an MLops setup.
