"""FastAPI inference server  —  the deployable edge endpoint.

Wraps the confidence-gated cascade behind a small HTTP API with the three
endpoints any production service needs:

  * ``POST /v1/call``  — natural language in, routed function call out.
  * ``GET  /healthz``  — liveness/readiness for orchestrators.
  * ``GET  /metrics``  — Prometheus exposition of live counters + latency.

Every request is recorded through :class:`~functiongemma.telemetry.Telemetry`,
so the same server that answers also produces the logs and metrics the rest of
the MLops loop consumes. On startup it loads the calibrated abstention threshold
if one has been fitted, so the deployed gate matches the one that was evaluated.

Run:  uvicorn serving.app:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

import os

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from functiongemma.cascade import Cascade
from functiongemma.confidence import Calibrator
from functiongemma.config import load_config
from functiongemma.telemetry import Telemetry
from serving.schemas import CallRequest, CallResponse

cfg = load_config(os.environ.get("FG_CONFIG"))

# Load a fitted calibrator if present, else fall back to the config default.
if os.path.exists("calibrator.json"):
    calibrator = Calibrator.load("calibrator.json")
else:
    calibrator = Calibrator(tau=cfg.tau, target_accuracy=cfg.target_accuracy)

cascade = Cascade(calibrator=calibrator)
telemetry = Telemetry(event_log_path=cfg.event_log_path)

app = FastAPI(
    title="FunctionGemma Edge",
    version="0.2.0",
    description="Compact, confidence-gated function calling for the edge.",
)


@app.get("/healthz")
def healthz():
    """Liveness + the currently active abstention threshold."""
    return {"status": "ok", "tau": cascade.calibrator.tau}


@app.post("/v1/call", response_model=CallResponse)
def call(req: CallRequest):
    """Route one request through the cascade and record telemetry."""
    decision = cascade.route(req.request)
    telemetry.observe(decision)
    return decision.to_dict()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text exposition of live service metrics."""
    return telemetry.render_prometheus()
