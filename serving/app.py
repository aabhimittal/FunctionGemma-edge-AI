"""FastAPI inference server  —  the deployable edge endpoint.

Wraps the confidence-gated cascade behind a small HTTP API with the endpoints a
production service needs:

  * ``POST /v1/call``  — natural language in, routed function call out.
  * ``GET  /healthz``  — liveness/readiness for orchestrators.
  * ``GET  /health``   — rolling drift/health report + active alerts.
  * ``GET  /metrics``  — Prometheus exposition of live counters + latency.

Production hardening wired in here:
  * the cloud tier runs behind a **circuit breaker**, so a network outage
    degrades to local abstention instead of hanging every request;
  * the telemetry event log is **PII-redacted** at the boundary, because it
    contains verbatim user requests; and
  * a **HealthMonitor** tracks abstain/escalation/repair rates and confidence
    drift, surfaced at ``/health`` for a probe or dashboard.

On startup it loads the calibrated abstention threshold if one has been fitted,
so the deployed gate matches the one that was evaluated.

Run:  uvicorn serving.app:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

import os

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from functiongemma.cascade import Cascade
from functiongemma.confidence import Calibrator
from functiongemma.config import load_config
from functiongemma.monitor import HealthMonitor
from functiongemma.privacy import default_redactor
from functiongemma.reliability import CircuitBreaker
from functiongemma.telemetry import Telemetry
from serving.schemas import CallRequest, CallResponse

cfg = load_config(os.environ.get("FG_CONFIG"))

# Load a fitted calibrator if present, else fall back to the config default.
if os.path.exists("calibrator.json"):
    calibrator = Calibrator.load("calibrator.json")
else:
    calibrator = Calibrator(tau=cfg.tau, target_accuracy=cfg.target_accuracy)

cascade = Cascade(calibrator=calibrator, cloud_breaker=CircuitBreaker())
telemetry = Telemetry(event_log_path=cfg.event_log_path, redactor=default_redactor)
monitor = HealthMonitor()

app = FastAPI(
    title="FunctionGemma Edge",
    version="0.4.0",
    description="Compact, confidence-gated function calling for the edge.",
)


@app.get("/healthz")
def healthz():
    """Liveness + the currently active abstention threshold + breaker state."""
    return {
        "status": "ok",
        "tau": cascade.calibrator.tau,
        "cloud_breaker": cascade.cloud_breaker.state,
    }


@app.post("/v1/call", response_model=CallResponse)
def call(req: CallRequest):
    """Route one request through the cascade and record telemetry + health."""
    decision = cascade.route(req.request)
    telemetry.observe(decision)
    monitor.observe(decision)
    return decision.to_dict()


@app.get("/health")
def health():
    """Rolling drift/health report and any active alerts."""
    return {"report": monitor.report(), "alerts": monitor.alerts()}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text exposition of live service metrics."""
    return telemetry.render_prometheus()
