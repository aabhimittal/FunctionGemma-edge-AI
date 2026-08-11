"""FunctionGemma-edge-AI: a minimal, runnable walk-through of how a compact
function-calling model works end to end — plus the MLops scaffolding that turns
it into a shippable edge system.

Core pipeline (the 5 minimal files):
    tools  ->  prompt  ->  model  ->  parse  ->  call

Novel layer (confidence-gated edge->cloud cascade):
    constrained decoding  +  calibrated abstention  +  cascade routing

MLops layer:
    registry (versioning)  ·  telemetry (logs+metrics)  ·  config

Runtime layer (request lifecycle on the device):
    guard (policy+sanitization)  ·  planner (multi-intent)  ·  executor (agent)

Conversation layer (multi-turn on the device):
    session (slot filling + follow-ups)  ·  idempotency (exactly-once effects)

Scale & operations layer (production hardening):
    retrieval (large catalogues)  ·  budget (adaptive cost control)  ·
    reliability (circuit breaker)  ·  monitor (drift)  ·  privacy (PII redaction)  ·
    shadow (canary rollout)
"""

from .budget import EscalationBudgetController
from .cascade import Cascade, Decision, coverage_report
from .confidence import Calibrator, score
from .config import Config, load_config
from .constrained import constrain
from .executor import IMPLEMENTATIONS, Agent, execute
from .guard import Guardrail, Verdict, screen_request
from .idempotency import IdempotencyCache, fingerprint
from .model import cloud_backend, generate, mock_backend
from .monitor import HealthMonitor, psi
from .parser import InvalidCall, parse, validate
from .planner import Plan, Planner
from .privacy import Redactor, default_redactor
from .prompt import build_prompt, pretty
from .registry import ModelRegistry
from .reliability import CircuitBreaker, CircuitOpen, guarded_call
from .retrieval import ToolRetriever
from .session import Pending, Session
from .shadow import ShadowRunner, calls_agree
from .telemetry import Telemetry
from .tools import TOOLS, get_tool, load_tools

__version__ = "0.5.0"


def run(user_request, backend=mock_backend):
    """Full *core* pipeline: text in -> validated function call out.

    This is the simple, single-model path (no cascade). For the confidence-gated
    edge->cloud routing use :class:`Cascade` instead.
    """
    prompt = build_prompt(TOOLS, user_request)
    raw = generate(prompt, backend=backend)
    return parse(raw)


__all__ = [
    "TOOLS",
    "get_tool",
    "load_tools",
    "build_prompt",
    "pretty",
    "generate",
    "mock_backend",
    "cloud_backend",
    "parse",
    "validate",
    "InvalidCall",
    "score",
    "Calibrator",
    "constrain",
    "Cascade",
    "Decision",
    "coverage_report",
    "ModelRegistry",
    "Telemetry",
    "Config",
    "load_config",
    "Guardrail",
    "Verdict",
    "screen_request",
    "Planner",
    "Plan",
    "Agent",
    "execute",
    "IMPLEMENTATIONS",
    "ToolRetriever",
    "EscalationBudgetController",
    "CircuitBreaker",
    "CircuitOpen",
    "guarded_call",
    "HealthMonitor",
    "psi",
    "Redactor",
    "default_redactor",
    "Session",
    "Pending",
    "IdempotencyCache",
    "fingerprint",
    "ShadowRunner",
    "calls_agree",
    "run",
]
