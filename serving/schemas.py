"""Request/response contracts for the serving layer.

Pydantic models give us validation, OpenAPI docs and typed responses for free —
the same output contract the model is trained to, now enforced at the API edge.
"""

from typing import Any

from pydantic import BaseModel, Field


class CallRequest(BaseModel):
    request: str = Field(..., description="The natural-language user request.")


class CallResponse(BaseModel):
    tier: str = Field(..., description="Which tier answered: edge | cloud | abstain.")
    call: dict[str, Any] | None = Field(
        None, description="The chosen {name, arguments}, or null on abstention."
    )
    confidence: float = Field(..., description="Edge-model confidence in [0, 1].")
    repairs: list[str] = Field(
        default_factory=list, description="Schema repairs applied by constrained decoding."
    )
    latency_ms: float = Field(..., description="End-to-end routing latency.")
    error: str | None = Field(None, description="Reason for abstention, if any.")
