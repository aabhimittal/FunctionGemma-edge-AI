"""FunctionGemma-edge-AI: a minimal, runnable walk-through of how a compact
function-calling model works end to end.

Pipeline:  tools  ->  prompt  ->  model  ->  parse  ->  call
"""

from .tools import TOOLS, get_tool
from .prompt import build_prompt, pretty
from .model import generate, mock_backend
from .parser import parse, InvalidCall


def run(user_request, backend=mock_backend):
    """Full pipeline: text in -> validated function call out."""
    prompt = build_prompt(TOOLS, user_request)
    raw = generate(prompt, backend=backend)
    return parse(raw)


__all__ = [
    "TOOLS",
    "get_tool",
    "build_prompt",
    "pretty",
    "generate",
    "mock_backend",
    "parse",
    "InvalidCall",
    "run",
]
