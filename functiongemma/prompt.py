"""Step 2 — Turn the tools + user request into a single prompt string.

A compact edge model has a small context window, so the prompt must be tight.
We list the tools compactly, then the user's request, then ask the model to
reply with ONE JSON object. This "output contract" is what makes parsing easy.
"""

import json


SYSTEM = (
    "You are FunctionGemma, a function-calling assistant. "
    "Choose exactly one tool and reply ONLY with JSON like "
    '{"name": <tool>, "arguments": {...}}. No prose.'
)


def build_prompt(tools, user_request):
    """Assemble the full prompt the model sees."""
    lines = [SYSTEM, "", "Tools:"]
    for tool in tools:
        params = ", ".join(tool["parameters"].keys())
        lines.append(f"- {tool['name']}({params}): {tool['description']}")
    lines += ["", f"User: {user_request}", "JSON:"]
    return "\n".join(lines)


def pretty(call):
    """Format a parsed call back into readable JSON (for demos/logs)."""
    return json.dumps(call, indent=2)
