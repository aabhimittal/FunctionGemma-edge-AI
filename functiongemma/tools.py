"""Step 1 — Describe the tools the model is allowed to call.

A "function calling" model does not run code itself. It only *chooses* a
function and fills in the arguments. So the first thing we need is a small
catalogue of tools, each described in a way both humans and the model can read.
"""

# A tool schema is just a plain dict: name, description, and typed parameters.
# This is the same shape Google/OpenAI use, kept minimal on purpose.
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
        },
    },
    {
        "name": "set_timer",
        "description": "Start a countdown timer.",
        "parameters": {
            "minutes": {"type": "integer", "description": "Duration in minutes"},
        },
    },
    {
        "name": "send_message",
        "description": "Send a text message to a contact.",
        "parameters": {
            "to": {"type": "string", "description": "Contact name"},
            "body": {"type": "string", "description": "Message text"},
        },
    },
]


def get_tool(name):
    """Look up a tool schema by name (or return None)."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None
