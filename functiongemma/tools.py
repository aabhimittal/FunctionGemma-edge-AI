"""Step 1 — Describe the tools the model is allowed to call.

A "function calling" model does not run code itself. It only *chooses* a
function and fills in the arguments. So the first thing we need is a small
catalogue of tools, each described in a way both humans and the model can read.

The catalogue is loaded from ``data/tools_catalog.json`` so that the training
data generator, the evaluator and the serving layer all read the *same*
contract — a small but important MLops discipline: edge and cloud must never
disagree about what a valid call looks like. If the file is missing (e.g. the
package is vendored somewhere without the data dir) we fall back to a built-in
copy so the library always imports.
"""

import json
import os

# data/tools_catalog.json lives one level up from this package.
_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "tools_catalog.json",
)

# A tool schema is just a plain dict: name, description, and typed parameters.
# This is the same shape Google/OpenAI use, kept minimal on purpose. Used only
# if the JSON catalogue cannot be read.
_BUILTIN_TOOLS = [
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


def load_tools(path=_CATALOG_PATH):
    """Load the tool catalogue from JSON, falling back to the built-in list."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["tools"]
    except (OSError, KeyError, json.JSONDecodeError):
        return _BUILTIN_TOOLS


#: The active tool catalogue. Import this everywhere.
TOOLS = load_tools()


def get_tool(name, tools=None):
    """Look up a tool schema by name (or return None)."""
    for tool in tools if tools is not None else TOOLS:
        if tool["name"] == name:
            return tool
    return None
