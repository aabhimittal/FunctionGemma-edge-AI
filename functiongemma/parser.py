"""Step 4 — Parse and validate the model's JSON output.

Models sometimes wrap JSON in extra text or hallucinate an argument. On the
edge there is no big server to clean this up, so validation must be cheap and
strict: extract the JSON, check the tool exists, and check the arguments match
the schema. A bad call is rejected rather than executed.
"""

import json
import re

from .tools import get_tool


class InvalidCall(Exception):
    """Raised when the model output can't become a safe, valid call."""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse(raw_output):
    """Turn raw model text into a validated {'name', 'arguments'} dict."""
    match = _JSON_RE.search(raw_output)
    if not match:
        raise InvalidCall("no JSON object found in model output")

    try:
        call = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise InvalidCall(f"malformed JSON: {exc}") from exc

    name = call.get("name")
    if name is None:
        raise InvalidCall("model abstained: no tool selected")

    tool = get_tool(name)
    if tool is None:
        raise InvalidCall(f"unknown tool: {name!r}")

    args = call.get("arguments", {})
    _check_arguments(tool, args)
    return {"name": name, "arguments": args}


def _check_arguments(tool, args):
    """Ensure every required parameter is present with the right type."""
    if not isinstance(args, dict):
        raise InvalidCall("arguments must be an object")

    for pname, spec in tool["parameters"].items():
        if pname not in args:
            raise InvalidCall(f"missing argument: {pname}")
        _check_type(pname, args[pname], spec["type"])


def _check_type(pname, value, expected):
    ok = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
    }.get(expected, True)
    if not ok:
        raise InvalidCall(f"argument {pname!r} should be {expected}")
