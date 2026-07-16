"""Schema-constrained decoding  —  the second half of the novel idea.

The failure mode of a small model is not usually *bad intent*, it is *invalid
form*: it picks a tool that doesn't exist, misspells an argument key, or emits a
string where an integer belongs. On a server you'd retry or repair with a second
model. On the edge you have neither the latency budget nor a second model.

The industrial fix is **constrained decoding**: only let the model emit tokens
that keep the output on a valid path through the tool grammar. A full
implementation masks the logits at each step with a JSON-schema-derived
automaton (libraries: Outlines, llguidance, XGrammar). That needs the real model
in the loop, so here we ship the *decode-time projection* that the same idea
reduces to once the raw text exists:

    raw model text  ->  snap to the nearest valid call  ->  typed, in-schema call

``constrain`` never invents a tool that wasn't offered and never keeps an
argument the schema doesn't define; it only *repairs* recoverable mistakes
(fuzzy tool-name match, integer strings like "10" -> 10) and reports what it
changed. Unrepairable output is left for the parser to reject — constrained
decoding reduces the reject rate, it does not hide real failures.
"""

import json
import re

from .tools import get_tool

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def constrain(raw_output, tools):
    """Project raw model text onto the tool grammar.

    Returns ``(call_or_None, repairs)`` where ``repairs`` is a list of the edits
    applied. ``None`` means the text could not be snapped onto any tool.
    """
    repairs = []
    match = _JSON_RE.search(raw_output or "")
    if not match:
        return None, repairs
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None, repairs
    if not isinstance(obj, dict):
        return None, repairs

    name = obj.get("name")
    if name is None:
        return None, repairs  # a genuine abstention, not a form error.

    tool = get_tool(name, tools)
    if tool is None:
        fixed = _nearest_tool_name(str(name), tools)
        if fixed is None:
            return None, repairs
        repairs.append(f"tool {name!r} -> {fixed!r}")
        name = fixed
        tool = get_tool(name, tools)

    raw_args = obj.get("arguments") or {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    args = {}
    for pname, spec in tool["parameters"].items():
        if pname not in raw_args:
            continue  # missing required arg is a real failure -> parser rejects.
        value, changed = _coerce(raw_args[pname], spec["type"])
        if changed:
            repairs.append(f"{pname}: {raw_args[pname]!r} -> {value!r}")
        args[pname] = value

    # Drop keys the schema never declared (hallucinated arguments).
    for extra in set(raw_args) - set(tool["parameters"]):
        repairs.append(f"dropped unknown argument {extra!r}")

    return {"name": name, "arguments": args}, repairs


def _coerce(value, expected):
    """Best-effort snap of a value to the schema type. Returns (value, changed)."""
    if expected == "integer":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, int):
            return value, False
        m = re.search(r"-?\d+", str(value))
        if m:
            return int(m.group()), True
    if expected == "string" and not isinstance(value, str):
        return str(value), True
    return value, False


def _nearest_tool_name(name, tools, max_edits=2):
    """Fuzzy-match a misspelled tool name to a real one via edit distance."""
    best, best_d = None, max_edits + 1
    for tool in tools:
        d = _levenshtein(name.lower(), tool["name"].lower())
        if d < best_d:
            best, best_d = tool["name"], d
    return best if best_d <= max_edits else None


def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
