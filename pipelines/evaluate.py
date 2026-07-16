"""Evaluation harness  —  the quality gate every other step trusts.

Runs a backend over the golden set (``data/eval/golden.jsonl``) and reports the
metrics that matter for function calling:

  * **tool_accuracy**  — did we pick the right tool (abstaining when we should)?
  * **arg_accuracy**    — of the calls where the tool was right, did the
                          required arguments match?
  * **exact_match**     — tool AND all checked arguments correct.
  * **abstain_precision / recall** — do we abstain when (and only when) we should?

It also emits the (confidence, correct) pairs and can fit the abstention
:class:`~functiongemma.confidence.Calibrator` on them, so the same run that
scores a model also produces the calibrated threshold shipped alongside it.

Usage:
    python -m pipelines.evaluate                       # print metrics
    python -m pipelines.evaluate --out metrics.json    # + write metrics
    python -m pipelines.evaluate --fit-calibrator calibrator.json
"""

import argparse
import json

from functiongemma.confidence import Calibrator, score
from functiongemma.constrained import constrain
from functiongemma.model import mock_backend
from functiongemma.parser import InvalidCall, validate
from functiongemma.prompt import build_prompt
from functiongemma.tools import TOOLS


def load_golden(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def predict(request, backend, tools=TOOLS):
    """Return ``(call_or_None, confidence)`` for one request via a backend."""
    raw = backend(build_prompt(tools, request))
    call, _ = constrain(raw, tools)
    if call is not None:
        try:
            call = validate(call, tools)
        except InvalidCall:
            call = None
    return call, score(request, call, tools=tools)


def _args_match(expected_args, got_args):
    """Every argument the golden example pins must match (case-insensitive)."""
    for key, want in expected_args.items():
        got = got_args.get(key)
        if isinstance(want, str) and isinstance(got, str):
            if want.lower() != got.lower():
                return False
        elif got != want:
            return False
    return True


def evaluate(backend=mock_backend, golden_path="data/eval/golden.jsonl", tools=TOOLS):
    """Score ``backend`` on the golden set. Returns (metrics, pairs)."""
    golden = load_golden(golden_path)
    tool_correct = tool_and_args_correct = 0
    abstain_tp = abstain_fp = abstain_fn = 0
    pairs = []  # (confidence, was_the_whole_call_correct)

    for ex in golden:
        want = ex["expected"]
        want_name = want.get("name")
        got, conf = predict(ex["request"], backend, tools)
        got_name = got["name"] if got else None

        name_ok = got_name == want_name
        args_ok = name_ok and (
            got_name is None or _args_match(want.get("arguments", {}), got["arguments"])
        )
        tool_correct += name_ok
        tool_and_args_correct += args_ok

        # Abstention confusion matrix (positive class = "should abstain").
        if want_name is None and got_name is None:
            abstain_tp += 1
        elif want_name is not None and got_name is None:
            abstain_fp += 1
        elif want_name is None and got_name is not None:
            abstain_fn += 1

        pairs.append((conf, bool(args_ok)))

    n = len(golden) or 1
    metrics = {
        "n": len(golden),
        "tool_accuracy": round(tool_correct / n, 4),
        "exact_match": round(tool_and_args_correct / n, 4),
        "arg_accuracy": round(
            tool_and_args_correct / tool_correct, 4
        ) if tool_correct else 0.0,
        "abstain_precision": round(abstain_tp / (abstain_tp + abstain_fp), 4)
        if (abstain_tp + abstain_fp) else 1.0,
        "abstain_recall": round(abstain_tp / (abstain_tp + abstain_fn), 4)
        if (abstain_tp + abstain_fn) else 1.0,
    }
    return metrics, pairs


def main():
    ap = argparse.ArgumentParser(description="Evaluate a FunctionGemma backend.")
    ap.add_argument("--golden", default="data/eval/golden.jsonl")
    ap.add_argument("--out", help="write metrics.json here")
    ap.add_argument("--fit-calibrator", help="fit + save abstention threshold here")
    ap.add_argument("--target-accuracy", type=float, default=0.9)
    args = ap.parse_args()

    metrics, pairs = evaluate(golden_path=args.golden)
    print(json.dumps(metrics, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"wrote {args.out}")

    if args.fit_calibrator:
        confs, correct = zip(*pairs, strict=False) if pairs else ([], [])
        cal = Calibrator(target_accuracy=args.target_accuracy).fit(confs, correct)
        cal.save(args.fit_calibrator)
        print(f"fitted calibrator tau={cal.tau} -> {args.fit_calibrator}")


if __name__ == "__main__":
    main()
