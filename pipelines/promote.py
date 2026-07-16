"""Promotion gate  —  the CI/CD decision: does this model reach production?

The whole pipeline converges here. A candidate is only allowed into the
``production`` slot of the model registry if it clears the evaluation gate
(tool accuracy >= threshold). This is the one place quality is enforced, so
"production" in the registry always means "passed the bar". Wire this into CI to
block regressions automatically; run it locally to promote by hand.

Exit code is non-zero when the gate fails, so a CI job can gate a deploy on it.

Usage:
    python -m pipelines.evaluate --out metrics.json --fit-calibrator calibrator.json
    python -m pipelines.promote --metrics metrics.json --calibrator calibrator.json
"""

import argparse
import json
import sys

from functiongemma.config import load_config
from functiongemma.registry import ModelRegistry


def main():
    ap = argparse.ArgumentParser(description="Gate + register a model candidate.")
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--calibrator", help="calibrator.json to attach as an artifact")
    ap.add_argument("--name", default="functiongemma-edge")
    ap.add_argument("--registry", default="registry")
    ap.add_argument("--config", help="optional config file for the gate threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.metrics, encoding="utf-8") as fh:
        metrics = json.load(fh)

    gate = cfg.min_tool_accuracy
    passed = metrics.get("tool_accuracy", 0.0) >= gate

    registry = ModelRegistry(args.registry)
    artifacts = {"metrics": args.metrics}
    if args.calibrator:
        artifacts["calibrator"] = args.calibrator
    version = registry.register(args.name, metrics=metrics, artifacts=artifacts,
                                tags={"gate": "pass" if passed else "fail"})

    print(f"registered version {version} "
          f"(tool_accuracy={metrics.get('tool_accuracy')}, gate>={gate})")

    if passed:
        registry.promote(version, "production")
        print(f"PASS: promoted {version} -> production")
        sys.exit(0)
    else:
        registry.promote(version, "staging")
        print(f"FAIL: kept {version} in staging (below gate)")
        sys.exit(1)


if __name__ == "__main__":
    main()
