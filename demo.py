"""Runnable demo: `python demo.py`

Two acts, both on the built-in mock backend (no model download):

  1. The **core pipeline** — text in, one validated JSON call out.
  2. The **confidence-gated cascade** — the same requests routed edge -> cloud
     with a calibrated abstention gate, printing which tier answered and why.
"""

from functiongemma import (
    TOOLS,
    Calibrator,
    Cascade,
    InvalidCall,
    build_prompt,
    coverage_report,
    pretty,
    run,
)

EXAMPLES = [
    "What's the weather in Paris?",
    "Set a timer for 10 minutes",
    "Send a message to Alex",
    "Schedule a meeting at 3",
    "Play Radiohead",
    "Tell me a joke",  # no matching tool -> the model abstains
]


def core_demo():
    print("#" * 62)
    print("# 1. Core pipeline  (single model:  prompt -> model -> parse)")
    print("#" * 62)
    for request in EXAMPLES:
        print("=" * 60)
        print("User:", request)
        try:
            print("Call:", pretty(run(request)))
        except InvalidCall as exc:
            print("No valid call:", exc)


def cascade_demo():
    print("\n" + "#" * 62)
    print("# 2. Confidence-gated cascade  (edge -> cloud, calibrated gate)")
    print("#" * 62)
    # A middling threshold so easy requests stay on-device and vaguer ones escalate.
    cascade = Cascade(calibrator=Calibrator(tau=0.55))
    decisions = [cascade.route(r) for r in EXAMPLES]
    for d in decisions:
        call = d.call["name"] if d.call else "—"
        print(f"[{d.tier:>7}] conf={d.confidence:0.2f}  {d.request!r} -> {call}")
    print("-" * 60)
    print("coverage report:", coverage_report(decisions))


def main():
    core_demo()
    cascade_demo()
    print("\nExample prompt sent to the model:\n")
    print(build_prompt(TOOLS, EXAMPLES[0]))


if __name__ == "__main__":
    main()
