"""Runnable demo: `python demo.py`

Shows the full FunctionGemma pipeline on a few example requests using the
built-in mock backend (no model download required).
"""

from functiongemma import build_prompt, run, pretty, InvalidCall, TOOLS

EXAMPLES = [
    "What's the weather in Paris?",
    "Set a timer for 10 minutes",
    "Send a message to Alex",
    "Tell me a joke",  # no matching tool -> the model abstains
]


def main():
    for request in EXAMPLES:
        print("=" * 60)
        print("User:", request)
        try:
            call = run(request)
            print("Call:")
            print(pretty(call))
        except InvalidCall as exc:
            print("No valid call:", exc)

    # Peek at the prompt the model actually receives for one example.
    print("=" * 60)
    print("Example prompt sent to the model:\n")
    print(build_prompt(TOOLS, EXAMPLES[0]))


if __name__ == "__main__":
    main()
