# Contributing

Thanks for your interest! This repo values **small, readable, runnable** changes.

## Setup

```bash
pip install -r requirements-dev.txt
pre-commit install     # optional: run lint on every commit
```

## Before you open a PR

```bash
make lint     # ruff
make test     # pytest
make eval     # evaluation gate must still pass (no metric regression)
```

CI runs the same three steps on Python 3.10–3.12 plus a dry-run of the pipeline.

## Guidelines
- **Keep the core dependency-free.** The `functiongemma` package must import and
  run on the standard library alone. Heavy deps (torch, transformers, fastapi)
  belong behind lazy imports or in the serving/ML extras.
- **One idea per module.** New behaviour usually means a new small file, not a
  bigger existing one.
- **Tests are part of the change.** Add a focused test next to your module.
- **Docs track behaviour.** Update `README.md` / `docs/` if you change what the
  pipeline does.
- **Never let the library execute a tool.** The boundary — model chooses, host
  executes — is the whole design; keep it intact.

## Adding a tool
1. Add it to `data/tools_catalog.json` (the single source of truth).
2. Add a mock rule in `functiongemma/model.py` so the demo covers it.
3. Add golden examples in `data/eval/golden.jsonl` (including when to abstain).
