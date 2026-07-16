# Developer entrypoints. `make help` lists them. The pipeline targets mirror the
# CI job, so "works on my machine" and "works in CI" stay the same thing.

.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install demo test lint data eval calibrate promote pipeline serve docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dev + serving dependencies
	$(PY) -m pip install -r requirements-dev.txt

demo: ## Run the end-to-end demo (mock backend, no downloads)
	$(PY) demo.py

test: ## Run the test suite
	$(PY) -m pytest -q

lint: ## Lint with ruff
	$(PY) -m ruff check .

data: ## Generate synthetic training data
	$(PY) -m pipelines.generate_data --n 500 --out data/train.jsonl

eval: ## Evaluate the current backend -> metrics.json
	$(PY) -m pipelines.evaluate --out metrics.json --fit-calibrator calibrator.json

promote: eval ## Gate on metrics.json and register/promote in the model registry
	$(PY) -m pipelines.promote --metrics metrics.json --calibrator calibrator.json

pipeline: data eval promote ## Full offline MLops loop: data -> eval -> gate -> promote

serve: ## Run the FastAPI edge server
	$(PY) -m uvicorn serving.app:app --host 0.0.0.0 --port 8000

docker: ## Build the serving container image
	docker build -t functiongemma-edge:local .

clean: ## Remove generated artifacts
	rm -rf metrics.json calibrator.json events.jsonl registry data/train.jsonl \
		artifacts .pytest_cache **/__pycache__
