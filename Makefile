.PHONY: help install ingest eval ablate test lint clean docker

PY ?= python
export PYTHONPATH := src

help:
	@echo "install  install the package with dev extras"
	@echo "ingest   build artifacts/manifest.sqlite from data/raw"
	@echo "eval     run the golden-set evaluation (non-zero exit on regression)"
	@echo "ablate   regenerate reports/ablation.md"
	@echo "test     run the test suite"
	@echo "lint     ruff check"
	@echo "clean    remove artifacts and generated reports"

install:
	$(PY) -m pip install -e ".[dev]"

ingest:
	$(PY) -m pria.cli ingest

eval: ingest
	$(PY) -m pria.cli eval

ablate: ingest
	$(PY) -m pria.cli ablate

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

clean:
	rm -rf artifacts reports .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker:
	docker compose up --build
