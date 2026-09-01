.PHONY: help setup data smoke test lint

help:
	@echo "setup   - pip install -e . (+dev)"
	@echo "data    - download + build labelled table + 70/15/15 split (phase 0)"
	@echo "smoke   - fast sanity training run (residual bridge, 50 samples, 1 epoch)"
	@echo "test    - pytest"
	@echo "lint    - ruff check"

setup:
	pip install -e ".[dev,metrics]"

data:
	python scripts/phase0_build_data.py

smoke:
	python -m src.cli.train --bridge residual --smoke

test:
	pytest -q

lint:
	ruff check src scripts tests
