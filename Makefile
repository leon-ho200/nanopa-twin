PY ?= python
PKG = nanopa_twin

.PHONY: help install dev lint format type test smoke docker clean

help:
	@echo "targets: install dev lint format type test smoke docker clean"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check $(PKG) tests
	$(PY) -m isort --check-only $(PKG) tests
	$(PY) -m black --check $(PKG) tests

format:
	$(PY) -m isort $(PKG) tests
	$(PY) -m black $(PKG) tests

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest

smoke:
	$(PY) -m $(PKG).dial fit --preset _smoke --out runs --epochs 2

docker:
	docker build -t nanopa-twin .

clean:
	rm -rf runs build dist *.egg-info .pytest_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
