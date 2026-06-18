.PHONY: setup test lint format clean

setup:
	pip install -e ".[dev]"
	pre-commit install
	dvc init -q || true

test:
	pytest -q

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
