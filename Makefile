.PHONY: train serve test test-unit test-integration test-e2e test-coverage lint format docker-build docker-run run-all smoke-test clean

train:
	python -m training.train

serve:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

test-unit:
	pytest tests/unit/ -m unit

test-integration:
	pytest tests/integration/ -m integration

test-e2e:
	pytest tests/e2e/ -m e2e

test-coverage:
	pytest --cov=app --cov-report=term-missing --cov-report=html

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

docker-build:
	docker build -t penguin-ml-api .

docker-run:
	docker compose up

run-all: train test docker-build

smoke-test:
	python scripts/test_live.py $(URL)

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
