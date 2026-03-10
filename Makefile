.PHONY: install ingest evaluate serve test lint clean

install:
	pip install -e ".[dev]"

ingest:
	python scripts/run_ingestion.py

evaluate:
	python scripts/run_evaluation.py

serve:
	uvicorn serve.api:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
