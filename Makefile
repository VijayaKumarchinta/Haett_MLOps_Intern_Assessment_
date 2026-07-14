.PHONY: help install pipeline fast-pipeline api test lint clean

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:        ## Install dependencies
	pip install -r requirements.txt

pipeline:       ## Run full pipeline (with hyperparameter tuning)
	python src/run_pipeline.py

fast-pipeline:  ## Run pipeline without hyperparameter tuning (fast)
	N_HPARAM_ITER=5 python src/run_pipeline.py

api:            ## Start the API server (model must exist)
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run: pipeline   ## Alias for full pipeline

start:          ## Pipeline + API in one command
	python run.py --all --fast

test:           ## Run all tests
	python -m pytest tests/ -v --tb=short

lint:           ## Lint and format check
	pip install -q ruff black 2>/dev/null; \
	black --check --diff src/ tests/; \
	ruff check src/ tests/

docker-build:   ## Build Docker image
	docker build -t haett-churn-api .

docker-run:     ## Run with Docker
	docker run -p 8000:8000 -v $(PWD)/models:/app/models haett-churn-api

docker-compose: ## Run with docker-compose (includes MLflow)
	docker-compose up --build

clean:          ## Clean generated files
	rm -rf data/raw/* data/processed/* data/features/*
	rm -f models/churn_model.pkl models/tuned_model.pkl
	rm -f models/optimal_threshold.txt models/feature_names.txt
	rm -f models/model_metadata.json models/*.csv models/*.png
	rm -rf mlruns/
	@echo "Cleaned all generated data and models."
