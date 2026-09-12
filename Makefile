.PHONY: help test lint format clean smoke-teacher smoke-train smoke-benchmark run-teacher run-train run-benchmark

# Default Python environment
PYTHON ?= python
KEDRO ?= kedro
PYTEST ?= pytest

help:
	@echo "========================================================================"
	@echo "Financial Knowledge Graph (FKG) Extraction Pipeline - Automation Engine"
	@echo "========================================================================"
	@echo "Smoke Run Targets:"
	@echo "  make smoke-teacher    : Fast sanity run of Teacher distillation (5 chunks)"
	@echo "  make smoke-train      : Fast 1-epoch training run of DynamicKGExtractor student"
	@echo "  make smoke-benchmark  : Fast 1-epoch multi-encoder benchmark run"
	@echo ""
	@echo "Production Execution Targets:"
	@echo "  make run-teacher      : Launch full teacher distillation in background (nohup)"
	@echo "  make run-train        : Train student model across company-stratified splits"
	@echo "  make run-benchmark    : Run 4-encoder benchmark (BERT, FinBERT, SEC-BERT, RoBERTa)"
	@echo ""
	@echo "Testing & Quality Targets:"
	@echo "  make test             : Run full pytest test suite (45+ tests)"
	@echo "  make lint             : Check code formatting and style with ruff"
	@echo "  make format           : Automatically format codebase with ruff"
	@echo "  make clean            : Remove temporary cache files and __pycache__"
	@echo "========================================================================"

# --- Smoke Run Targets ---

smoke-teacher:
	@echo "==> Running Teacher Extraction Smoke Test (5 chunks, streaming micro-batch)..."
	$(KEDRO) run --from-nodes generate_teacher_triplets_node --params teacher.max_samples=5,teacher.batch_size=5

smoke-train:
	@echo "==> Running Student Training Smoke Test (1 epoch, batch size 2)..."
	$(KEDRO) run --from-nodes train_model_node --params training.epochs=1,training.batch_size=2,training.compile_model=false

smoke-benchmark:
	@echo "==> Running Benchmark Smoke Test (1 epoch, bert-base-uncased)..."
	$(KEDRO) run --pipeline benchmark --params training.epochs=1,benchmark.models="['bert-base-uncased']",training.compile_model=false

# --- Production Execution Targets ---

run-teacher:
	@echo "==> Launching Teacher Distillation in background with nohup..."
	@nohup $(KEDRO) run --from-nodes generate_teacher_triplets_node > teacher_distill.log 2>&1 & \
		echo "Teacher distillation started in background! PID: $$!"
	@echo "Monitor live logs with: tail -f teacher_distill.log"

run-train:
	@echo "==> Starting Student Model Training Pipeline..."
	$(KEDRO) run --pipeline training

run-benchmark:
	@echo "==> Starting Multi-Encoder Benchmark Pipeline..."
	$(KEDRO) run --pipeline benchmark

# --- Testing & Quality Targets ---

test:
	@echo "==> Running Pytest Test Suite..."
	$(PYTEST) tests/ -o pythonpath=src -v

lint:
	@echo "==> Checking code quality with ruff..."
	ruff check .
	ruff format --check .

format:
	@echo "==> Formatting code with ruff..."
	ruff format .
	ruff check --fix .

clean:
	@echo "==> Cleaning cache directories..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
