.PHONY: help test tes lint ruff check-all format clean smoke-teacher smoke-train smoke-benchmark run-teacher run-train run-benchmark

# Python & execution settings
PYTHON ?= python
KEDRO ?= kedro
PYTEST ?= pytest
RUFF ?= ruff
GPU ?= 1

# If GPU is specified, prepend CUDA_VISIBLE_DEVICES and unbuffer python output
ifneq ($(strip $(GPU)),)
  GPU_ENV = CUDA_VISIBLE_DEVICES=$(GPU) PYTHONUNBUFFERED=1
else
  GPU_ENV = PYTHONUNBUFFERED=1
endif

help:
	@echo "========================================================================"
	@echo "Financial Knowledge Graph (FKG) Extraction Pipeline - Automation Engine"
	@echo "========================================================================"
	@echo "Quality & Testing Targets:"
	@echo "  make test / make tes  : Run full pytest test suite (45+ tests)"
	@echo "  make lint / make ruff : Run ruff lint & format checks"
	@echo "  make format           : Auto-fix formatting and lint issues with ruff"
	@echo "  make check-all        : Run comprehensive quality gate (lint + format + tests)"
	@echo ""
	@echo "Smoke Run Targets (default GPU=1):"
	@echo "  make smoke-teacher    : Fast sanity run of Teacher distillation (5 chunks)"
	@echo "  make smoke-train      : Fast 1-epoch training run of DynamicKGExtractor student"
	@echo "  make smoke-benchmark  : Fast 1-epoch multi-encoder benchmark run"
	@echo "  * Override GPU with   : make smoke-teacher GPU=0 (or GPU=)"
	@echo ""
	@echo "Production Execution Targets:"
	@echo "  make run-teacher      : Launch full teacher distillation in background (nohup)"
	@echo "  make run-train        : Train student model across company-stratified splits"
	@echo "  make run-benchmark    : Run 4-encoder benchmark (BERT, FinBERT, SEC-BERT, RoBERTa)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            : Remove temporary cache files and __pycache__"
	@echo "========================================================================"

# --- Testing & Quality Targets ---

test:
	@echo "==> Running Pytest Test Suite..."
	$(PYTEST) tests/ -o pythonpath=src -v

# Alias for make test
tes: test

lint:
	@echo "==> Checking code quality with ruff..."
	$(RUFF) check .
	$(RUFF) format --check .

ruff: lint

format:
	@echo "==> Formatting code and fixing lint issues with ruff..."
	$(RUFF) format .
	$(RUFF) check --fix .

check-all:
	@echo "========================================================"
	@echo "==> Step 1/2: Running Ruff Lint & Format Checks..."
	@echo "========================================================"
	$(RUFF) check .
	$(RUFF) format --check .
	@echo "========================================================"
	@echo "==> Step 2/2: Running Complete Pytest Suite..."
	@echo "========================================================"
	$(PYTEST) tests/ -o pythonpath=src -v
	@echo "========================================================"
	@echo "==> All Quality Gates Passed Successfully! 🚀"
	@echo "========================================================"

# --- Smoke Run Targets ---

smoke-teacher:
	@echo "==> Running Teacher Extraction Smoke Test on GPU=$(GPU)..."
	$(GPU_ENV) $(KEDRO) run --nodes generate_teacher_triplets_node --params teacher.max_samples=5,teacher.batch_size=5

smoke-train:
	@echo "==> Running Student Training Smoke Test (1 epoch, batch size 2)..."
	$(GPU_ENV) $(KEDRO) run --pipeline training --params training.epochs=1,training.batch_size=2,training.compile_model=false

smoke-benchmark-encoders:
	@echo "==> Running Encoder Benchmark Smoke Test (1 epoch)..."
	$(GPU_ENV) $(KEDRO) run --pipeline benchmark_encoders --params benchmark.epochs=1

smoke-benchmark-decoders:
	@echo "==> Running Decoder Benchmark Smoke Test (1 epoch)..."
	$(GPU_ENV) $(KEDRO) run --pipeline benchmark_decoders --params benchmark.epochs=1
# --- Production Execution Targets ---

run-teacher:
	@echo "==> Launching Teacher Distillation in background on GPU=$(GPU) with nohup..."
	@$(GPU_ENV) nohup $(KEDRO) run --nodes generate_teacher_triplets_node > teacher_distill.log 2>&1 & \
		echo "Teacher distillation started in background! PID: $$!"
	@echo "Monitor live logs with: tail -f teacher_distill.log"

run-train:
	@echo "==> Launching Student Model Training Pipeline in background on GPU=$(GPU) with nohup..."
	@$(GPU_ENV) nohup $(KEDRO) run --pipeline training > training.log 2>&1 & \
		echo "Training started in background! PID: $$!"
	@echo "Monitor live logs with: tail -f training.log"

run-benchmark-encoders:
	@echo "==> Launching Multi-Encoder Benchmark in background on GPU=$(GPU) with nohup..."
	@$(GPU_ENV) nohup $(KEDRO) run --pipeline benchmark_encoders > benchmark_encoders.log 2>&1 & \
		echo "Encoder Benchmark started in background! PID: $$!"
	@echo "Monitor live logs with: tail -f benchmark_encoders.log"

run-benchmark-decoders:
	@echo "==> Launching Multi-Decoder Benchmark in background on GPU=$(GPU) with nohup..."
	@$(GPU_ENV) nohup $(KEDRO) run --pipeline benchmark_decoders > benchmark_decoders.log 2>&1 & \
		echo "Decoder Benchmark started in background! PID: $$!"
	@echo "Monitor live logs with: tail -f benchmark_decoders.log"

report:
	@echo "==> Generating Model Evaluation Report & Visualizations..."
	$(GPU_ENV) $(KEDRO) run --pipeline reporting

# --- Maintenance ---

clean:
	@echo "==> Cleaning cache directories..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
