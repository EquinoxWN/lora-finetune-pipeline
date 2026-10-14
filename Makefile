.PHONY: setup setup-gpu lint test data run run-lite show purge audit ci

PY ?= python
# CPU wheels for CI and the unit tests; the GPU run uses CUDA 13.0 wheels (setup-gpu).
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
TORCH_GPU_INDEX ?= https://download.pytorch.org/whl/cu130
TORCH_VERSION ?= 2.14.1

# Every cache, download and temp file stays inside the repo's git-ignored .tmp folder.
TMPD := $(CURDIR)/.tmp
export TMP := $(TMPD)
export TEMP := $(TMPD)
export TMPDIR := $(TMPD)
export HF_HOME := $(TMPD)/hf
export HF_HUB_CACHE := $(TMPD)/hf/hub
export HF_XET_CACHE := $(TMPD)/hf/xet
export HF_DATASETS_CACHE := $(TMPD)/hf/datasets
export TRANSFORMERS_CACHE := $(TMPD)/hf/hub
export TORCH_HOME := $(TMPD)/torch
export TRITON_CACHE_DIR := $(TMPD)/triton
export CUDA_CACHE_PATH := $(TMPD)/cuda-cache
export MPLCONFIGDIR := $(TMPD)/mpl
export PIP_CACHE_DIR := $(TMPD)/pip
export WANDB_DISABLED := true
export WANDB_MODE := disabled
export HF_HUB_DISABLE_TELEMETRY := 1
export DO_NOT_TRACK := 1
export PYTHONDONTWRITEBYTECODE := 1

setup:
	mkdir -p .tmp
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install "torch==$(TORCH_VERSION)" --index-url $(TORCH_INDEX)
	$(PY) -m pip install -e ".[dev]" "setuptools>=80"

# Local GPU run: CUDA torch, bitsandbytes, accelerate and the NVML energy counter.
setup-gpu:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install "torch==$(TORCH_VERSION)" --index-url $(TORCH_GPU_INDEX)
	$(PY) -m pip install -e ".[gpu,dev]" "setuptools>=80"

# ruff, ruff format and mypy --strict.
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy

# Unit and end-to-end tests on a tiny random model; no GPU and no download.
test:
	mkdir -p .tmp
	$(PY) -m pytest -q

# Download BANKING77 (SHA-256 pinned), de-duplicate, split and check for leakage.
data:
	$(PY) -m lora_finetune_pipeline.cli prepare-data --config configs/m1.json

# GPU: zero-shot and few-shot prompting vs a QLoRA adapter on the full held-out split;
# deletes the base model and adapter once the numbers are written.
run:
	$(PY) -m lora_finetune_pipeline.cli run --config configs/m1.json --out docs/results/m1-run.json

# The smaller run for machines with less memory or time (385 held-out messages, 150 steps).
run-lite:
	$(PY) -m lora_finetune_pipeline.cli run --config configs/lite.json --out docs/results/lite-run.json

show:
	$(PY) -m lora_finetune_pipeline.cli show docs/results/m1-run.json

# Delete any model, adapter and dataset files left in .tmp.
purge:
	$(PY) -m lora_finetune_pipeline.cli purge --data

# Known vulnerabilities in the installed Python dependencies. pip-audit skips torch's local
# version (+cpu / +cu130), so the matching public torch release is audited on its own.
audit:
	mkdir -p .tmp
	$(PY) -m pip_audit --skip-editable --cache-dir .tmp/pip-audit
	echo "torch==$(TORCH_VERSION)" > .tmp/torch-req.txt
	$(PY) -m pip_audit -r .tmp/torch-req.txt --no-deps --disable-pip --cache-dir .tmp/pip-audit

ci: setup lint test
