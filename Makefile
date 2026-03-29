SHELL := /bin/sh
.DEFAULT_GOAL := help

WEB := apps/web
API := apps/api
API_PORT := 8000
PYTHON ?= python
DATA_ROOT ?= data
CHECKPOINT ?= results/reconmmt/best_reconmmt.pt
IMAGE_SIZE ?= 256
BATCH_SIZE ?= 4
EPOCHS ?= 30
OUTPUT_DIR ?= results/reconmmt

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
	OS := mac
else ifeq ($(UNAME_S),Linux)
	OS := linux
else ifneq (,$(findstring MINGW,$(UNAME_S)))
	OS := windows
else ifneq (,$(findstring MSYS,$(UNAME_S)))
	OS := windows
else
	OS := windows
endif

API_VENV := $(API)/.venv

ifeq ($(OS),windows)
	PY_API := $(API_VENV)/Scripts/python.exe
else
	PY_API := $(API_VENV)/bin/python
endif

.PHONY: help api web install compile smoke train test infer export clean-results

help:
	@echo "Available targets:"
	@echo "  install        Install Python dependencies for the research model"
	@echo "  compile        Compile-check the Python source files"
	@echo "  smoke          Run the ReconMMT smoke test"
	@echo "  train          Train ReconMMT on data/\<split\>/..."
	@echo "  test           Evaluate a ReconMMT checkpoint"
	@echo "  infer          Run one inference sample"
	@echo "  export         Export a checkpoint to ONNX"
	@echo "  clean-results  Remove generated result files from results/reconmmt"
	@echo "  api            Run the app API prototype"
	@echo "  web            Run the app web prototype"
	@echo ""
	@echo "Configurable variables:"
	@echo "  PYTHON=$(PYTHON) DATA_ROOT=$(DATA_ROOT) CHECKPOINT=$(CHECKPOINT)"
	@echo "  IMAGE_SIZE=$(IMAGE_SIZE) BATCH_SIZE=$(BATCH_SIZE) EPOCHS=$(EPOCHS) OUTPUT_DIR=$(OUTPUT_DIR)"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision pillow onnx

compile:
	$(PYTHON) -m py_compile main.py train.py test.py models/algo.py

smoke:
	$(PYTHON) main.py smoke-test --image-size 64

train:
	$(PYTHON) train.py --data-root $(DATA_ROOT) --output-dir $(OUTPUT_DIR) --image-size $(IMAGE_SIZE) --batch-size $(BATCH_SIZE) --epochs $(EPOCHS)

test:
	$(PYTHON) test.py --data-root $(DATA_ROOT) --checkpoint $(CHECKPOINT) --image-size $(IMAGE_SIZE) --batch-size $(BATCH_SIZE)

infer:
	@echo "Usage:"
	@echo "  make infer PYTHON=python CHECKPOINT=results/reconmmt/best_reconmmt.pt RGB=... THERMAL=... DEPTH=... OUTPUT=results/reconmmt/infer.png"
	$(PYTHON) main.py infer --checkpoint $(CHECKPOINT) --rgb $(RGB) --thermal $(THERMAL) --depth $(DEPTH) --output $(OUTPUT) --image-size $(IMAGE_SIZE)

export:
	$(PYTHON) main.py export-onnx --checkpoint $(CHECKPOINT) --output $(OUTPUT_DIR)/reconmmt.onnx --image-size $(IMAGE_SIZE)

clean-results:
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('$(OUTPUT_DIR)'), ignore_errors=True)"

api:
	cd $(API) && poetry run uvicorn main:app --app-dir src --reload --port $(API_PORT)

web:
	cd $(WEB) && npm run dev
