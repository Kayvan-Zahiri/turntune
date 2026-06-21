# turntune — common tasks.
#
# The one command for a stranger:  make run
# It creates a venv, installs turntune, and launches the local web UI, which on first
# run downloads the Silero VAD model + a small eot-bench subset.

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: run dev test fmt lint clean

## run: create venv, install, and launch the local web UI at http://localhost:8000
run: $(VENV)
	$(BIN)/turntune serve --open

$(VENV):
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	# Regular (non-editable) install: robust everywhere, including environments where
	# .pth-based editable installs aren't honored. For live-editing, see `make dev`.
	$(BIN)/pip install ".[dev]"

## dev: editable install for contributors (live code edits; needs .pth support)
dev: $(VENV)
	$(BIN)/pip install -e ".[dev]"

## test: run the test suite (uses bundled offline fixtures, no network)
test: $(VENV)
	$(BIN)/pytest

## fmt: auto-format and fix lint
fmt: $(VENV)
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

## lint: check formatting and lint without modifying
lint: $(VENV)
	$(BIN)/ruff format --check .
	$(BIN)/ruff check .

## clean: remove the venv and the runtime cache
clean:
	rm -rf $(VENV) .turntune_cache
