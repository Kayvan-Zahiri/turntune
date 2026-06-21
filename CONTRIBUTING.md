# Contributing to turntune

Thanks for your interest! turntune is built in public and designed to be extended.
There are two clean seams; you can add to either without touching the harness,
metrics, or UI.

## Dev setup

```bash
make dev          # creates .venv and installs turntune with dev extras
make test         # runs the suite against bundled offline fixtures (no network)
make fmt          # auto-format + fix lint with ruff
make lint         # check-only
```

## Seam #1 — add a detector

A detector is split into two halves on purpose:

- **`extract(frames) -> FrameSignal`** — the *expensive* pass. Runs your model over
  the 20 ms / 16 kHz audio frames once and returns a per-frame signal. Must be
  **causal** (no peeking at future audio) and is **cached** by turntune.
- **`decide(signal, params) -> EotDecision`** — the *cheap, pure* pass. Turns the
  cached signal plus the current tuning params into an end-of-turn decision. This is
  what the threshold sweep and the live sliders hammer, so keep it fast and
  side-effect free.

Also declare `name`, `version`, `frame_ms`, `default_params()`, and
`param_space()` (the latter drives both the sweep grid and the auto-generated UI
sliders). Register your class so the CLI/UI can find it by name.

See [`examples/custom_detector.py`](./examples/custom_detector.py) for a complete,
minimal worked example, and `src/turntune/detectors/silero_vad.py` for the shipped
default.

> A future semantic / end-of-turn model is the same shape: `extract()` emits EOT
> logits, `decide()` thresholds them — reusing the exact same cache-once / replay
> machinery. It lives behind the optional `[semantic]` extra and is **not** part of v0.

## Seam #2 — add a scenario source

Implement a `ScenarioLoader` whose `load()` yields `Scenario` objects normalized to
the common schema: audio (16 kHz mono), `hold`/`eot` silence spans, and a
`true_eot_s`. The bundled fixtures loader
(`src/turntune/scenarios/fixtures.py`) is the simplest worked example;
`eot_bench.py` is the default.

## Pull requests

- Keep v0 scope in mind — see the "out of scope" list in the README.
- Run `make lint` and `make test` before opening a PR.
- New detectors/loaders should come with a small test.
