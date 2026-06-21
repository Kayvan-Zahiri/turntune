"""Command-line entry point — the one-command brain.

`turntune serve` (default) bootstraps everything (download the model + an eot-bench
subset, materialize wavs, build the cached signals), then launches the local web UI
and opens a browser. `turntune run` / `turntune sweep` are headless variants.
"""

from __future__ import annotations

import argparse
import json

from . import __version__, config
from . import detectors as _detectors  # noqa: F401  (registers built-in detectors)
from . import scenarios as _scenarios  # noqa: F401  (registers built-in loaders)
from .config import (
    DEFAULT_DETECTOR,
    DEFAULT_LANGUAGE,
    DEFAULT_LIMIT,
    DEFAULT_PORT,
    DEFAULT_SWEEP_AXIS,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="turntune", description="Tune turn-taking in voice agents.")
    p.add_argument("--version", action="version", version=f"turntune {__version__}")
    sub = p.add_subparsers(dest="command")

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--detector", default=DEFAULT_DETECTOR, help="detector adapter name")
        sp.add_argument(
            "--dataset",
            "--loader",
            dest="loader",
            default="eot-bench",
            help="scenario loader name (e.g. eot-bench, fixtures)",
        )
        sp.add_argument("--language", default=DEFAULT_LANGUAGE)
        sp.add_argument(
            "--limit", type=int, default=DEFAULT_LIMIT, help="max scenarios to load on first run"
        )
        sp.add_argument(
            "--sweep-axis",
            default=DEFAULT_SWEEP_AXIS,
            help="which knob to vary along the Pareto curve",
        )

    serve = sub.add_parser("serve", help="launch the local web UI (default)")
    add_common(serve)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument(
        "--realtime",
        action="store_true",
        help="stream audio at realtime pace (demo fidelity; off for sweeps)",
    )
    serve.add_argument(
        "--open", dest="open_browser", action="store_true", help="open a browser window"
    )
    serve.add_argument("--no-browser", dest="open_browser", action="store_false")
    serve.set_defaults(open_browser=False)

    run = sub.add_parser("run", help="headless: compute and print the sweep summary")
    add_common(run)

    sweep = sub.add_parser("sweep", help="headless: write the full sweep to JSON")
    add_common(sweep)
    sweep.add_argument("--out", default="sweep.json")

    return p


def _build_state(args):
    """Bootstrap: load scenarios, build (cached) per-frame signals, assemble app state."""
    from .cache import SignalCache
    from .detectors.registry import create as make_detector
    from .harness import run_detector
    from .metrics import MetricsEngine
    from .scenarios.registry import create as make_loader
    from .server.app import AppState

    loader = make_loader(args.loader)
    print(
        f"Loading scenarios via '{args.loader}' (language={args.language}, limit={args.limit}) ..."
    )
    scenarios = list(loader.load(limit=args.limit, language=args.language))
    if not scenarios:
        raise SystemExit(
            "No scenarios loaded. Try --dataset fixtures, or check the dataset/language."
        )

    detector = make_detector(args.detector)
    cache = SignalCache(config.cache_dir())
    pace = "realtime" if getattr(args, "realtime", False) else "fast"
    signals = run_detector(scenarios, detector, cache, pace=pace, progress=True)

    return AppState(
        scenarios=scenarios,
        signals=signals,
        detector=detector,
        metrics=MetricsEngine(config.DEFAULT_TOLERANCE_S),
        sweep_axis=args.sweep_axis,
        dataset=args.loader,
        language=args.language,
    )


def _sweep_points(state, sweep_axis):
    from .sweep import build_grid

    grid = build_grid(state.detector, sweep_axis, state.detector.default_params())
    return state.metrics.sweep(state.signals, state.scenarios, state.detector, grid)


def cmd_serve(args) -> int:
    import threading
    import webbrowser

    import uvicorn

    from .server.app import create_app

    state = _build_state(args)
    app = create_app(state)
    url = f"http://127.0.0.1:{args.port}"
    print(
        f"\nturntune serving at {url}  "
        f"({len(state.scenarios)} scenarios, detector={state.detector.name})\n"
    )
    if args.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def cmd_run(args) -> int:
    state = _build_state(args)
    eng = state.metrics
    pts = _sweep_points(state, args.sweep_axis)
    print(f"\nlatency-vs-cutoff curve (sweep over {args.sweep_axis}):")
    for p in pts:
        lat = "n/a" if p.p50_latency_s != p.p50_latency_s else f"{p.p50_latency_s:.2f}s"
        cut, nof = p.cutoff_rate * 100, p.no_fire_rate * 100
        print(
            f"  {args.sweep_axis}={p.params[args.sweep_axis]:.2f} -> "
            f"cutoff={cut:4.0f}%  p50={lat}  no_fire={nof:3.0f}%"
        )
    print("\noperating points:")
    for b in (0.05, 0.10, 0.20, 0.30):
        lat = eng.latency_at_cutoff(pts, b)
        shown = f"{lat:.2f}s" if lat is not None else "unreachable"
        print(f"  latency @ <= {int(b * 100):>2}% cutoff: {shown}")
    return 0


def cmd_sweep(args) -> int:
    state = _build_state(args)
    eng = state.metrics
    pts = _sweep_points(state, args.sweep_axis)
    out = {
        "detector": state.detector.name,
        "dataset": args.loader,
        "language": args.language,
        "n_scenarios": len(state.scenarios),
        "sweep_axis": args.sweep_axis,
        "points": [
            {
                "params": p.params,
                "cutoff_rate": p.cutoff_rate,
                "p50_latency_s": None if p.p50_latency_s != p.p50_latency_s else p.p50_latency_s,
                "p90_latency_s": None if p.p90_latency_s != p.p90_latency_s else p.p90_latency_s,
                "no_fire_rate": p.no_fire_rate,
                "n": p.n,
            }
            for p in pts
        ],
        "summary": {
            str(int(b * 100)): eng.latency_at_cutoff(pts, b) for b in (0.05, 0.10, 0.20, 0.30)
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote sweep for {len(pts)} points -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "serve"
    if command == "serve" and not hasattr(args, "port"):
        # `turntune` with no subcommand -> serve with defaults
        args = build_parser().parse_args(["serve"])
    return {"serve": cmd_serve, "run": cmd_run, "sweep": cmd_sweep}[command](args)


if __name__ == "__main__":
    raise SystemExit(main())
