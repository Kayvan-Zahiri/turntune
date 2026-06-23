"""FastAPI app: serves the static UI and a small JSON API on a single port.

Endpoints:
  GET  /api/meta        available detectors + their param_spaces, status, no-fire bound
  POST /api/sweep       {detector, params, sweep_axis} -> grid points + pareto + summary
  POST /api/score       {detector, params} -> per-scenario counts + caught-cutoff list
  GET  /api/audio/{id}  wav for browser playback (Range-enabled via FileResponse)
  GET  /                static single-page UI (no build step)

Cached signals are held in memory (one set per loaded detector), so /sweep and /score
round-trip in milliseconds and the curve updates live as sliders move or the detector
is switched.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from .. import config
from ..detectors.base import Detector
from ..metrics import MetricsEngine
from ..sweep import build_grid
from ..types import FrameSignal, Scenario

STATIC_DIR = Path(__file__).parent / "static"
CUTOFF_BUDGETS = [0.05, 0.10, 0.20, 0.30]
# A no-fire (the agent never taking the floor) is a failure too, not a smaller error
# than a cutoff. Operating points above this bound are flagged everywhere.
NO_FIRE_BOUND = 0.05


def _nn(x: float | None) -> float | None:
    """NaN/None -> None (valid JSON), else round for compact payloads."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 4)


@dataclass
class DetectorState:
    detector: Detector
    signals: dict[str, FrameSignal]


@dataclass
class AppState:
    scenarios: list[Scenario]
    detectors: dict[str, DetectorState]  # name -> state (one signal set per detector)
    default_detector: str
    metrics: MetricsEngine
    sweep_axis: str = config.DEFAULT_SWEEP_AXIS
    dataset: str = "eot-bench"
    language: str = config.DEFAULT_LANGUAGE
    no_fire_bound: float = NO_FIRE_BOUND
    _by_id: dict[str, Scenario] = field(default_factory=dict)

    def __post_init__(self):
        self._by_id = {s.id: s for s in self.scenarios}

    def duration(self, sc: Scenario) -> float:
        d = sc.meta.get("duration")
        return float(d) if d else len(sc.audio) / sc.sample_rate

    def resolve(self, name: str | None) -> DetectorState:
        return (
            self.detectors.get(name or self.default_detector)
            or self.detectors[self.default_detector]
        )


def _axes(det: Detector) -> list[dict]:
    return [
        {
            "name": n,
            "lo": ax.lo,
            "hi": ax.hi,
            "step": ax.step,
            "default": ax.default,
            "label": ax.label,
            "help": ax.help,
        }
        for n, ax in det.param_space().items()
    ]


class ScoreBody(BaseModel):
    params: dict
    detector: str | None = None


class SweepBody(BaseModel):
    params: dict
    sweep_axis: str | None = None
    detector: str | None = None


def create_app(state: AppState):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="turntune")
    eng = state.metrics

    @app.get("/api/meta")
    def meta():
        return {
            "dataset": state.dataset,
            "language": state.language,
            "n_scenarios": len(state.scenarios),
            "sweep_axis": state.sweep_axis,
            "cutoff_budgets": CUTOFF_BUDGETS,
            "no_fire_bound": state.no_fire_bound,
            "default_detector": state.default_detector,
            "detectors": {
                name: {"axes": _axes(ds.detector), "defaults": ds.detector.default_params()}
                for name, ds in state.detectors.items()
            },
        }

    @app.post("/api/sweep")
    def sweep(body: SweepBody):
        ds = state.resolve(body.detector)
        det, sigs = ds.detector, ds.signals
        axis = body.sweep_axis or state.sweep_axis
        grid = build_grid(det, axis, body.params)
        pts = eng.sweep(sigs, state.scenarios, det, grid)
        # Pareto over only the operating points that respect the no-fire bound, so the
        # frontier can't be a low-cutoff point that's actually bought by silence.
        within = [p for p in pts if p.no_fire_rate <= state.no_fire_bound]
        front = eng.pareto(within)
        current = eng._aggregate(
            body.params, eng.score_all(sigs, state.scenarios, det, body.params)
        )
        return {
            "sweep_axis": axis,
            "no_fire_bound": state.no_fire_bound,
            "points": [
                {
                    "x": _nn(p.params.get(axis)),
                    "cutoff": _nn(p.cutoff_rate),
                    "p50": _nn(p.p50_latency_s),
                    "p90": _nn(p.p90_latency_s),
                    "no_fire": _nn(p.no_fire_rate),
                }
                for p in pts
            ],
            "pareto": [{"cutoff": _nn(p.cutoff_rate), "p50": _nn(p.p50_latency_s)} for p in front],
            "current": {
                "x": _nn(current.params.get(axis)),
                "cutoff": _nn(current.cutoff_rate),
                "p50": _nn(current.p50_latency_s),
                "p90": _nn(current.p90_latency_s),
                "no_fire": _nn(current.no_fire_rate),
            },
            # headline is no-fire-bounded: best latency at <=X% cutoff AND <=Y% no-fire
            "summary": {
                str(int(b * 100)): _nn(eng.latency_at(pts, b, state.no_fire_bound))
                for b in CUTOFF_BUDGETS
            },
        }

    @app.post("/api/score")
    def score(body: ScoreBody):
        ds = state.resolve(body.detector)
        results = eng.score_all(ds.signals, state.scenarios, ds.detector, body.params)
        counts = {
            k: sum(1 for r in results if r.klass == k) for k in ("correct", "cutoff", "missed")
        }
        cutoffs = []
        for r in eng.caught_cutoffs(results):
            sc = state._by_id[r.scenario_id]
            cutoffs.append(
                {
                    "id": sc.id,
                    "fired_s": _nn(r.eot_s),
                    "true_eot_s": _nn(r.true_eot_s),
                    "duration": _nn(state.duration(sc)),
                    "audio": f"/api/audio/{sc.id}",
                    "spans": [
                        {"start": _nn(s.start_s), "end": _nn(s.end_s), "label": s.label}
                        for s in sc.spans
                    ],
                }
            )
        return {"counts": counts, "n": len(results), "cutoffs": cutoffs}

    @app.get("/api/audio/{scenario_id}")
    def audio(scenario_id: str):
        sc = state._by_id.get(scenario_id)
        if sc is None or not Path(sc.wav_path).exists():
            raise HTTPException(status_code=404, detail="audio not found")
        return FileResponse(sc.wav_path, media_type="audio/wav")

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app
