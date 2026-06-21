"""FastAPI app: serves the static UI and a small JSON API on a single port.

Endpoints:
  GET  /api/meta        detector param_space (-> sliders), scenario count, status
  POST /api/sweep       {params, sweep_axis} -> grid points + pareto + summary
  POST /api/score       {params} -> per-scenario counts + caught-cutoff list
  GET  /api/audio/{id}  wav for browser playback (Range-enabled via FileResponse)
  GET  /                static single-page UI (no build step)

Cached signals are held in memory, so /sweep and /score round-trip in milliseconds and
the curve updates live as sliders move.
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


def _nn(x: float | None) -> float | None:
    """NaN/None -> None (valid JSON), else round for compact payloads."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 4)


@dataclass
class AppState:
    scenarios: list[Scenario]
    signals: dict[str, FrameSignal]
    detector: Detector
    metrics: MetricsEngine
    sweep_axis: str = config.DEFAULT_SWEEP_AXIS
    dataset: str = "eot-bench"
    language: str = config.DEFAULT_LANGUAGE
    _by_id: dict[str, Scenario] = field(default_factory=dict)

    def __post_init__(self):
        self._by_id = {s.id: s for s in self.scenarios}

    def duration(self, sc: Scenario) -> float:
        d = sc.meta.get("duration")
        return float(d) if d else len(sc.audio) / sc.sample_rate


class ScoreBody(BaseModel):
    params: dict


class SweepBody(BaseModel):
    params: dict
    sweep_axis: str | None = None


def create_app(state: AppState):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="turntune")
    det = state.detector
    eng = state.metrics

    @app.get("/api/meta")
    def meta():
        return {
            "detector": det.name,
            "dataset": state.dataset,
            "language": state.language,
            "n_scenarios": len(state.scenarios),
            "sweep_axis": state.sweep_axis,
            "cutoff_budgets": CUTOFF_BUDGETS,
            "defaults": det.default_params(),
            "axes": [
                {
                    "name": name,
                    "lo": ax.lo,
                    "hi": ax.hi,
                    "step": ax.step,
                    "default": ax.default,
                    "label": ax.label,
                    "help": ax.help,
                }
                for name, ax in det.param_space().items()
            ],
        }

    @app.post("/api/sweep")
    def sweep(body: SweepBody):
        axis = body.sweep_axis or state.sweep_axis
        grid = build_grid(det, axis, body.params)
        pts = eng.sweep(state.signals, state.scenarios, det, grid)
        front = eng.pareto(pts)
        current = eng._aggregate(
            body.params, eng.score_all(state.signals, state.scenarios, det, body.params)
        )
        return {
            "sweep_axis": axis,
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
            "summary": {
                str(int(b * 100)): _nn(eng.latency_at_cutoff(pts, b)) for b in CUTOFF_BUDGETS
            },
        }

    @app.post("/api/score")
    def score(body: ScoreBody):
        results = eng.score_all(state.signals, state.scenarios, det, body.params)
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
