"""Metrics on synthetic signals -> known cutoff/latency/pareto values."""

from __future__ import annotations

import numpy as np

from conftest import FakeDetector, make_signal
from turntune.metrics import MetricsEngine
from turntune.policy import silence_hangover
from turntune.types import Scenario, SweepPoint


def _speech_then_silence(speech_s: float, total_s: float, dt: float = 0.02) -> np.ndarray:
    n = int(round(total_s / dt))
    n_speech = int(round(speech_s / dt))
    mask = np.zeros(n, dtype=np.float32)
    mask[:n_speech] = 1.0
    return mask


def _scenario(true_eot_s: float) -> Scenario:
    return Scenario(
        id="s",
        audio=np.zeros(1, np.float32),
        sample_rate=16000,
        spans=[],
        true_eot_s=true_eot_s,
        wav_path="s.wav",
    )


def test_cutoff_classification():
    # Speech ends at 2.0s; the real turn ends at 3.0s -> firing at ~2.3s is a cutoff.
    sig = make_signal(_speech_then_silence(2.0, 4.0))
    det = FakeDetector()
    res = MetricsEngine(tol_s=0.1).score_one(
        sig, _scenario(3.0), det, {"speech_threshold": 0.5, "min_silence_s": 0.3}
    )
    assert res.klass == "cutoff"
    assert res.latency_s is None


def test_latency_for_correct_decision():
    # Speech ends at 2.0s and that IS the true end -> fire at 2.0 + min_silence.
    sig = make_signal(_speech_then_silence(2.0, 4.0))
    det = FakeDetector()
    res = MetricsEngine(tol_s=0.1).score_one(
        sig, _scenario(2.0), det, {"speech_threshold": 0.5, "min_silence_s": 0.3}
    )
    assert res.klass == "correct"
    assert abs(res.latency_s - 0.30) < 1e-6


def test_missed_when_never_fires():
    sig = make_signal(np.ones(50, dtype=np.float32))  # all speech, never goes silent
    det = FakeDetector()
    res = MetricsEngine().score_one(
        sig, _scenario(1.0), det, {"speech_threshold": 0.5, "min_silence_s": 0.3}
    )
    assert res.klass == "missed"


def test_sweep_trades_cutoffs_for_latency():
    # One clip with a 0.5s mid-turn hold then the real end at 3.0s.
    dt = 0.02
    mask = np.zeros(int(4.0 / dt), np.float32)
    mask[: int(1.0 / dt)] = 1.0  # speech 0.0-1.0
    mask[int(1.5 / dt) : int(3.0 / dt)] = 1.0  # speech 1.5-3.0 (a 0.5s hold at 1.0-1.5)
    sig = make_signal(mask)
    sc = _scenario(3.0)
    det = FakeDetector()
    eng = MetricsEngine(tol_s=0.1)
    signals = {"s": sig}

    # Short silence threshold fires in the 0.5s hold -> cutoff.
    short = eng._aggregate(
        {"min_silence_s": 0.3},
        eng.score_all(signals, [sc], det, {"speech_threshold": 0.5, "min_silence_s": 0.3}),
    )
    assert short.cutoff_rate == 1.0
    # Long enough to clear the hold -> correct, with latency ~= min_silence.
    long = eng._aggregate(
        {"min_silence_s": 0.6},
        eng.score_all(signals, [sc], det, {"speech_threshold": 0.5, "min_silence_s": 0.6}),
    )
    assert long.cutoff_rate == 0.0
    assert abs(long.p50_latency_s - 0.6) < 1e-6


def test_pareto_and_latency_at_cutoff():
    pts = [
        SweepPoint(
            {"min_silence_s": 0.1},
            cutoff_rate=0.50,
            p50_latency_s=0.1,
            p90_latency_s=0.1,
            no_fire_rate=0.0,
            n=10,
        ),
        SweepPoint(
            {"min_silence_s": 0.3},
            cutoff_rate=0.20,
            p50_latency_s=0.3,
            p90_latency_s=0.3,
            no_fire_rate=0.0,
            n=10,
        ),
        SweepPoint(
            {"min_silence_s": 0.6},
            cutoff_rate=0.05,
            p50_latency_s=0.6,
            p90_latency_s=0.6,
            no_fire_rate=0.0,
            n=10,
        ),
        SweepPoint(
            {"min_silence_s": 0.9},
            cutoff_rate=0.05,
            p50_latency_s=0.9,
            p90_latency_s=0.9,
            no_fire_rate=0.0,
            n=10,
        ),
    ]
    front = MetricsEngine.pareto(pts)
    # The 0.9 point is dominated (same cutoff as 0.6 but higher latency).
    assert {p.params["min_silence_s"] for p in front} == {0.1, 0.3, 0.6}
    # "latency at ≤10% cutoff" -> the 0.6 operating point.
    assert MetricsEngine.latency_at_cutoff(pts, 0.10) == 0.6
    assert MetricsEngine.latency_at_cutoff(pts, 0.0) is None


def _sp(cutoff, lat, no_fire):
    return SweepPoint(
        {}, cutoff_rate=cutoff, p50_latency_s=lat, p90_latency_s=lat, no_fire_rate=no_fire, n=10
    )


def test_best_point_bounds_no_fire():
    # A tempting low-cutoff/low-latency point that only achieves it by NOT firing (20%
    # no-fire) vs an honest point at higher latency but low no-fire.
    pts = [_sp(0.05, 0.30, 0.20), _sp(0.08, 0.60, 0.02)]
    # cutoff-only: the gamed 0.30s point wins...
    assert MetricsEngine.latency_at_cutoff(pts, 0.10) == 0.30
    # ...but once no-fire is bounded <=5%, it is excluded -> the honest 0.60s point.
    assert MetricsEngine.latency_at(pts, 0.10, 0.05) == 0.60
    bp = MetricsEngine.best_point(pts, 0.10, 0.05)
    assert bp.p50_latency_s == 0.60 and bp.no_fire_rate <= 0.05


def test_silence_hangover_timeout_forces_fire():
    # 10 speech frames then 40 silence frames (0.8s of trailing silence).
    sig = make_signal(np.array([1.0] * 10 + [0.0] * 40, dtype=np.float32))
    base = {"speech_threshold": 0.5, "min_silence_s": 1.0}  # needs 1.0s silence -> not enough
    assert silence_hangover(sig, base).fired is False
    # timeout forces an end-of-turn after 0.4s of silence (20 frames after last speech).
    d = silence_hangover(sig, {**base, "timeout_s": 0.4})
    assert d.fired is True
    assert abs(d.eot_s - 0.60) < 1e-6  # last speech idx 9 + 20 frames -> idx 29 -> 0.60s
