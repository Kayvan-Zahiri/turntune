"""The metrics engine: score decisions against ground truth and build the curve.

This is where cutoff rate, endpointing latency, the Pareto frontier, and the
"latency at ≤X% cutoff" headline come from. See docs/metrics.md for precise
definitions. Latency is measured on the AUDIO timeline (conversational silence after
the true end of turn), matching eot-bench — not wall-clock compute time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .detectors.base import Detector
from .types import FrameSignal, PolicyParams, Scenario, ScenarioResult, SweepPoint


class MetricsEngine:
    def __init__(self, tol_s: float = 0.2):
        # A fire earlier than (true_eot_s - tol) is a cutoff; tol is a small grace
        # margin around the boundary between the last hold and the trailing silence.
        # Matches config.DEFAULT_TOLERANCE_S (the value the CLI/UI use).
        self.tol = tol_s

    def score_one(
        self,
        signal: FrameSignal,
        scenario: Scenario,
        detector: Detector,
        params: PolicyParams,
    ) -> ScenarioResult:
        d = detector.decide(signal, params)
        if not d.fired or d.eot_s is None:
            return ScenarioResult(scenario.id, "missed", None, scenario.true_eot_s, None)
        if d.eot_s < scenario.true_eot_s - self.tol:
            return ScenarioResult(scenario.id, "cutoff", d.eot_s, scenario.true_eot_s, None)
        latency = max(0.0, d.eot_s - scenario.true_eot_s)
        return ScenarioResult(scenario.id, "correct", d.eot_s, scenario.true_eot_s, latency)

    def score_all(
        self,
        signals: dict[str, FrameSignal],
        scenarios: Iterable[Scenario],
        detector: Detector,
        params: PolicyParams,
    ) -> list[ScenarioResult]:
        return [self.score_one(signals[s.id], s, detector, params) for s in scenarios]

    def _aggregate(self, params: PolicyParams, results: list[ScenarioResult]) -> SweepPoint:
        n = len(results)
        cutoffs = sum(1 for r in results if r.klass == "cutoff")
        misses = sum(1 for r in results if r.klass == "missed")
        latencies = [r.latency_s for r in results if r.klass == "correct"]
        p50 = float(np.percentile(latencies, 50)) if latencies else math.nan
        p90 = float(np.percentile(latencies, 90)) if latencies else math.nan
        return SweepPoint(
            params=dict(params),
            cutoff_rate=cutoffs / n if n else math.nan,
            p50_latency_s=p50,
            p90_latency_s=p90,
            no_fire_rate=misses / n if n else math.nan,
            n=n,
        )

    def sweep(
        self,
        signals: dict[str, FrameSignal],
        scenarios: Iterable[Scenario],
        detector: Detector,
        grid: list[PolicyParams],
    ) -> list[SweepPoint]:
        """Replay decide() over cached signals for every grid point (the fast loop)."""
        scenarios = list(scenarios)
        return [self._aggregate(p, self.score_all(signals, scenarios, detector, p)) for p in grid]

    @staticmethod
    def pareto(points: list[SweepPoint]) -> list[SweepPoint]:
        """Non-dominated lower-left envelope, minimizing (cutoff_rate, p50 latency).

        Points with no correctly-endpointed scenarios (latency = NaN) can't be placed
        on the curve and are excluded.
        """
        valid = [
            p for p in points if not math.isnan(p.p50_latency_s) and not math.isnan(p.cutoff_rate)
        ]
        front: list[SweepPoint] = []
        for p in valid:
            dominated = any(
                q is not p
                and q.cutoff_rate <= p.cutoff_rate
                and q.p50_latency_s <= p.p50_latency_s
                and (q.cutoff_rate < p.cutoff_rate or q.p50_latency_s < p.p50_latency_s)
                for q in valid
            )
            if not dominated:
                front.append(p)
        front.sort(key=lambda p: (p.cutoff_rate, p.p50_latency_s))
        return front

    @staticmethod
    def best_point(
        points: list[SweepPoint], max_cutoff: float = 1.0, max_no_fire: float = 1.0
    ) -> SweepPoint | None:
        """Lowest-latency operating point within BOTH a cutoff and a no-fire ceiling.

        Bounding no-fire alongside cutoff is what makes the headline ungameable: a policy
        can't lower its cutoff rate just by refusing to take the floor (a no-fire — the
        agent hanging in silence — is a failure too, not a smaller error than a cutoff).
        """
        cands = [
            p
            for p in points
            if not math.isnan(p.p50_latency_s)
            and not math.isnan(p.cutoff_rate)
            and p.cutoff_rate <= max_cutoff
            and p.no_fire_rate <= max_no_fire
        ]
        return min(cands, key=lambda p: p.p50_latency_s) if cands else None

    @staticmethod
    def latency_at(
        points: list[SweepPoint], max_cutoff: float = 1.0, max_no_fire: float = 1.0
    ) -> float | None:
        """Lowest p50 latency within a cutoff AND a no-fire ceiling (None if unreachable)."""
        p = MetricsEngine.best_point(points, max_cutoff, max_no_fire)
        return p.p50_latency_s if p else None

    @staticmethod
    def latency_at_cutoff(points: list[SweepPoint], max_rate: float) -> float | None:
        """Lowest p50 latency at or below `max_rate` cutoff (no no-fire constraint)."""
        return MetricsEngine.latency_at(points, max_rate, 1.0)

    @staticmethod
    def caught_cutoffs(results: list[ScenarioResult]) -> list[ScenarioResult]:
        """The failure-playback list: scenarios classified as cutoff."""
        return [r for r in results if r.klass == "cutoff"]
