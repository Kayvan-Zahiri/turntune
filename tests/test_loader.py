"""Loader tests — the eot-bench field mapping and the offline fixtures loader.

Both run fully offline (no Hugging Face download).
"""

from __future__ import annotations

import numpy as np

from turntune.scenarios.eot_bench import scenario_from_row, spans_and_eot
from turntune.scenarios.fixtures import FixturesLoader
from turntune.types import SR


def test_eot_bench_row_maps_to_scenario():
    # A real-shaped eot-bench row: two mid-turn pauses, then the trailing silence.
    row = {
        "id": "en__875_82__user_turn_007",
        "language": "en",
        "duration": 14.0,
        "silence_spans": [
            {"start": 6.3, "end": 7.3},
            {"start": 9.7, "end": 9.9},
            {"start": 12.5, "end": 14.0},
        ],
        "words": [{"start": 12.24, "end": 12.48, "word": "day."}],
        "messages": [{"role": "user", "content": "..."}],
    }
    spans, eot = spans_and_eot(row["silence_spans"], row["words"], row["duration"])

    # Final span is the true end of turn; earlier spans are holds.
    assert [s.label for s in spans] == ["hold", "hold", "eot"]
    assert eot == 12.5  # start of the last silence span

    sc = scenario_from_row(row, audio=np.zeros(10, np.float32), sample_rate=SR, wav_path="x.wav")
    assert sc.id == row["id"]
    assert sc.true_eot_s == 12.5
    assert sc.meta["messages"] == row["messages"]


def test_spans_and_eot_without_silence_spans():
    # No labelled pause -> turn ends at the last word.
    spans, eot = spans_and_eot([], [{"start": 1.0, "end": 1.5, "word": "ok"}], 2.0)
    assert spans == []
    assert eot == 1.5


def test_fixtures_load_offline():
    scenarios = list(FixturesLoader().load())
    assert len(scenarios) >= 3
    for sc in scenarios:
        assert sc.sample_rate == SR
        assert sc.audio.dtype == np.float32
        assert sc.audio.ndim == 1
        # exactly one eot span, and true_eot_s sits at its start
        eot_spans = [s for s in sc.spans if s.label == "eot"]
        assert len(eot_spans) == 1
        assert abs(sc.true_eot_s - eot_spans[0].start_s) < 1e-6
