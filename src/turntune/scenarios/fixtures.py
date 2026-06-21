"""Offline/CI loader over tests/data — zero network.

Reads the bundled tiny synthetic wavs + labels.json into Scenarios. Powers offline CI
and lets a stranger without Hugging Face access still exercise the full pipeline via
`turntune serve --dataset fixtures`.

The fixture audio is synthetic (modulated noise + silence) with hand-labelled spans;
it's a structural smoke test, not realistic speech. The real evaluation uses eot-bench.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .. import audio as audio_mod
from ..types import SR, Scenario, SilenceSpan
from .registry import register

# Bundled in-package fixtures: src/turntune/data/fixtures (shipped in the wheel).
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"


@register("fixtures")
class FixturesLoader:
    name = "fixtures"

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    def load(
        self,
        limit: int | None = None,
        language: str = "en",
        split: str = "validation",
    ) -> Iterable[Scenario]:
        labels_path = self.data_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"fixtures not found at {labels_path}. Generate them with "
                f"`python src/turntune/data/_generate.py`, or use `--dataset eot-bench`."
            )
        entries = json.loads(labels_path.read_text()).get("scenarios", [])
        for entry in entries[: limit if limit is not None else None]:
            wav_path = self.data_dir / entry["wav"]
            arr, sr = audio_mod.load_wav(str(wav_path))
            arr = audio_mod.resample_16k_mono(arr, sr)
            spans = [
                SilenceSpan(start_s=float(s["start"]), end_s=float(s["end"]), label=s["label"])
                for s in entry.get("silence_spans", [])
            ]
            eot_spans = [s for s in spans if s.label == "eot"]
            true_eot_s = float(
                entry.get("true_eot_s") or (eot_spans[-1].start_s if eot_spans else 0.0)
            )
            yield Scenario(
                id=entry["id"],
                audio=arr,
                sample_rate=SR,
                spans=spans,
                true_eot_s=true_eot_s,
                wav_path=str(wav_path),
                meta={
                    "language": entry.get("language", language),
                    "duration": entry.get("duration"),
                },
            )
