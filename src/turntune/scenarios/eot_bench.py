"""Default loader: LiveKit eot-bench from Hugging Face -> normalized Scenarios.

Verified against the live `livekit/eot-bench-data` dataset:
  - one config per language ("en", "de", ...); the only split is "validation"
  - each row: id, audio (Audio, 16 kHz), language, duration,
    silence_spans[{start,end}], words[{start,end,word}], messages[{role,content}]
  - GROUND TRUTH (LiveKit's own definition): "The final silence span is the true end
    of the user's turn; every earlier silence span is a mid-turn hesitation."
    => true_eot_s = start of the last silence span; earlier spans are `hold`.

Audio is read with decoding turned OFF and decoded here via soundfile, so turntune
never pulls in a torch/torchcodec audio backend.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .. import audio as audio_mod
from .. import config
from ..types import SR, Scenario, SilenceSpan
from .registry import register


def spans_and_eot(
    silence_spans: list[dict],
    words: list[dict],
    duration: float | None,
) -> tuple[list[SilenceSpan], float]:
    """Map raw eot-bench fields to (labelled spans, true_eot_s).

    Pure and network-free so it can be unit-tested on a mocked row.
    """
    spans: list[SilenceSpan] = []
    if silence_spans:
        last = len(silence_spans) - 1
        for i, s in enumerate(silence_spans):
            spans.append(
                SilenceSpan(
                    start_s=float(s["start"]),
                    end_s=float(s["end"]),
                    label="eot" if i == last else "hold",
                )
            )
        true_eot_s = spans[-1].start_s
    else:
        # No labelled pause: the turn ends at the last word (or the audio end).
        true_eot_s = float(words[-1]["end"]) if words else float(duration or 0.0)
    return spans, true_eot_s


def scenario_from_row(row: dict, audio, sample_rate: int, wav_path: str) -> Scenario:
    """Assemble a Scenario from an eot-bench row plus already-decoded audio."""
    spans, true_eot_s = spans_and_eot(
        row.get("silence_spans") or [],
        row.get("words") or [],
        row.get("duration"),
    )
    meta = {
        "language": row.get("language"),
        "duration": row.get("duration"),
        "words": row.get("words") or [],
        "messages": row.get("messages") or [],
    }
    return Scenario(
        id=row["id"],
        audio=audio,
        sample_rate=sample_rate,
        spans=spans,
        true_eot_s=true_eot_s,
        wav_path=wav_path,
        meta=meta,
    )


@register("eot-bench")
class EotBenchLoader:
    name = "eot-bench"

    def __init__(self, cache_root: Path | None = None, dataset_id: str | None = None):
        self.cache_root = Path(cache_root) if cache_root else config.cache_dir()
        self.dataset_id = dataset_id or config.DATASET_ID

    def load(
        self,
        limit: int | None = None,
        language: str = "en",
        split: str = "validation",
    ) -> Iterable[Scenario]:
        # Imported lazily so `import turntune` stays cheap and CI (fixtures only)
        # never needs `datasets`.
        from datasets import Audio, load_dataset

        wav_dir = self.cache_root / "wavs" / "eot-bench" / language
        wav_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(self.dataset_id, language, split=split, streaming=True)
        # decode=False -> rows carry raw {bytes,path}; we decode via soundfile.
        ds = ds.cast_column("audio", Audio(decode=False))

        count = 0
        for row in ds:
            if limit is not None and count >= limit:
                break
            a = row["audio"]
            if a.get("bytes") is not None:
                arr, sr = audio_mod.decode_bytes(a["bytes"])
            else:
                arr, sr = audio_mod.load_wav(a["path"])
            arr = audio_mod.resample_16k_mono(arr, sr)

            wav_path = wav_dir / f"{row['id']}.wav"
            if not wav_path.exists():
                audio_mod.write_pcm_wav(str(wav_path), arr, SR)

            yield scenario_from_row(row, audio=arr, sample_rate=SR, wav_path=str(wav_path))
            count += 1
