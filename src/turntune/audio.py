"""Audio mechanics in one place: decode, resample to 16k mono, frame, and export.

Centralizing resampling here guarantees every timestamp (ground truth, frame index,
EOT decision) lives on the same 16 kHz timeline.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator

import numpy as np
import soundfile as sf
import soxr

from .types import FRAME_MS, SR


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # (n_samples, n_channels) -> mono
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Decode an audio file to float32 in [-1, 1]; return (samples, sample_rate)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return _to_mono_float32(audio), int(sr)


def decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode encoded audio bytes (wav/flac/ogg) to float32 mono; return (samples, sr).

    Used by the eot-bench loader, which reads the dataset's Audio column WITHOUT
    `datasets`' own decoding so we never pull in a torch/torchcodec backend.
    """
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return _to_mono_float32(audio), int(sr)


def resample_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    """Downmix to mono and resample to SR (16 kHz). No-op when already 16k mono."""
    audio = _to_mono_float32(audio)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR).astype(np.float32, copy=False)
    return audio


def iter_frames(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = FRAME_MS,
    *,
    pace: str = "fast",
) -> Iterator[np.ndarray]:
    """Yield consecutive `frame_ms` frames (320 samples @ 16k), zero-padding the tail.

    pace="fast" yields as fast as possible (used for sweeps); pace="realtime" sleeps
    one frame duration between yields to mimic a live mic. EOT timestamps derive from
    frame index, so both paces produce identical decisions — pacing is for demo
    fidelity only.
    """
    if sr != SR:
        audio = resample_16k_mono(audio, sr)
    audio = np.asarray(audio, dtype=np.float32)
    n = frame_ms * SR // 1000  # samples per frame (320)
    total = (len(audio) + n - 1) // n
    dt = frame_ms / 1000.0
    for i in range(total):
        chunk = audio[i * n : (i + 1) * n]
        if len(chunk) < n:  # zero-pad the final partial frame
            chunk = np.concatenate([chunk, np.zeros(n - len(chunk), dtype=np.float32)])
        if pace == "realtime":
            time.sleep(dt)
        yield chunk


def write_pcm_wav(path: str, audio: np.ndarray, sr: int = SR) -> None:
    """Write 16-bit PCM wav for browser playback."""
    sf.write(path, np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")
