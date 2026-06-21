"""Silero adapter: 20ms->512-sample buffering + extract/decide determinism.

Skipped when the ONNX model isn't already cached (e.g. offline CI), so the suite
stays fully offline by default. Run `turntune` once (or these tests locally with a
network connection) to populate the model cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from turntune import config
from turntune.detectors.silero_vad import SileroVadDetector

_MODEL = config.cache_dir() / "models" / "silero_vad.onnx"
pytestmark = pytest.mark.skipif(
    not _MODEL.exists(), reason="Silero ONNX not cached; run turntune once to download it"
)


def _audio() -> np.ndarray:
    rng = np.random.default_rng(0)
    speech = rng.normal(0, 1, 16000).astype(np.float32) * 0.3  # 1.0s noisy "speech"
    silence = np.zeros(8000, np.float32)  # 0.5s silence
    return np.concatenate([speech, silence])


def test_extract_shape_and_range():
    det = SileroVadDetector()
    from turntune.audio import iter_frames

    sig = det.extract(iter_frames(_audio(), 16000, det.frame_ms))
    # one probability per 20ms frame over the 1.5s clip
    assert sig.speech_prob.ndim == 1
    assert len(sig.speech_prob) == int(round(1.5 / 0.02))
    assert float(sig.speech_prob.min()) >= 0.0
    assert float(sig.speech_prob.max()) <= 1.0


def test_extract_is_deterministic():
    det = SileroVadDetector()
    from turntune.audio import iter_frames

    a = det.extract(iter_frames(_audio(), 16000, det.frame_ms))
    b = det.extract(iter_frames(_audio(), 16000, det.frame_ms))
    assert np.array_equal(a.speech_prob, b.speech_prob)
