"""Semantic detector tests.

The content-gated decision is tested offline via policy.content_gated_hangover (no
model or heavy deps needed — runs in CI). The extract/model path is a smoke test,
skipped unless the optional `[semantic]` extra (transformers + torch) is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from turntune.policy import content_gated_hangover
from turntune.types import FrameSignal


def _sig(values) -> FrameSignal:
    # packed semantic signal: <0 = speech (sentinel), [0,1] = P(EOT) during a silence gap
    return FrameSignal("s", "semantic-turn", "v", 20, np.asarray(values, dtype=np.float32))


def test_content_gate_blocks_when_transcript_incomplete():
    # speech, then a long pause whose transcript still looks INCOMPLETE (P=0.1)
    sig = _sig([-1.0] * 10 + [0.1] * 60)
    d = content_gated_hangover(sig, {"eot_threshold": 0.5, "min_silence_s": 0.2})
    assert d.fired is False  # the content gate prevents a cutoff a VAD would make


def test_content_gate_fires_when_transcript_complete():
    # speech, then a pause whose transcript looks COMPLETE (P=0.9)
    sig = _sig([-1.0] * 10 + [0.9] * 60)
    d = content_gated_hangover(sig, {"eot_threshold": 0.5, "min_silence_s": 0.2})
    assert d.fired is True
    # last speech frame = idx 9; k_min = ceil(0.2/0.02) = 10 -> fires at idx 19 -> 0.40s
    assert abs(d.eot_s - 0.40) < 1e-6


def test_timeout_forces_fire_when_unconfident():
    sig = _sig([-1.0] * 10 + [0.1] * 120)
    d = content_gated_hangover(sig, {"eot_threshold": 0.5, "min_silence_s": 0.2, "timeout_s": 1.0})
    assert d.fired is True  # backstop fires despite low confidence


def test_no_fire_before_any_speech():
    sig = _sig([0.9] * 50)  # high P but no speech ever occurred -> never fires
    d = content_gated_hangover(sig, {"eot_threshold": 0.5, "min_silence_s": 0.2})
    assert d.fired is False


_HAVE_SEMANTIC = True
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    _HAVE_SEMANTIC = False


@pytest.mark.skipif(not _HAVE_SEMANTIC, reason="optional [semantic] extra not installed")
def test_semantic_extract_smoke():
    from turntune import config
    from turntune.audio import iter_frames
    from turntune.detectors.semantic import SemanticTurnDetector
    from turntune.types import Scenario

    det = SemanticTurnDetector(cache_root=config.cache_dir())
    audio = np.zeros(16_000, np.float32)  # 1 s -> 50 frames
    sc = Scenario(
        id="t",
        audio=audio,
        sample_rate=16_000,
        spans=[],
        true_eot_s=0.8,
        wav_path="t.wav",
        meta={
            "words": [
                {"start": 0.0, "end": 0.4, "word": "Hello"},
                {"start": 0.4, "end": 0.8, "word": "there."},
            ],
            "messages": [],
        },
    )
    sig = det.extract(iter_frames(audio, 16_000, det.frame_ms), sc)
    assert len(sig.speech_prob) == 50
    assert sig.speech_prob.min() >= -1.0 and sig.speech_prob.max() <= 1.0
    assert isinstance(det.decide(sig, det.default_params()).fired, bool)
