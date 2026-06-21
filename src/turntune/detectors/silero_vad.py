"""The shipped default detector: Silero VAD via onnxruntime (no torch).

extract() buffers incoming 20ms/320-sample frames into the 512-sample windows Silero
v5 requires at 16kHz, runs the stateful ONNX RNN (state shape (2,1,128)), and maps
the speech probability back onto the 20ms grid (holding the most-recent value between
windows). decide() delegates to the shared silence-hangover policy.

Knob mapping (spec knob -> param -> eot-bench's policy name):
  confidence threshold      -> speech_threshold  (= eot-bench `threshold`)
  silence / min EOT delay   -> min_silence_s      (= `action_delay`; primary sweep axis)
  (optional) force timeout   -> timeout_s          (= eot-bench `timeout`)
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .. import config
from ..types import SR, Axis, EotDecision, FrameSignal, PolicyParams
from .registry import register

WINDOW = 512  # Silero v5 requires exactly 512 samples @ 16 kHz
CONTEXT = 64  # v5 also prepends the previous chunk's last 64 samples (fed 576 total)


def ensure_silero_onnx(cache_root: Path | None = None) -> Path:
    """Return the path to the Silero ONNX model, downloading + sha-checking on first run."""
    cache_root = Path(cache_root) if cache_root else config.cache_dir()
    path = cache_root / "models" / "silero_vad.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Silero VAD model -> {path} ...", file=sys.stderr)
        with urllib.request.urlopen(config.SILERO_ONNX_URL, timeout=120) as resp:
            data = resp.read()
        path.write_bytes(data)
    if config.SILERO_ONNX_SHA256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config.SILERO_ONNX_SHA256:
            print(
                f"WARNING: Silero model sha256 {digest} != pinned "
                f"{config.SILERO_ONNX_SHA256} (upstream may have changed).",
                file=sys.stderr,
            )
    return path


@register("silero-vad")
class SileroVadDetector:
    name = "silero-vad"
    version = "v5-onnx"
    frame_ms = 20

    def __init__(self, onnx_path: str | None = None, cache_root: Path | None = None):
        import onnxruntime as ort

        self.onnx_path = onnx_path or str(ensure_silero_onnx(cache_root))
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def default_params(self) -> PolicyParams:
        return {k: ax.default for k, ax in self.param_space().items()}

    def param_space(self) -> dict[str, Axis]:
        return {
            "speech_threshold": Axis(
                0.1,
                0.9,
                0.05,
                0.5,
                "Confidence (speech prob)",
                "VAD probability at/above which a frame counts as speech",
            ),
            "min_silence_s": Axis(
                0.1,
                1.5,
                0.05,
                0.6,
                "Silence before EOT",
                "Trailing silence required to declare end-of-turn (action_delay)",
            ),
            "timeout_s": Axis(
                0.0,
                5.0,
                0.5,
                0.0,
                "Force EOT timeout (0=off)",
                "Force end-of-turn after this much silence; 0 disables",
            ),
        }

    def extract(self, frames: Iterable[np.ndarray]) -> FrameSignal:
        """Stream 20ms frames -> one speech probability per frame (causal).

        Silero v5 expects each 512-sample window prepended with the previous window's
        last 64 samples (576 total); without that context it sees no speech.
        """
        buf = np.zeros(0, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(CONTEXT, dtype=np.float32)
        sr = np.array(SR, dtype=np.int64)
        probs: list[float] = []
        last = 0.0
        for frame in frames:
            buf = np.concatenate([buf, np.asarray(frame, dtype=np.float32)])
            while len(buf) >= WINDOW:
                chunk = buf[:WINDOW]
                buf = buf[WINDOW:]
                x = np.concatenate([context, chunk])[None, :]  # (1, 576)
                out, state = self._sess.run(
                    ["output", "stateN"],
                    {"input": x, "state": state, "sr": sr},
                )
                context = chunk[-CONTEXT:]
                last = float(out[0, 0])
            probs.append(last)  # hold the most-recent window prob between windows
        return FrameSignal(
            "", self.name, self.version, self.frame_ms, np.asarray(probs, dtype=np.float32)
        )

    def decide(self, signal: FrameSignal, params: PolicyParams) -> EotDecision:
        from ..policy import silence_hangover

        return silence_hangover(signal, params)
