"""Semantic (transcript-based) end-of-turn detector — a content-aware counterpart to
the silence-based Silero VAD baseline.

Model: `anyreach-ai/semantic-turn-taking` (Apache-2.0), a Qwen2.5-0.5B-Instruct
fine-tune that predicts a turn-taking *action* from the conversation transcript:
`start_speaking` (user is done -> respond) vs `continue_listening` (user is mid-
utterance -> keep waiting). We use P(start_speaking) as the end-of-turn probability.

(LiveKit's turn-detector is the obvious choice, but its model license forbids use
outside the LiveKit Agents framework, so it can't ship in an Apache-2.0 tool. This
model is the closest openly-licensed, transcript-based equivalent.)

Mirrors the Silero extract/decide split:
  extract(): runs the transformer once per growing-transcript prefix (the EXPENSIVE
    pass), caching a per-frame signal: P(EOT) during silence gaps, a -1 sentinel while
    a word is being spoken.
  decide(): a cheap, pure content-gated silence-hangover — fire when trailing silence
    (from the word timestamps) >= min_silence_s AND P(EOT) >= eot_threshold.

Transcript source is eot-bench's own gold `words` (with timestamps) fed incrementally
by time — no live STT dependency in v0. Heavy deps (transformers, torch) live behind
the optional `turntune[semantic]` extra; importing this module does NOT import them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .. import config
from ..types import Axis, EotDecision, FrameSignal, PolicyParams, Scenario
from .registry import register

MODEL_ID = "anyreach-ai/semantic-turn-taking"
ACTIONS = ["start_speaking", "continue_listening", "start_listening", "continue_speaking"]
_INSTALL_HINT = (
    "the 'semantic' detector needs extra deps: pip install 'turntune[semantic]' "
    "(transformers + torch)"
)


def _prefix_text(words: list[dict], k: int) -> str:
    return " ".join(w["word"] for w in words[:k]).strip()


@register("semantic-turn")
class SemanticTurnDetector:
    name = "semantic-turn"
    version = "anyreach-v1"
    frame_ms = 20

    def __init__(
        self,
        cache_root=None,
        *,
        use_history: bool = True,
        max_prompt_tokens: int = 512,
        batch_size: int = 8,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(_INSTALL_HINT) from e

        self._torch = torch
        self.use_history = use_history
        self.max_prompt_tokens = max_prompt_tokens
        self.batch_size = batch_size
        hf_cache = str((cache_root or config.cache_dir()) / "hf")

        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=hf_cache)
        self.tok.padding_side = "left"  # so logits[:, -1] is the real last token per row
        self.tok.truncation_side = "left"  # keep the end (current turn + <|predict|>)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=hf_cache, torch_dtype=torch.float32
        )
        self.model.eval()
        self._action_ids = [self.tok.convert_tokens_to_ids(f"<|{a}|>") for a in ACTIONS]

    def default_params(self) -> PolicyParams:
        return {k: ax.default for k, ax in self.param_space().items()}

    def param_space(self) -> dict[str, Axis]:
        return {
            "eot_threshold": Axis(
                0.1,
                0.9,
                0.05,
                0.5,
                "EoT confidence",
                "P(end-of-turn) from the transcript required to end the turn",
            ),
            "min_silence_s": Axis(
                0.1,
                1.5,
                0.05,
                0.6,
                "Silence before EOT",
                "Trailing silence (from word timing) before acting",
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

    # ---- expensive pass: P(EOT) per transcript prefix -> packed per-frame signal ----
    def _prompt(self, history: list[dict], current_text: str) -> str:
        s = ""
        if self.use_history:
            for m in history:
                s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        s += f"<|im_start|>user\n{current_text}<|im_end|>\n<|predict|>"
        return s

    def _eot_probs(self, history: list[dict], words: list[dict]) -> np.ndarray:
        """P(start_speaking) for each prefix of 1..N words (batched transformer forward)."""
        torch = self._torch
        prompts = [self._prompt(history, _prefix_text(words, k)) for k in range(1, len(words) + 1)]
        out: list[float] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            enc = self.tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
            )
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]  # (b, vocab)
            action_logits = logits[:, self._action_ids]  # (b, 4)
            probs = torch.softmax(action_logits, dim=-1)[:, 0]  # P(start_speaking)
            out.extend(probs.tolist())
        return np.asarray(out, dtype=np.float32)

    def extract(
        self, frames: Iterable[np.ndarray], scenario: Scenario | None = None
    ) -> FrameSignal:
        if scenario is None:
            raise ValueError("semantic-turn detector requires a Scenario (transcript words)")
        # Consume frames to count them (and honor realtime pacing); audio itself unused.
        n_frames = sum(1 for _ in frames)
        dt = self.frame_ms / 1000.0
        words = scenario.meta.get("words") or []
        history = scenario.meta.get("messages") or []

        signal = np.full(n_frames, -1.0, dtype=np.float32)  # default: treated as speech
        if not words:
            # No transcript -> all silence with P=0 (never fires); leave as needed.
            signal[:] = 0.0
            return FrameSignal("", self.name, self.version, self.frame_ms, signal)

        pe = self._eot_probs(history, words)  # P(EOT) for prefixes of 1..N words
        pe_by_k = np.concatenate([[0.0], pe]).astype(np.float32)  # index by #words seen

        word_starts = np.array([w["start"] for w in words], dtype=np.float64)
        word_ends = np.array([w["end"] for w in words], dtype=np.float64)
        frame_start = np.arange(n_frames) * dt

        # speech frame = some word overlaps [frame_start, frame_start + dt)
        is_speech = np.zeros(n_frames, dtype=bool)
        for ws, we in zip(word_starts, word_ends, strict=True):
            lo = max(0, int(math.floor(ws / dt)))
            hi = min(n_frames, int(math.ceil(we / dt)))
            is_speech[lo:hi] = True

        # words completed by the start of each frame -> which prefix's P(EOT) applies
        k_done = np.searchsorted(word_ends, frame_start, side="right")
        silence_vals = pe_by_k[k_done]  # P(EOT) for the transcript seen so far
        signal = np.where(is_speech, -1.0, silence_vals).astype(np.float32)
        return FrameSignal("", self.name, self.version, self.frame_ms, signal)

    # ---- cheap pass: content-gated silence hangover (pure, replayed across the sweep) ----
    def decide(self, signal: FrameSignal, params: PolicyParams) -> EotDecision:
        from ..policy import content_gated_hangover

        return content_gated_hangover(signal, params)


__all__ = ["SemanticTurnDetector", "MODEL_ID"]
