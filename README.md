# turntune

**See your voice agent's latency-vs-cutoff tradeoff, find the conversations where it talks over people, and dial in the endpointing policy.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
[![CI](https://github.com/Kayvan-Zahiri/turntune/actions/workflows/ci.yml/badge.svg)](https://github.com/Kayvan-Zahiri/turntune/actions/workflows/ci.yml)

![turntune comparing the silence-based Silero VAD against a transcript-based semantic detector — switching detectors live, with cutoff, latency and no-fire shown together and a no-fire-aware curve](docs/img/hero.gif)

> Two detectors over 40 eot-bench conversations. Switch from **silero-vad** (silence) to
> **semantic-turn** (transcript) and tune live: cutoff, latency, and **no-fire** update
> together as equal readouts. On the curve, points within the no-fire budget are blue and
> points whose low cutoff is *bought by silence* (the agent never answering) are greyed —
> so a tempting low-cutoff setting can't hide its cost. Here the semantic model settles on
> a real operating point: **20% cutoff at 0.46s with 3% no-fire**.

---

## Why turn-taking matters

Turn-taking — deciding the moment the user is *done* speaking — is one of the
highest-leverage quality problems in a voice agent. Endpoint **too early** and the
agent talks over people mid-sentence. Endpoint **too late** and every exchange
fills with awkward dead air. The two failure modes pull in opposite directions, so
there's a real tradeoff to navigate.

`turntune` makes that tradeoff **measurable and tunable**. Point it at a turn
detector and it shows you, over a set of real conversations: how often the detector
cuts users off, how much latency it adds, and exactly how that curve shifts when
you change the endpointing policy.

## What you get

- 📈 **The latency-vs-cutoff Pareto curve** for your detector — with **no-fire** held to a
  budget, so a low cutoff can't be bought by the agent simply never answering.
- 🔊 **A playable list of the conversations it cut off** — hear the talk-over for
  yourself instead of guessing.
- 🎚️ **Live tuning knobs** — drag the confidence / silence / timeout sliders and watch the
  curve, the three readouts (**cutoff / latency / no-fire**), and the failure list update.
- 🔀 **Two detectors to compare** — silence-based Silero VAD (default) and a transcript-based
  semantic end-of-turn model — switchable live in the UI.

## 30-second quickstart

```bash
git clone https://github.com/Kayvan-Zahiri/turntune
cd turntune
make run
```

`make run` creates a virtualenv, installs `turntune`, and launches the local web UI.
On first run it downloads the Silero VAD model (~2 MB) and a small subset of
LiveKit's [eot-bench](https://huggingface.co/datasets/livekit/eot-bench-data)
scenarios, then opens **http://localhost:8000**.

No GPU, no API keys, no account. Everything runs locally.

> No Hugging Face access or want to try it instantly offline? `turntune serve --dataset fixtures`
> runs against a tiny bundled scenario set (a structural smoke test — synthetic audio,
> so no real cutoffs; the real evaluation uses eot-bench).

## Reading the results

![Reading the turntune UI: the three readouts (cutoff / latency / no-fire), the no-fire-aware tradeoff curve, the detector selector and tuning sliders, and the caught-cutoff list](docs/img/screenshot.png)

What's on screen (here, the semantic detector at **20% cutoff / 0.46s latency / 3% no-fire** —
that's 1 of 40 turns; see the [comparison](#silence-vs-semantics--an-honest-comparison) note on rounding):

- **Three readouts, equal weight (top).** Cutoff, latency, **and no-fire** — the share of turns
  where the detector *never* takes the floor (the agent hangs in silence). No-fire turns red when
  it exceeds the budget, so you can't read a low cutoff without seeing what it cost.
- **The curve.** Each point is one policy setting: x = cutoff rate, y = latency, **lower-left is
  better**. Points within the no-fire budget are **blue**; points whose low cutoff is *bought by
  silence* are **greyed**, and the frontier line is drawn only within the budget. The headline
  below states the bound explicitly: *best latency @ ≤X% cutoff **and ≤5% no-fire***.
- **The knobs (right).** Pick the **detector** (`silero-vad` / `semantic-turn`), choose the sweep
  axis, and drag the sliders — the curve, readouts, and failure list all recompute live.
- **The caught cutoffs (below).** Every conversation the detector talked over: a timeline (gray =
  mid-turn pause, green = true end of turn, red = where it fired) with audio you can *play* to hear
  the talk-over.

## How it works

```
eot-bench loader ─▶ 20ms / 16kHz audio harness ─▶ Silero VAD (run ONCE, signal cached)
                                                          │
                                  cheap endpointing policy replayed across the sweep
                                                          │
                                   metrics ─▶ latency-vs-cutoff curve + failure playback
```

Each detector is split into two halves: an **expensive `extract()`** that runs the
neural model over the audio exactly once and caches a per-frame signal, and a
**cheap, pure `decide()`** that turns that signal + your tuning knobs into an
end-of-turn decision. Because tuning only re-runs the cheap half over cached
signals, the whole sweep recomputes in milliseconds — so the curve feels live as
you drag a slider. (Audio is streamed in 20 ms frames; decision times come from the
frame index, not the wall clock, so the fast sweep gives the same answer as real-time
streaming.)

The same split powers both shipped detectors through one `Detector` seam: Silero VAD's
`extract()` emits a per-frame speech probability; the semantic detector's emits an
end-of-turn probability from the transcript — both feed the identical harness and metrics.

## Detectors

Two detectors ship, both behind the same seam, both scored by the identical harness and
metrics so the comparison is apples-to-apples:

- **`silero-vad` (default).** Silence-based endpointing.
  [Silero VAD](https://github.com/snakers4/silero-vad) (MIT, ~2 MB ONNX) gives a speech
  probability per 20 ms frame; the policy endpoints after enough trailing silence. Runs on
  `onnxruntime` — **no torch** — so the default first-run stays light.
- **`semantic-turn` (optional).** Content-based endpointing. A fine-tune of
  Qwen2.5-0.5B-Instruct ([anyreach-ai/semantic-turn-taking](https://huggingface.co/anyreach-ai/semantic-turn-taking),
  Apache-2.0 weights) reads the growing **transcript** and predicts whether the turn is
  complete, so it can keep listening through a mid-utterance pause that pure silence would
  cut off. Install the extra and load both to compare live:

  ```bash
  pip install 'turntune[semantic]'      # adds transformers + torch; downloads a ~1 GB model
  turntune serve --detector silero-vad,semantic-turn
  ```

  The heavy deps load lazily, so installing the extra leaves the default Silero path
  unchanged and torch-free. (We use the openly-licensed anyreach model rather than LiveKit's
  turn-detector, whose model license restricts use to the LiveKit Agents framework.)

## The tuning knobs

These are the same knobs LiveKit's eot-bench exposes, so a policy setting maps between
the two (the eot-bench name is in parentheses). The arrows show what happens as you
**increase** each knob.

| Knob (eot-bench name) | What it does | Cutoffs | Latency |
|---|---|---|---|
| `speech_threshold` (`threshold`), 0.1–0.9 | VAD probability above which a frame counts as speech | ↑ (more audio read as silence → endpoints sooner) | ↓ |
| `min_silence_s` (`action_delay`), 0.1–1.5 — **primary sweep axis** | trailing silence required before declaring end-of-turn | ↓ (waits out mid-turn pauses) | ↑ |
| `timeout_s` (`timeout`), optional | force end-of-turn after this much silence — **now applied**; converts a would-be no-fire into a late fire | – | **bounds no-fire** (at some latency/cutoff cost) |

The table shows the Silero knobs; the **semantic** detector swaps `speech_threshold` for
`eot_threshold` (end-of-turn confidence read from the transcript) and shares `min_silence_s`
and `timeout_s`. Sliders are generated from whichever detector is selected.

## How the metrics are defined

- **Cutoff (false endpoint):** the detector declared end-of-turn during a *mid-turn
  pause*, before the user was actually done. The **cutoff rate** is the fraction of
  conversations where that happens.
- **Latency:** how long *after* the true end of turn the detector took to fire,
  measured on the audio timeline (conversational dead air) — not compute time. A model
  that waits 600 ms to be sure still shows 600 ms of latency.
- **No-fire (missed endpoint):** a turn where end-of-turn is *never* declared within the
  audio — the agent never takes the floor and the conversation stalls in silence. A no-fire is
  a failure, **not a smaller one than a cutoff**.
- **All three are reported together, and the headline bounds both cutoff and no-fire** —
  *best latency @ ≤X% cutoff **and ≤5% no-fire*** — so a policy can't lower its cutoff rate just
  by refusing to fire. The Pareto frontier is the non-dominated lower-left envelope under that
  bound.

Ground truth comes straight from eot-bench: the **final** silence in each clip is the
true end of turn; every **earlier** silence is a mid-turn hold. Full definitions in
[`docs/metrics.md`](./docs/metrics.md).

## Silence vs. semantics — an honest comparison

Both detectors over the same 40 English eot-bench clips, swept over the full three-knob policy
(confidence × silence × timeout), scored identically. Because a low cutoff can be **bought by
silence**, the headline bounds no-fire to ≤5% — the best latency reachable at each cutoff budget
*and* ≤5% no-fire:

| cutoff budget (with ≤5% no-fire) | `silero-vad` | `semantic-turn` |
|---|---|---|
| ≤ 5% cutoff | unreachable | unreachable |
| ≤ 10% cutoff | **1.06s** (10% cut, 5% no-fire) | **unreachable** |
| ≤ 20% cutoff | 0.62s (18% cut, 0% no-fire) | **0.46s** (20% cut, 3% no-fire) |
| ≤ 30% cutoff | 0.52s (30% cut, 0% no-fire) | **0.42s** (30% cut, 3% no-fire) |

**The honest read.** *Without* the no-fire bound the semantic model looks dramatically better — it
appears to reach 5% cutoff at 1.02s and 10% at 0.87s. But those points sit at **15% no-fire**: the
agent silently never answers on ~6 of 40 turns. Once no-fire is bounded to ≤5%, those wins
disappear — the semantic model **can't reach ≤10% cutoff at all**, where Silero can (1.06s). Its
**genuine, modest advantage is the moderate-cutoff / lower-latency regime**: ~0.46s vs 0.62s at
≤20% cutoff, ~0.42s vs 0.52s at ≤30% cutoff, both within the no-fire budget. Content-based
endpointing buys you **fewer cutoffs at low latency in the middle of the curve — not ultra-low
cutoffs for free**.

> No-fire and cutoff percentages are over **N = 40 clips and rounded half-up**, so a displayed
> **3% no-fire = 1 of 40 turns** (2.5%, rounded up). The in-app headline reflects the current
> knob settings (a 1-D slice along the chosen axis), so it can differ from this full-sweep
> frontier by a knob step.

## Adding your own detector

`turntune` has two pluggable seams. To add a detector, implement the `Detector` protocol
(`extract` + `decide`) and register it — the audio-only `silero-vad` and the transcript-based
`semantic-turn` are worked examples of each shape. See
[`examples/custom_detector.py`](./examples/custom_detector.py) and
[`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Adding your own scenarios

Implement a `ScenarioLoader` that yields `Scenario` objects with `hold`/`eot`
spans. The bundled fixtures loader is the worked example. (eot-bench is the v0
default; custom and synthetic scenario sets are a clean drop-in later.)

## Configuration

```
turntune serve    # default: launch the local web UI
turntune run      # headless: print the curve + operating points
turntune sweep    # headless: write the full sweep to JSON (--out sweep.json)
```

Common flags (any subcommand): `--detector silero-vad` (or a comma-separated list like
`silero-vad,semantic-turn` to load several and switch in the UI — the semantic one needs
`pip install 'turntune[semantic]'`), `--dataset eot-bench|fixtures`, `--language en`,
`--limit 100`, `--sweep-axis min_silence_s`. `serve` also takes `--port 8000`, `--open`
(open a browser), and `--realtime` (stream at mic pace — demo fidelity; off for sweeps).

The runtime cache (downloaded model, eot-bench subset, per-frame signals) lives in
`.turntune_cache/`. Wipe it with `rm -rf .turntune_cache`.

## Roadmap / out of scope for v0

v0 is **component-level** evaluation of a single detector. Explicitly **not** in v0:

- End-to-end testing of a live deployed agent (driving audio over WebRTC/telephony
  and timestamping its spoken response).
- Synthetic scenario generation (TTS + programmatic pause/disfluency insertion).
- Hosting, multi-tenancy, auth, billing.

## Data & licensing

This project's **code is Apache-2.0** (see [`LICENSE`](./LICENSE)). The eot-bench
scenarios are **downloaded at runtime from Hugging Face and are not vendored** here;
they belong to LiveKit and are licensed **CC-BY-4.0** — attribution required; see the
[dataset card](https://huggingface.co/datasets/livekit/eot-bench-data). Built on
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT), the
[semantic turn-taking model](https://huggingface.co/anyreach-ai/semantic-turn-taking)
(Apache-2.0; downloaded only when you install the `[semantic]` extra), and
[LiveKit eot-bench](https://github.com/livekit/eot-bench) (Apache-2.0).

## Contributing & troubleshooting

See [`CONTRIBUTING.md`](./CONTRIBUTING.md). Common first-run issues (onnxruntime
wheels, Hugging Face download/rate limits, port already in use) are documented
there. Tests run fully offline against bundled fixtures: `make test`.
