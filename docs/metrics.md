# Metrics: how turntune scores a detector

This document defines, precisely, what turntune measures. The headline output is the
**latency-vs-cutoff Pareto curve** and the summary _"latency at ≤X% cutoff."_ The
definitions follow LiveKit's eot-bench so the numbers are directly comparable.

## The scenario model

Each eot-bench scenario is one **complete human user turn** plus a list of
**silence spans** (every pause ≥ 100 ms). LiveKit's ground-truth convention:

> The final silence span is the true end of the user's turn; every earlier silence
> span is a mid-turn hesitation.

So turntune labels the spans:

- the **last** silence span is the `eot` span — `true_eot_s` is **its start** (≈ the
  end of the last word);
- every **earlier** span is a `hold` — a mid-turn pause where the user has *not*
  finished. Firing here means talking over them.

(See `turntune/scenarios/eot_bench.py::spans_and_eot`.)

## Per-scenario classification

turntune streams the whole clip through the detector, which fires (at most) once at
time `eot_s`. Given a tolerance `tol_s` (default 0.1 s):

- **cutoff** — `eot_s < true_eot_s - tol_s`. The detector endpointed during a `hold`
  pause, before the turn was actually over. This is a false endpoint / talk-over.
- **correct** — fired at or after the true end (within tolerance).
- **missed** (no-fire) — never fired within the clip (dead air / no endpoint).

**Endpointing latency** (for `correct` decisions only) = `max(0, eot_s - true_eot_s)`.
This is measured on the **audio timeline** — conversational silence after the true end
of turn, *not* wall-clock compute time. As LiveKit puts it: an instant model that
waits 600 ms to be sure still shows 600 ms of latency.

## Aggregates and the curve

For a given policy setting (knob values):

- **cutoff rate** = fraction of scenarios classified `cutoff`.
- **latency** = p50 (plotted) / p90 (tooltip) over the `correct` scenarios.
- **no-fire rate** = fraction classified `missed`, tracked separately.

Sweeping a knob (default `min_silence_s`) produces one `(cutoff_rate, latency)` point
per setting. The **Pareto frontier** is the non-dominated lower-left envelope: settings
where you can't lower the cutoff rate without raising latency (or vice versa). The
headline **latency at ≤X% cutoff** is the lowest latency reachable at or below an X%
cutoff rate — e.g. _"best latency @ 10% cutoff."_

A setting with no `correct` scenarios has undefined latency and is excluded from the
curve.
