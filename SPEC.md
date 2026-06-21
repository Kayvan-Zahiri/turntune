# Turn-Taking Tuning Tool for Voice Agents
*(working name — rename as you like)*

## What this is
A developer tool for tuning turn-taking in voice agents. You point it at a turn-detection setup, it shows you exactly where the agent cuts users off versus where it leaves dead air, and it lets you tune the endpointing policy to the right operating point — instead of tuning by ear, which is how every voice team does it today.

One-liner: *See your voice agent's latency-vs-cutoff tradeoff, find the conversations where it talks over people, and dial in the endpointing policy.*

## This is built in public
This is an open-source project, developed in public from day one. Practical implications for how to build it:
- Public GitHub repo from the first commit. Permissive license (MIT or Apache-2.0).
- The README is a primary public surface, not an afterthought. Write it for a voice-AI developer discovering the project cold: what it does, why turn-taking matters, a 30-second quickstart, and a screenshot/gif of the tradeoff curve and a caught cutoff.
- Clean, documented, easy to clone and run locally. A good first-run experience matters more than feature breadth.
- Optimize for "a stranger can try it in five minutes," not for lock-in.

## The problem it solves
Turn-taking — deciding when the user is done speaking — is one of the highest-leverage quality problems in voice agents. Endpoint too early and the agent talks over people; too late and the conversation fills with dead air. Today, teams tune this by ear and bug reports. There's no easy way to see, for a given agent, how often it cuts users off, how much latency it adds, and how that tradeoff shifts when you change the policy. This tool makes that measurable and tunable.

## v0 scope (build this first)
Component-level evaluation of a single turn-detector, run against open scenario data, with a tunable tradeoff curve and failure playback.

- **Scenario source:** use LiveKit's `eot-bench` open datasets (on Hugging Face, Apache-2.0) as the scenario set for v0. Each item has audio plus a ground-truth end-of-turn. Structure the loader so custom/synthetic scenarios can be added later.
- **Detector under test:** support at least one open turn-detector out of the box (e.g. Silero VAD and/or an open semantic / end-of-turn model). Make the detector a pluggable adapter so others can be added.
- **Harness / runner:** stream each scenario's audio into the detector at realtime pace (e.g. 20 ms PCM frames at 16 kHz, the way a live mic delivers it), and record the exact timestamp the detector declares end-of-turn.
- **Metrics:** per scenario, compare the detector's decision against ground truth → compute (a) cutoff / false-endpoint rate (fired during a mid-turn pause) and (b) endpointing latency (delay after the true end). Sweep the detector's threshold/policy params and, for each setting, record the (cutoff rate, latency) pair → the latency-vs-cutoff Pareto curve. Report a clear summary like "latency at ≤X% cutoff."
- **Failure surfacing:** list the scenarios where the detector cut the user off, with audio playback, so the user can hear the failure.
- **Tuning interface:** expose the endpointing knobs (silence threshold, detector confidence threshold, minimum end-of-turn delay, etc.); let the user change them, re-run, watch the curve move, and pick an operating point.
- **UI:** a simple local web UI — the tradeoff curve chart, the list of caught cutoffs with playback, and the tuning controls. Local-first; no auth, no hosting.

## Suggested stack
- Python for the harness, detector adapters, and metrics (matches the audio/ML ecosystem and `eot-bench`).
- A lightweight local web UI for the curve, clip playback, and tuning controls — a small FastAPI/Flask backend serving a minimal React or plain-HTML frontend is fine. Keep v0 simple.
- Stack is flexible; prioritize a clean, runnable v0 over stack choices.

## Defer — do NOT build in v0
- End-to-end testing of a live deployed agent (driving audio into a real agent over WebRTC/telephony and timestamping its spoken response). That's a v2 and much harder. v0 is component-level only.
- Synthetic scenario generation (TTS + programmatic pause/disfluency insertion). Later — v0 uses eot-bench data.
- Hosting, multi-tenancy, auth, billing, any paid tier.

## v0 is done when
A voice developer can clone the repo, run it against a supported open detector and the eot-bench scenarios with one command, and get: the latency-vs-cutoff curve for that detector, a playable list of the conversations it cut off, and the ability to change the endpointing threshold and see the curve update — plus a README good enough that someone who finds the repo understands it and tries it in a few minutes.
