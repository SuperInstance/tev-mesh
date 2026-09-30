# tev-mesh

TEV1 bidirectional feedback mesh — a decision model (Together AI's
`tev1:0.8b`, Jev-style `/v1/systemone` API) running at the edge of a glyph
canvas, with in-flight SGD over a thin projection layer: inference that
trains while it infers. JS reference implementation; the PyTorch harness in
the spec is mirrored honestly (no torch on this host).

## Why the e-gate is structural, not optional

Verified facts about tev1 (kimi_search, 2026-10-01): 0.8B params, 812MB,
Qwen3.5-0.8B SFT, `/v1/systemone` decision API (choice/noul/score, up to 64
questions/call), vendor-claimed ~50ms E2E. Independent eval (Bespoke,
13 datasets, 3880 decisions): **63.5% vs Jev's 76.0%** — Together's own 88%
is a dev-set number (UnverifiedClaim). The model card itself flags that
logprobs are **preferences, not calibrated confidence**, and prompt
injection / multilingual / calibration are untested. That is the third
independent convergence with the realm-ml JEV-gate flag (AUC 0.510) and
jev-fusion claim-4 (judge ≈ free heuristic, 56% off oracle). So every
decision output passes through `eGate` — a two-sided uniform-rank e-value
gate — before threshold routing.

## Layout (v0.1)

- `src/client.mjs` — `/v1/systemone` client, `live` + `playback` modes.
  Playback is keyed by request hash and **refuses unknown requests** — no
  silent canned answers. `choiceScores` marks preferences uncalibrated.
- `src/mesh.mjs` — `BidirectionalQuiltEngine` mirror: projection → MSE →
  SGD with **structural zero-grad discipline** (the gradient is computed
  fresh per step; the ZC-2 accumulation bug cannot exist here, pinned);
  `runDrift` quantifies adaptation vs a frozen snapshot across a regime
  shift; `eGate` two-sided e-values on preference scores.
- `fixtures/systemone.sample.json` — SYNTHETIC playback fixture (marked).
- `test/mesh.test.mjs` — 5 pins.

```
node --test test/mesh.test.mjs   # 5/5
```

## Honest marks

- All suite numbers MEASURED on this host (drift experiment is a synthetic
  Float64 regime shift, deterministic seed).
- No Vulkan semaphore / real VRAM on this host — the sync story is spec +
  pinned loop shape, not hardware-validated.
- tev1 capability numbers are CITED (see above); only the loop behavior is
  MEASURED.
