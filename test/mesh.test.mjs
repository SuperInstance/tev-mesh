// tev-mesh pins.
import test from "node:test";
import assert from "node:assert/strict";
import { makeMesh, runDrift, eGate } from "../src/mesh.mjs";
import { systemOne, choiceScores } from "../src/client.mjs";
import { createHash } from "node:crypto";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// deterministic target stream: regime A for t<half, regime B after
function driftStream({ n = 300, tokenDim = 8, modelDim = 16, seed = 11 }) {
  let s = seed >>> 0;
  const rnd = () => ((s ^= s << 13, s ^= s >>> 17, s ^= s << 5, s >>> 0) / 2 ** 32 - 0.5);
  const dirA = Array.from({ length: modelDim }, rnd);
  const dirB = Array.from({ length: modelDim }, rnd);
  return Array.from({ length: n }, (_, t) => {
    const dir = t < n / 2 ? dirA : dirB;
    const x = Array.from({ length: tokenDim }, rnd);
    return { x, target: dir.map((d, j) => d * (x[0] + 1) * 0.5 + 0.05 * rnd()) };
  });
}

test("TM.1 in-flight SGD adapts across a regime shift; frozen does not", () => {
  const engine = makeMesh({ tokenDim: 8, modelDim: 16, lr: 0.2, seed: 7 });
  const stream = driftStream({});
  const { adaptive, frozen } = runDrift({ engine, stream });
  const win = 50;
  const tailAd = adaptive.slice(-win).reduce((a, v) => a + v, 0) / win;
  const tailFr = frozen.slice(-win).reduce((a, v) => a + v, 0) / win;
  assert.ok(tailAd < tailFr * 0.5, `adaptive tail ${tailAd.toFixed(4)} ≪ frozen tail ${tailFr.toFixed(4)}`);
});

test("TM.2 the gradient is fresh per step (ZC-2 discipline is structural)", () => {
  const e1 = makeMesh({ tokenDim: 4, modelDim: 6, lr: 0.1, seed: 3 });
  const x = [1, 0, 1, 0], t1 = [1, 0, 1, 0, 1, 0], t2 = [0, 1, 0, 1, 0, 1];
  const wBefore = JSON.stringify(e1.W);
  const r1 = e1.step(x, t1);
  assert.ok(r1.loss > 0);
  const wAfter1 = JSON.stringify(e1.W);
  assert.notEqual(wBefore, wAfter1, "step mutated weights");
  // repeated IDENTICAL step: loss decreases (no accumulation blowup);
  // with the ZC-2 bug, repeated steps would diverge or oscillate wildly
  let prev = r1.loss;
  for (let i = 0; i < 40; i++) {
    const r = e1.step(x, t1);
    assert.ok(r.loss <= prev + 1e-12, "monotone on a stationary target");
    prev = r.loss;
  }
  const r2 = e1.step(x, t2);
  assert.ok(Number.isFinite(r2.loss), "finite loss on new target");
});

test("TM.3 e-gate on preference scores: out-of-calibration flags high", () => {
  const calib = [0.5, 0.55, 0.6, 0.62, 0.65, 0.7, 0.72, 0.75, 0.8, 0.85];
  const res = eGate([0.3, 0.66, 0.97], calib);
  assert.ok(res[0].e > 1, "far below calibration mass → flagged");
  assert.ok(res[1].e < 1, "inside mass → not flagged");
  assert.ok(res[2].e > 1, "far above mass → flagged (novel-preference detector)");
  assert.ok(res.every((r) => r.e <= 100), "cap respected");
});

test("TM.4 playback client: honest canned responses, unknown requests refused", async () => {
  const model = "tev1:0.8b";
  const state = { canvas: "demo" };
  const questions = { gate: { type: "choice", instructions: "route?", criteria: { hold: null, recompile: null } } };
  const key = createHash("sha256").update(JSON.stringify({ model, state, questions })).digest("hex").slice(0, 16);
  const dir = mkdtempSync(join(tmpdir(), "tev-"));
  const fixturePath = join(dir, "fx.json");
  writeFileSync(fixturePath, JSON.stringify({
    [key]: { answers: { gate: { type: "choice", choice: "recompile", probabilities: { hold: 0.2, recompile: 0.8 }, confidence: 0.8 } } },
  }));
  const res = await systemOne({ model, state, questions, mode: "playback", fixturePath });
  assert.equal(res._playback, true);
  assert.equal(res.answers.gate.choice, "recompile");
  await assert.rejects(
    () => systemOne({ model, state: { canvas: "UNKNOWN" }, questions, mode: "playback", fixturePath }),
    /no canned response/,
  );
});

test("TM.5 choiceScores marks preferences uncalibrated (model-card discipline)", async () => {
  const { scores, calibrated } = choiceScores({ type: "choice", choice: "a", probabilities: { a: 0.7, b: 0.3 }, confidence: 0.65 });
  assert.equal(scores.a, 0.7);
  assert.equal(calibrated, false, "never raw-routed: calibrate first");
});
