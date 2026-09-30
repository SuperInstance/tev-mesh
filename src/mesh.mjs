// BidirectionalQuiltEngine — JS reference of the captain's TEV1 mesh spec.
// Pure math (Float64), no torch on this host (honest mark); the loop shape
// is what matters: read substrate → project → loss → in-flight SGD step,
// with the ZC-2 discipline BAKED IN (gradient is recomputed from zero each
// step — no accumulation bug can exist here; pinned by test).

// makeMesh({tokenDim, modelDim, lr, seed}) → engine with a deterministic
// small projection matrix.
export function makeMesh({ tokenDim = 8, modelDim = 16, lr = 0.05, seed = 7 } = {}) {
  // deterministic small init (xorshift-ish, not crypto)
  let s = seed >>> 0 || 1;
  const rnd = () => ((s ^= s << 13, s ^= s >>> 17, s ^= s << 5, s >>> 0) / 2 ** 32 - 0.5);
  const W = Array.from({ length: modelDim }, () => Array.from({ length: tokenDim }, rnd));
  const b = Array.from({ length: modelDim }, rnd);

  const project = (x) => W.map((row, j) => row.reduce((acc, w, i) => acc + w * x[i], b[j]));

  // one bidirectional step: forward + immediate SGD update (zero-grad by
  // construction — grad computed fresh from this step's forward pass only)
  function step(x, target) {
    const y = project(x);
    const N = modelDim;
    let loss = 0;
    const err = new Array(N);
    for (let j = 0; j < N; j++) { err[j] = y[j] - target[j]; loss += err[j] * err[j]; }
    loss /= N;
    // dL/dW_ji = (2/N) err_j x_i ; dL/db_j = (2/N) err_j
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < tokenDim; i++) W[j][i] -= lr * (2 / N) * err[j] * x[i];
      b[j] -= lr * (2 / N) * err[j];
    }
    return { y, loss, gradNorm: Math.sqrt(err.reduce((a, e) => a + e * e, 0)) };
  }

  const frozen = (x) => project(x);
  return { step, frozen, W, b, dims: { tokenDim, modelDim, lr } };
}

// Drift loop: stream of (x, target) pairs where the target REGIME shifts at
// `shiftAt`. Returns loss curves for the adaptive engine and a frozen copy
// of its initial weights — the quantified "trains while it infers" claim.
export function runDrift({ engine, stream, shiftAt = Math.floor(stream.length / 2) } = {}) {
  const frozenW = engine.W.map((r) => [...r]);
  const frozenB = [...engine.b];
  const frozenEngine = { frozen: (x) => frozenW.map((row, j) => row.reduce((a, w, i) => a + w * x[i], frozenB[j])) };
  const adaptive = [], frozen = [];
  for (let t = 0; t < stream.length; t++) {
    const { x, target } = stream[t];
    const r = engine.step(x, target);
    const yf = frozenEngine.frozen(x);
    let lf = 0;
    for (let j = 0; j < yf.length; j++) lf += (yf[j] - target[j]) ** 2;
    adaptive.push(r.loss);
    frozen.push(lf / yf.length);
  }
  return { adaptive, frozen, shiftAt };
}

// E-calibrated gate on tev1 preference scores (ZC-3 discipline, standalone
// copy — cite labs/egate exp1 for the rank construction). Scores are
// SIMILARITIES (higher = more preferred). sides:"low" flags scores far BELOW
// calibration mass; sides:"both" (default) flags either tail via the exact
// two-sided uniform-rank p-value: under exchangeability r = #{calib ≤ v} is
// uniform on 0..NC, so p_two = 2(m+1)/(NC+1) with m = min(r, NC−r) is a valid
// p, and e = (1/p − 1)/ENORM inherits validity by Markov + truncation.
export function eGate(scores, calib, { cap = 100, sides = "both" } = {}) {
  const sorted = [...calib].sort((a, c) => a - c);
  const NC = sorted.length;
  const enorm = NC > 1 ? (sorted.reduce((a, _, i) => a + 1 / (i + 1), 0) - 1) / (NC + 1) * NC : 1;
  return scores.map((v) => {
    let lo = 0, hi = NC;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] <= v) lo = mid + 1; else hi = mid; }
    const r = lo; // rank of v among calibration (count at-or-below)
    let raw;
    if (sides === "low") {
      const kp = Math.max(r, 1); // avoid 0; below-mass ⇒ small ⇒ large e
      raw = (NC - kp) / kp / enorm;
    } else {
      const m = Math.min(r, NC - r);
      const pTwo = (2 * (m + 1)) / (NC + 1);
      raw = (1 / pTwo - 1) / enorm;
    }
    if (!Number.isFinite(raw) || raw <= 0) return { e: 0, sat: false };
    return { e: Math.min(raw, cap), sat: raw > cap };
  });
}
