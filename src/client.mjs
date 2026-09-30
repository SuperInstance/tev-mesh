// TEV1 /v1/systemone client — live + playback modes.
// Verified facts (kimi_search 2026-10-01): tev1 is Together AI's Jev-style
// decision model (Qwen3.5-0.8B SFT, 812MB, ~50ms vendor-claimed, /v1/systemone
// endpoint, 1–64 questions/call, choice|noul|score types, 2–26 options).
// Independent eval: 63.5% (Bespoke 13-dataset, 3880 decisions; Jev 76.0%).
// Model card flags: logprobs are preferences, NOT calibrated confidence;
// prompt injection / multilingual / calibration untested.
// The mesh treats every answer as RANK evidence and runs the e-calibrated
// gate (mesh.mjs) before threshold routing — never raw probabilities.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

export async function systemOne({ baseUrl, apiKey, model = "tev1:0.8b", state, questions, mode = "playback", fixturePath }) {
  if (mode === "playback") {
    const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));
    // key the fixture by the request hash so playback is honest about which
    // canned answer serves which request; unknown requests are REFUSED
    const key = createHash("sha256").update(JSON.stringify({ model, state, questions })).digest("hex").slice(0, 16);
    const canned = fixture[key];
    if (!canned) throw new Error(`playback: no canned response for request ${key} — record one or use live mode`);
    return { ...canned, _playback: true, _requestKey: key };
  }
  if (mode !== "live") throw new Error(`unknown mode ${mode}`);
  const res = await fetch(`${baseUrl}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model, state, questions }),
  });
  if (!res.ok) throw new Error(`systemone ${res.status}: ${await res.text()}`);
  return res.json();
}

// Normalize a choice answer into a plain scores record + an honest mark on
// calibration provenance (feeds the e-gate in mesh.mjs).
export function choiceScores(answer) {
  if (answer?.type !== "choice" || !answer.probabilities)
    throw new Error("choiceScores: not a choice answer");
  return {
    scores: answer.probabilities, // PREFERENCES, not confidence (model card)
    confidenceField: answer.confidence, // concentration, NOT P(correct)
    calibrated: false,
  };
}
