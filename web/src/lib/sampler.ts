// The staged Plackett-Luce sampler with retirements, ported from f1p.eval.forecast.sample_orders
// so the What-if Lab can re-run a forecast in the browser. Same model, same estimators:
//   - each car is classified with probability pFinish;
//   - place k (for the first stageScales.length places) goes to an exponential race among the
//     classified cars still unplaced, with rates exp(stageScales[k] * logStrength);
//   - the remaining classified cars follow by plain Plackett-Luce (Gumbel-max) and the
//     retirements fill the slots behind them;
//   - P(win) is Rao-Blackwellised: the exact PL win probability given each sampled finisher set.
// No imports: this module also runs under `node --experimental-strip-types` for the
// cross-language check (scripts/check-sampler.ts).

export interface SampleInput {
  logStrength: number[];
  pFinish: number[];
  stageScales: number[] | null;
  samples: number;
  seed: number;
}

export interface SampleOutput {
  /** positionProbs[i][k] = P(car i finishes in place k+1). */
  positionProbs: number[][];
  pWin: number[];
  pFinish: number[];
}

/** mulberry32: a small, fast, well-distributed 32-bit PRNG (seeded, reproducible). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296; // in (0, 1), never 0
  };
}

export function sample(input: SampleInput): SampleOutput {
  const { logStrength: ls, pFinish, samples } = input;
  const n = ls.length;
  const stages = input.stageScales ? Math.min(input.stageScales.length, n) : 0;
  const scales = input.stageScales ?? [];
  const u = rng(input.seed);
  const max = Math.max(...ls);
  const centred = ls.map((v) => v - max);
  // exp(-scale_k * centred_i) for every stage, precomputed
  const invW: Float64Array[] = [];
  for (let k = 0; k < stages; k++) {
    const s = scales[k] ?? 1;
    invW.push(Float64Array.from(centred, (c) => Math.exp(-s * c)));
  }
  const firstScale = stages > 0 ? (scales[0] ?? 1) : 1;
  const winW = Float64Array.from(centred, (c) => Math.exp(firstScale * c));

  const counts = Array.from({ length: n }, () => new Float64Array(n));
  const winAcc = new Float64Array(n);
  const finAcc = new Float64Array(n);
  const key = new Float64Array(n);
  const finished = new Uint8Array(n);
  const avail = new Uint8Array(n);
  const order = new Int32Array(n);

  for (let s = 0; s < samples; s++) {
    let tot = 0;
    for (let i = 0; i < n; i++) {
      const fin = u() < (pFinish[i] ?? 1) ? 1 : 0;
      finished[i] = fin;
      avail[i] = fin;
      key[i] = (ls[i] ?? 0) - Math.log(-Math.log(u())) - (fin ? 0 : 1e9);
      if (fin) {
        finAcc[i] = (finAcc[i] ?? 0) + 1;
        tot += winW[i] ?? 0;
      }
    }
    if (tot > 0) {
      for (let i = 0; i < n; i++)
        if (finished[i]) winAcc[i] = (winAcc[i] ?? 0) + (winW[i] ?? 0) / tot;
    }
    for (let k = 0; k < stages; k++) {
      let best = -1;
      let bestT = Number.POSITIVE_INFINITY;
      const w = invW[k] as Float64Array;
      for (let i = 0; i < n; i++) {
        if (!avail[i]) continue;
        const t = -Math.log(u()) * (w[i] ?? 1);
        if (t < bestT) {
          bestT = t;
          best = i;
        }
      }
      if (best < 0) break;
      avail[best] = 0;
      key[best] = 1e12 - k;
    }
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => (key[b] ?? 0) - (key[a] ?? 0));
    for (let p = 0; p < n; p++) {
      const row = counts[order[p] as number] as Float64Array;
      row[p] = (row[p] ?? 0) + 1;
    }
  }
  const winTotal = winAcc.reduce((a, b) => a + b, 0) || 1;
  return {
    positionProbs: counts.map((row) => Array.from(row, (c) => c / samples)),
    pWin: Array.from(winAcc, (w) => w / winTotal),
    pFinish: Array.from(finAcc, (f) => f / samples),
  };
}
