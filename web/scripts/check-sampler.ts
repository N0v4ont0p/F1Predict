// Cross-language check: the browser sampler (src/lib/sampler.ts) must reproduce the Python
// forecasts in the marts within Monte Carlo error. Run with
//   node --experimental-strip-types scripts/check-sampler.ts [public/data]
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { sample } from "../src/lib/sampler.ts";

interface Car {
  driver_id: string;
  p_win: number;
  p_finish: number;
  position_probs: number[];
  log_strength?: number | null;
}
interface Fc {
  stage_scales?: number[] | null;
  cars: Car[];
}

const root = process.argv[2] ?? "public/data";
const index = JSON.parse(readFileSync(join(root, "index.json"), "utf8")) as {
  events: { event_id: string; status: string }[];
};
const ids = index.events.filter((e) => e.status !== "upcoming").map((e) => e.event_id);
let checked = 0;
const worst = { pos: 0, win: 0, fin: 0 };
for (const id of ids.slice(-4)) {
  const ev = JSON.parse(readFileSync(join(root, "events", `${id}.json`), "utf8")) as {
    checkpoints: Record<string, Fc>;
  };
  for (const [cp, fc] of Object.entries(ev.checkpoints)) {
    if (fc.cars.some((c) => c.log_strength == null)) continue;
    const out = sample({
      logStrength: fc.cars.map((c) => c.log_strength as number),
      pFinish: fc.cars.map((c) => c.p_finish),
      stageScales: fc.stage_scales ?? null,
      samples: 40_000,
      seed: 7,
    });
    fc.cars.forEach((c, i) => {
      const row = out.positionProbs[i] ?? [];
      c.position_probs.forEach((p, k) => {
        worst.pos = Math.max(worst.pos, Math.abs(p - (row[k] ?? 0)));
      });
      worst.win = Math.max(worst.win, Math.abs(c.p_win - (out.pWin[i] ?? 0)));
      worst.fin = Math.max(worst.fin, Math.abs(c.p_finish - (out.pFinish[i] ?? 0)));
    });
    checked++;
    console.log(`${id} ${cp}: ok so far (max |Δ| position ${worst.pos.toFixed(4)})`);
  }
}
const limits = { pos: 0.03, win: 0.012, fin: 0.012 };
console.log(
  `checked ${checked} forecasts: max |Δ| position ${worst.pos.toFixed(4)}, ` +
    `P(win) ${worst.win.toFixed(4)}, P(finish) ${worst.fin.toFixed(4)}`,
);
if (checked === 0) {
  console.error("no forecast with log-strengths in the marts — republish them");
  process.exit(1);
}
if (worst.pos > limits.pos || worst.win > limits.win || worst.fin > limits.fin) {
  console.error("the browser sampler disagrees with the Python forecasts");
  process.exit(1);
}
