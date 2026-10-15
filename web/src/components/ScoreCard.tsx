import { pct } from "@/lib/format";
import type { EventMart } from "@/lib/types";

const NAMES: Record<string, string> = {
  world_v1: "f1predict",
  quali: "Qualifying order",
  grid: "Grid order",
  standings: "Championship order",
};

/** After the race: how much probability each model gave the winner, at each checkpoint. */
export function ScoreCard({ event }: { event: EventMart }) {
  const winner = event.result?.[0];
  if (!winner) return null;
  const rows = (["C1", "C4"] as const).flatMap((cp) =>
    Object.entries(event.scores[cp] ?? {}).map(([model, s]) => ({
      cp,
      model,
      p: s.winner_logloss === undefined ? undefined : Math.exp(-s.winner_logloss),
      pl: s.finisher_pl_loglik,
    })),
  );
  return (
    <section className="glass p-5">
      <p className="eyebrow">Post-race score card</p>
      <p className="mt-1 text-sm text-[var(--color-carbon-300)]">
        Probability given to the winner,{" "}
        <span className="text-[var(--color-heat)]">{winner.name}</span>, by each model.
      </p>
      <table className="mt-4 w-full text-sm">
        <thead>
          <tr className="text-left">
            <th className="eyebrow pb-2 font-normal">Checkpoint</th>
            <th className="eyebrow pb-2 font-normal">Model</th>
            <th className="eyebrow pb-2 text-right font-normal">P(winner)</th>
            <th className="eyebrow pb-2 text-right font-normal">Order log-lik / car</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {rows.map((r) => (
            <tr
              key={`${r.cp}-${r.model}`}
              className={r.model === "world_v1" ? "text-white" : "text-[var(--color-carbon-300)]"}
            >
              <td className="py-2 font-mono text-xs">{r.cp}</td>
              <td className="py-2">{NAMES[r.model] ?? r.model}</td>
              <td className="tabular py-2 text-right font-mono">
                {r.p === undefined ? "—" : pct(r.p)}
              </td>
              <td className="tabular py-2 text-right font-mono">
                {r.pl === undefined ? "—" : r.pl.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
