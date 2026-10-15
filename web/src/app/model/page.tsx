import { Reliability, StageCurve } from "@/components/Charts";
import { getModel } from "@/lib/data";

export const metadata = { title: "Model Lab" };

const NAMES: Record<string, string> = {
  world_v1: "f1predict world model v1",
  world_v1_plain: "— one temperature",
  world_v1_noret: "— no retirement model",
  world_v1_nolap: "— no lap pace",
  world_v1_q: "— with qualifying in the filter",
  world_v1_2r: "— two ratings",
  quali: "Qualifying order",
  grid: "Grid order",
  standings: "Championship order",
  last_race: "Last race order",
  uniform: "Uniform",
};

const METRICS = [
  ["winner_logloss", "Winner log-loss", "lower"],
  ["podium_brier", "Podium Brier", "lower"],
  ["points_brier", "Points Brier", "lower"],
  ["finisher_pl_loglik", "Order log-lik / car", "higher"],
  ["ece_top3", "Calibration error (top 3)", "lower"],
] as const;

export default function ModelLab() {
  const m = getModel();
  if (!m)
    return (
      <div className="glass p-8">
        <h1 className="font-display text-3xl font-bold">Model Lab</h1>
        <p className="mt-2 text-[var(--color-carbon-300)]">
          The leaderboard has not been published yet (run <code>f1p model publish</code>).
        </p>
      </div>
    );
  return (
    <div className="space-y-8">
      <header>
        <p className="eyebrow">
          Rolling-origin backtest · {m.seasons[0]}–{m.seasons[1]}
        </p>
        <h1 className="font-display text-4xl font-bold">Model Lab</h1>
        <p className="mt-2 max-w-3xl text-sm text-[var(--color-carbon-300)]">
          Every race is forecast with only the data public at the checkpoint, then scored. Nothing
          the model sees was unknowable at the time; hyperparameters were fitted on races before the
          evaluation window.
        </p>
      </header>

      {(["C1", "C4"] as const).map((cp) => (
        <section key={cp} className="glass overflow-x-auto">
          <div className="flex items-center justify-between px-5 pt-5">
            <h2 className="font-display text-xl font-bold">
              {cp === "C1" ? "Before the weekend (C1)" : "After qualifying (C4)"}
            </h2>
            {m.comparisons[cp]?.winner_logloss && (
              <p className="font-mono text-xs text-[var(--color-carbon-300)]">
                vs {m.comparisons[cp].winner_logloss.reference}: Δ log-loss{" "}
                <span className="text-[var(--color-green)]">
                  {m.comparisons[cp].winner_logloss.mean_diff.toFixed(3)}
                </span>{" "}
                [{m.comparisons[cp].winner_logloss.lo.toFixed(3)},{" "}
                {m.comparisons[cp].winner_logloss.hi.toFixed(3)}]
              </p>
            )}
          </div>
          <table className="mt-3 w-full min-w-[720px] text-sm">
            <thead>
              <tr className="border-b border-white/5 text-left">
                <th className="eyebrow px-5 py-2 font-normal">Model</th>
                {METRICS.map(([k, label, dir]) => (
                  <th key={k} className="eyebrow px-3 py-2 text-right font-normal">
                    {label}{" "}
                    <span className="text-[var(--color-carbon-500)]">
                      {dir === "lower" ? "↓" : "↑"}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {m.leaderboard
                .filter((r) => r.checkpoint === cp)
                .map((r) => (
                  <tr
                    key={r.model}
                    className={
                      r.model === m.model
                        ? "bg-[color-mix(in_oklch,var(--color-signal)_8%,transparent)]"
                        : ""
                    }
                  >
                    <td className="px-5 py-2.5 font-display font-semibold">
                      {NAMES[r.model] ?? r.model}
                    </td>
                    {METRICS.map(([k]) => (
                      <td key={k} className="tabular px-3 py-2.5 text-right font-mono text-xs">
                        {r[k] === undefined ? "—" : (r[k] as number).toFixed(3)}
                      </td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      ))}

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Object.entries(m.reliability).map(([k, bins]) => (
          <Reliability key={k} bins={bins} title={`${k.replace("_", " · ")} calibration`} />
        ))}
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        {(["C1", "C4"] as const).map((cp) => {
          const c = m.combiner[cp];
          if (!c) return null;
          return (
            <figure key={cp} className="glass p-5">
              <figcaption>
                <p className="eyebrow">Stage temperature · {cp}</p>
                <p className="mt-1 text-sm text-[var(--color-carbon-300)]">
                  How sharply each place is decided, relative to the win. The front of a Grand Prix
                  is far more predictable than the midfield.
                </p>
              </figcaption>
              <StageCurve temps={c.stage_temperature} label={`Stage temperature curve at ${cp}`} />
              <p className="mt-2 font-mono text-xs text-[var(--color-carbon-300)]">
                weights{" "}
                {Object.entries(c.weights)
                  .map(([n, v]) => `${n} ${v.toFixed(2)}`)
                  .join(" · ")}
                {c.rain_gamma !== 0 && ` · rain γ ${c.rain_gamma.toFixed(2)}`}
              </p>
            </figure>
          );
        })}
      </section>
    </div>
  );
}
