import { pct } from "@/lib/format";
import { teamColour } from "@/lib/teams";
import type { Standing } from "@/lib/types";

/** Title odds and the fan of final points (5–95% and 25–75% ranges, median tick). */
export function TitleFan({ rows, kind }: { rows: Standing[]; kind: "driver" | "team" }) {
  const top = rows.filter((r) => r.p_title > 0 || (r.quantiles[4] ?? 0) > 0).slice(0, 10);
  const max = Math.max(...top.map((r) => r.quantiles[4] ?? 0), 1);
  const x = (v: number) => `${(v / max) * 100}%`;
  return (
    <ol className="space-y-3">
      {top.map((r) => {
        const colour = teamColour(r.lineage ?? r.constructor_id ?? "");
        const [q05, q25, q50, q75, q95] = r.quantiles as [number, number, number, number, number];
        const label = kind === "driver" ? (r.name ?? r.driver_id) : r.team;
        return (
          <li
            key={r.driver_id ?? r.constructor_id}
            className="grid grid-cols-[minmax(8rem,12rem)_5rem_1fr] items-center gap-4"
          >
            <div className="min-w-0">
              <p className="truncate font-display font-semibold">
                {kind === "driver" && (
                  <span className="mr-2 font-mono text-xs text-[var(--color-carbon-300)]">
                    {r.code}
                  </span>
                )}
                {label}
              </p>
              <p className="tabular font-mono text-[0.7rem] text-[var(--color-carbon-500)]">
                {r.current.toFixed(0)} pts now · {r.expected.toFixed(0)} expected
              </p>
            </div>
            <p
              className="tabular text-right font-display text-xl font-bold"
              style={{ color: r.p_title > 0.005 ? colour : "var(--color-carbon-500)" }}
            >
              {pct(r.p_title)}
            </p>
            <div
              className="relative h-7"
              role="img"
              aria-label={`${label}: 90% range ${q05.toFixed(0)}–${q95.toFixed(0)} points`}
            >
              <div className="absolute inset-y-3 left-0 right-0 rounded-full bg-white/[0.03]" />
              <div
                className="absolute inset-y-2.5 rounded-full opacity-35"
                style={{ left: x(q05), width: `calc(${x(q95)} - ${x(q05)})`, background: colour }}
              />
              <div
                className="absolute inset-y-1.5 rounded-full opacity-80"
                style={{
                  left: x(q25),
                  width: `calc(${x(q75)} - ${x(q25)})`,
                  background: colour,
                  boxShadow: `0 0 14px ${colour}`,
                }}
              />
              <div
                className="absolute inset-y-0.5 w-0.5 rounded bg-white"
                style={{ left: x(q50) }}
              />
              <div
                className="absolute inset-y-1 w-px bg-[var(--color-heat)]"
                style={{ left: x(r.current) }}
                title={`${r.current} points today`}
              />
            </div>
          </li>
        );
      })}
    </ol>
  );
}
