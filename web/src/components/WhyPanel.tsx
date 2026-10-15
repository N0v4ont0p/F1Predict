import { teamColour } from "@/lib/teams";
import type { CheckpointForecast } from "@/lib/types";

const LABEL: Record<string, string> = {
  rating: "Car + driver rating",
  grid: "Grid slot",
  quali: "Qualifying order",
  race_pace: "Recent race pace",
  lap_pace: "Race pace (lap model)",
  pace: "Qualifying pace",
  team_affinity: "Team at this circuit",
  driver_affinity: "Driver at this circuit",
  rating_q: "Rating (with qualifying)",
};

const COLOURS = [
  "var(--color-signal)",
  "var(--color-heat)",
  "oklch(0.72 0.17 320)",
  "var(--color-green)",
];

/** Why the model rates each car where it does: its log-strength split by signal. */
export function WhyPanel({ forecast, top = 8 }: { forecast: CheckpointForecast; top?: number }) {
  const cars = forecast.cars.slice(0, top).filter((c) => c.why && Object.keys(c.why).length);
  if (!cars.length) return null;
  const keys = Object.keys(cars[0]?.why ?? {});
  const extent = Math.max(
    ...cars.flatMap((c) => {
      const v = Object.values(c.why ?? {});
      return [
        v.filter((x) => x > 0).reduce((a, b) => a + b, 0),
        -v.filter((x) => x < 0).reduce((a, b) => a + b, 0),
      ];
    }),
    1e-6,
  );
  return (
    <section className="glass p-5">
      <p className="eyebrow">Why? · what drives each car&apos;s strength</p>
      <div className="mt-3 flex flex-wrap gap-4 text-xs text-[var(--color-carbon-300)]">
        {keys.map((k, i) => (
          <span key={k} className="flex items-center gap-1.5">
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: COLOURS[i % COLOURS.length] }}
            />
            {LABEL[k] ?? k}
          </span>
        ))}
      </div>
      <ul className="mt-4 space-y-2">
        {cars.map((c) => {
          let pos = 0;
          let neg = 0;
          return (
            <li key={c.driver_id} className="grid grid-cols-[3rem_1fr] items-center gap-3">
              <span className="font-mono text-xs" style={{ color: teamColour(c.lineage) }}>
                {c.code}
              </span>
              <div
                className="relative h-4"
                role="img"
                aria-label={`${c.name}: ${keys
                  .map(
                    (k) =>
                      `${LABEL[k] ?? k} ${(c.why?.[k] ?? 0) >= 0 ? "+" : ""}${(c.why?.[k] ?? 0).toFixed(2)}`,
                  )
                  .join(", ")}`}
              >
                <span className="absolute inset-y-[-2px] left-1/2 w-px bg-white/25" />
                {keys.map((k, i) => {
                  const v = c.why?.[k] ?? 0;
                  const w = (Math.abs(v) / extent) * 50;
                  const left =
                    v >= 0 ? 50 + (pos / extent) * 50 : 50 - ((neg + Math.abs(v)) / extent) * 50;
                  if (v >= 0) pos += v;
                  else neg += Math.abs(v);
                  return (
                    <span
                      key={k}
                      className="absolute inset-y-0.5 rounded-sm"
                      style={{
                        left: `${left}%`,
                        width: `${w}%`,
                        background: COLOURS[i % COLOURS.length],
                        opacity: 0.85,
                      }}
                      title={`${LABEL[k] ?? k}: ${v >= 0 ? "+" : ""}${v.toFixed(2)}`}
                    />
                  );
                })}
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-xs text-[var(--color-carbon-500)]">
        Log-strength relative to the field average; right of centre helps, left hurts.
      </p>
    </section>
  );
}
