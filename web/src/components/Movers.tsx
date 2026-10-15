import { pct } from "@/lib/format";
import { teamColour } from "@/lib/teams";
import type { EventMart } from "@/lib/types";

/** What qualifying changed: the biggest win-probability movers from C1 to C4. */
export function Movers({ event }: { event: EventMart }) {
  const c1 = event.checkpoints.C1;
  const c4 = event.checkpoints.C4;
  if (!c1 || !c4) return null;
  const before = new Map(c1.cars.map((c) => [c.driver_id, c.p_win]));
  const moves = c4.cars
    .map((c) => ({ car: c, delta: c.p_win - (before.get(c.driver_id) ?? 0) }))
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 6);
  const max = Math.max(...moves.map((m) => Math.abs(m.delta)), 1e-6);
  return (
    <section className="glass p-5">
      <p className="eyebrow">What qualifying changed</p>
      <p className="mt-1 text-sm text-[var(--color-carbon-300)]">
        Win probability before the weekend (C1) → after qualifying (C4).
      </p>
      <ul className="mt-4 space-y-2.5">
        {moves.map(({ car, delta }) => {
          const up = delta >= 0;
          return (
            <li key={car.driver_id} className="grid grid-cols-[3rem_1fr_7rem] items-center gap-3">
              <span className="font-mono text-xs" style={{ color: teamColour(car.lineage) }}>
                {car.code}
              </span>
              <div className="relative h-2 rounded-full bg-[var(--color-carbon-800)]">
                <div
                  className="absolute inset-y-0 rounded-full"
                  style={{
                    left: up ? "50%" : `${50 - (Math.abs(delta) / max) * 50}%`,
                    width: `${(Math.abs(delta) / max) * 50}%`,
                    background: up ? "var(--color-green)" : "var(--color-flag)",
                  }}
                />
                <span className="absolute inset-y-[-3px] left-1/2 w-px bg-white/30" />
              </div>
              <span className="tabular text-right font-mono text-xs">
                {pct(before.get(car.driver_id) ?? 0)} → {pct(car.p_win)}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
