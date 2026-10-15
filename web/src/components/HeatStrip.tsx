import { heat, pct } from "@/lib/format";

/** A car's whole finishing-position distribution, P1 → last, as a strip of heat cells. */
export function HeatStrip({
  probs,
  actual,
  label,
}: {
  probs: number[];
  actual?: number | null;
  label: string;
}) {
  const max = Math.max(...probs, 1e-9);
  return (
    <div
      className="grid h-6 w-full gap-px overflow-hidden rounded-[5px]"
      style={{ gridTemplateColumns: `repeat(${probs.length}, minmax(0, 1fr))` }}
      role="img"
      aria-label={`${label}: most likely P${probs.indexOf(max) + 1} (${pct(max)})`}
    >
      {probs.map((p, i) => (
        <div
          key={i}
          title={`P${i + 1}: ${pct(p)}`}
          className="relative"
          style={{ background: heat(p / Math.max(max, 0.25)) }}
        >
          {actual === i + 1 && (
            <span className="absolute inset-0 rounded-[3px] outline-2 -outline-offset-2 outline-[var(--color-heat)]" />
          )}
        </div>
      ))}
    </div>
  );
}
