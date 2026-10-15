import { pct } from "@/lib/format";

export function ProbBar({
  p,
  colour,
  wide = false,
}: {
  p: number;
  colour: string;
  wide?: boolean;
}) {
  return (
    <div className={`flex items-center gap-2 ${wide ? "w-full" : "w-full lg:w-28"}`}>
      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-carbon-800)]">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700"
          style={{
            width: `${Math.max(p * 100, 0.8)}%`,
            background: `linear-gradient(90deg, color-mix(in oklch, ${colour} 55%, transparent), ${colour})`,
            boxShadow: `0 0 12px ${colour}`,
          }}
        />
      </div>
      <span className="tabular w-12 text-right font-mono text-xs text-[var(--color-carbon-100)]">
        {pct(p)}
      </span>
    </div>
  );
}
