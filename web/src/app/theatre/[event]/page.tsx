import type { Metadata } from "next";
import Link from "next/link";
import { RaceTheatre } from "@/components/RaceTheatre";
import { getIndex, getTheatre, hasTheatre, readCircuit } from "@/lib/data";

export function generateStaticParams() {
  const ids = getIndex()
    .events.map((e) => e.event_id)
    .filter(hasTheatre);
  // a static export needs at least one page per dynamic route
  return ids.length ? ids.map((event) => ({ event })) : [{ event: "none" }];
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ event: string }>;
}): Promise<Metadata> {
  const { event } = await params;
  const t = getTheatre(event);
  return { title: t ? `${t.name} ${t.season} race theatre` : "Race theatre" };
}

export default async function Theatre({ params }: { params: Promise<{ event: string }> }) {
  const { event } = await params;
  const theatre = getTheatre(event);
  if (!theatre) {
    return (
      <div className="glass p-8 text-[var(--color-carbon-300)]">
        No simulated races published for this event — run{" "}
        <code className="font-mono">f1p model publish</code>.
      </div>
    );
  }
  const circuit = readCircuit(theatre.circuit_id);
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="eyebrow">
            Race theatre · R{theatre.round} ·{" "}
            {theatre.checkpoint === "C4"
              ? "from the real grid"
              : "grid sampled from the qualifying forecast"}
          </p>
          <h1 className="mt-2 font-display text-4xl font-bold leading-none tracking-tight">
            {theatre.name} {theatre.season}
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-[var(--color-carbon-300)]">
            {theatre.races.length} of the lap-by-lap simulator&apos;s races: pace from the lap
            model, tyre wear, pit stops, overtaking, safety cars and retirements. Each one is a
            plausible race, not a prediction of this one.
          </p>
        </div>
        <Link
          href={`/e/${theatre.event_id}/`}
          className="rounded-full border border-white/10 px-4 py-1.5 font-display text-xs font-semibold hover:bg-white/5"
        >
          ← Forecast
        </Link>
      </header>
      <RaceTheatre theatre={theatre} circuit={circuit} />
    </div>
  );
}
