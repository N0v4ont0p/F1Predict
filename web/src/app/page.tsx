import Link from "next/link";
import { EventHero } from "@/components/EventHero";
import { ScoreCard } from "@/components/ScoreCard";
import { SeasonStrip } from "@/components/SeasonStrip";
import { Tower } from "@/components/Tower";
import { featuredEvent, getEvent, getIndex } from "@/lib/data";
import { pct } from "@/lib/format";

export default function Hub() {
  const idx = getIndex();
  const event = getEvent(featuredEvent(idx));
  const raced = idx.events.filter((e) => e.p_winner_C4 !== undefined);
  const meanP = raced.reduce((s, e) => s + (e.p_winner_C4 ?? 0), 0) / Math.max(raced.length, 1);
  const hits = raced.filter((e) => e.favourite_C4?.driver_id === e.winner?.driver_id).length;

  return (
    <div className="space-y-8">
      <EventHero event={event} />

      <section className="grid gap-4 sm:grid-cols-3">
        {[
          [
            "Races forecast this season",
            String(raced.length),
            "every one scored against the result",
          ],
          ["Average P(winner) at C4", pct(meanP), "probability given to the eventual winner"],
          ["Favourite won", `${hits}/${raced.length}`, "the C4 favourite took the flag"],
        ].map(([k, v, sub]) => (
          <div key={k} className="glass p-5">
            <p className="eyebrow">{k}</p>
            <p className="tabular mt-2 font-display text-4xl font-bold">{v}</p>
            <p className="mt-1 text-xs text-[var(--color-carbon-500)]">{sub}</p>
          </div>
        ))}
      </section>

      <Tower event={event} />
      <ScoreCard event={event} />

      <section className="space-y-3">
        <div className="flex items-end justify-between">
          <div>
            <p className="eyebrow">{idx.season} season</p>
            <h2 className="font-display text-2xl font-bold">How much did we give the winner?</h2>
          </div>
          <Link href="/season/" className="text-sm text-[var(--color-signal)] hover:underline">
            Full season →
          </Link>
        </div>
        <SeasonStrip events={idx.events} active={event.event_id} />
      </section>
    </div>
  );
}
