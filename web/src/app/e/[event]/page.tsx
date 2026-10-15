import type { Metadata } from "next";
import Link from "next/link";
import { EventHero } from "@/components/EventHero";
import { Movers } from "@/components/Movers";
import { ScoreCard } from "@/components/ScoreCard";
import { SeasonStrip } from "@/components/SeasonStrip";
import { Tower } from "@/components/Tower";
import { WhyPanel } from "@/components/WhyPanel";
import { getEvent, getIndex, hasTheatre } from "@/lib/data";

export function generateStaticParams() {
  return getIndex()
    .events.filter((e) => e.status !== "upcoming")
    .map((e) => ({ event: e.event_id }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ event: string }>;
}): Promise<Metadata> {
  const { event } = await params;
  const ev = getEvent(event);
  return { title: `${ev.name} ${ev.season} forecast` };
}

export default async function EventPage({ params }: { params: Promise<{ event: string }> }) {
  const { event } = await params;
  const ev = getEvent(event);
  const idx = getIndex();
  const shown = ev.checkpoints.C4 ?? ev.checkpoints.C1;
  return (
    <div className="space-y-8">
      <EventHero event={ev} />
      {hasTheatre(ev.event_id) && (
        <Link
          href={`/theatre/${ev.event_id}/`}
          className="glass flex items-center justify-between gap-4 px-5 py-4 transition hover:bg-white/[0.03]"
        >
          <span>
            <span className="eyebrow block">Race theatre</span>
            <span className="font-display text-lg font-semibold">
              Watch simulated races on the circuit in 3D
            </span>
          </span>
          <span className="font-mono text-sm text-[var(--color-signal)]">→</span>
        </Link>
      )}
      <Tower event={ev} />
      <div className="grid gap-6 lg:grid-cols-2">
        {shown && <WhyPanel forecast={shown} />}
        <Movers event={ev} />
        <ScoreCard event={ev} />
        <ScoreCard event={ev} session="quali" />
      </div>
      <SeasonStrip events={idx.events} active={ev.event_id} />
    </div>
  );
}
