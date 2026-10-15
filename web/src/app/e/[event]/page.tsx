import type { Metadata } from "next";
import { EventHero } from "@/components/EventHero";
import { Movers } from "@/components/Movers";
import { ScoreCard } from "@/components/ScoreCard";
import { SeasonStrip } from "@/components/SeasonStrip";
import { Tower } from "@/components/Tower";
import { WhyPanel } from "@/components/WhyPanel";
import { getEvent, getIndex } from "@/lib/data";

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
      <Tower event={ev} />
      <div className="grid gap-6 lg:grid-cols-2">
        {shown && <WhyPanel forecast={shown} />}
        <Movers event={ev} />
        <ScoreCard event={ev} />
      </div>
      <SeasonStrip events={idx.events} active={ev.event_id} />
    </div>
  );
}
