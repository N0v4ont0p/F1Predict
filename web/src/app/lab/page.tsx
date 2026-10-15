import { type LabEvent, WhatIfLab } from "@/components/WhatIfLab";
import { featuredEvent, getIndex } from "@/lib/data";

export const metadata = { title: "What-if Lab" };

export default function Lab() {
  const idx = getIndex();
  const events: LabEvent[] = idx.events
    .filter((e) => e.status !== "upcoming")
    .map((e) => ({ event_id: e.event_id, name: e.name, round: e.round, status: e.status }));
  return <WhatIfLab events={events} initial={featuredEvent(idx)} />;
}
