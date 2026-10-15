import { dateLabel, pct } from "@/lib/format";
import { teamColour } from "@/lib/teams";
import type { EventMart } from "@/lib/types";
import { Countdown } from "./Countdown";
import { ProbabilityFlow } from "./ProbabilityFlow";

export function EventHero({ event }: { event: EventMart }) {
  const fc = event.checkpoints.C4 ?? event.checkpoints.C1;
  const cars = fc?.cars ?? [];
  const lanes = cars
    .slice(0, 10)
    .map((c) => ({ code: c.code, p: c.p_win, colour: teamColour(c.lineage) }));
  const rest = cars.slice(10).reduce((s, c) => s + c.p_win, 0);
  if (rest > 0) lanes.push({ code: "FIELD", p: rest, colour: "oklch(0.55 0.02 265)" });
  const rain = event.weather.forecast?.wet_prob ?? event.weather.climatology?.wet_prob;
  const winner = event.result?.[0];

  return (
    <section className="glass relative overflow-hidden">
      <div className="grid gap-6 p-6 sm:p-8 lg:grid-cols-[1fr_1.15fr]">
        <div className="flex flex-col justify-between gap-8">
          <div>
            <p className="eyebrow">
              Round {event.round} · {event.season}
              {event.sprint && (
                <span className="ml-2 text-[var(--color-signal)]">Sprint weekend</span>
              )}
            </p>
            <h1 className="mt-3 font-display text-5xl font-bold leading-[0.95] tracking-tight sm:text-6xl">
              {event.name}
              <span className="block text-2xl font-semibold text-[var(--color-carbon-300)] sm:text-3xl">
                Grand Prix
              </span>
            </h1>
            <p className="mt-4 text-sm text-[var(--color-carbon-300)]">
              {dateLabel(event.race_date)}
            </p>
          </div>

          <dl className="grid grid-cols-3 gap-4">
            <div>
              <dt className="eyebrow">{winner ? "Winner" : "Favourite"}</dt>
              <dd className="mt-1 font-display text-xl font-bold sm:text-2xl">
                {winner ? winner.code : (cars[0]?.code ?? "—")}
              </dd>
              {!winner && cars[0] && (
                <dd className="tabular font-mono text-sm text-[var(--color-signal)]">
                  {pct(cars[0].p_win)}
                </dd>
              )}
              {winner && <dd className="text-sm text-[var(--color-carbon-300)]">{winner.team}</dd>}
            </div>
            <div>
              <dt className="eyebrow">Rain risk</dt>
              <dd className="mt-1 font-display text-xl font-bold text-[var(--color-rain)] sm:text-2xl">
                {rain === undefined ? "—" : pct(rain, 0)}
              </dd>
              <dd className="text-xs text-[var(--color-carbon-500)]">
                {event.weather.forecast ? "race-window forecast" : "climatology"}
              </dd>
            </div>
            <div>
              <dt className="eyebrow">
                {event.status === "raced" ? "Checkpoint" : "Lights out in"}
              </dt>
              <dd className="mt-1 text-xs sm:text-sm">
                {event.status === "raced" ? (
                  <span className="font-mono">{event.checkpoints.C4 ? "C4 · pre-race" : "C1"}</span>
                ) : (
                  <Countdown to={event.race_start_utc ?? `${event.race_date}T13:00:00Z`} />
                )}
              </dd>
            </div>
          </dl>
        </div>
        <div className="relative min-h-[18rem]">
          <p className="eyebrow absolute left-0 top-0 z-10">Win probability, as a river</p>
          <div className="pt-6">
            <ProbabilityFlow lanes={lanes} />
          </div>
        </div>
      </div>
    </section>
  );
}
