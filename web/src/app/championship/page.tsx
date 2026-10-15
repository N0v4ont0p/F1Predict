import { TitleFan } from "@/components/TitleFan";
import { getChampionship } from "@/lib/data";
import { pct } from "@/lib/format";

export const metadata = { title: "Championship" };

export default function Championship() {
  const c = getChampionship();
  if (!c)
    return (
      <div className="glass p-8">
        <h1 className="font-display text-3xl font-bold">Championship</h1>
        <p className="mt-2 text-[var(--color-carbon-300)]">
          The season is complete — no title left to simulate.
        </p>
      </div>
    );
  const leader = c.drivers[0];
  return (
    <div className="space-y-8">
      <header className="glass relative overflow-hidden p-8">
        <p className="eyebrow">
          {c.season} championship · {c.remaining.length} rounds to go
          {c.sprints_remaining > 0 && ` · ${c.sprints_remaining} sprints`}
        </p>
        <h1 className="mt-3 font-display text-5xl font-bold leading-none tracking-tight">
          {leader ? (
            <>
              {leader.name}{" "}
              <span className="text-[var(--color-signal)]">{pct(leader.p_title)}</span>
            </>
          ) : (
            "Title odds"
          )}
        </h1>
        <p className="mt-3 max-w-2xl text-sm text-[var(--color-carbon-300)]">
          {c.n_sims.toLocaleString("en-GB")} simulated finishes to the season. Every remaining race
          is forecast from what is public today; each simulated season draws its own form from the
          model&apos;s uncertainty, so streaks carry from race to race.
        </p>
      </header>

      <section className="glass p-6">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-2">
          <h2 className="font-display text-2xl font-bold">Drivers</h2>
          <p className="font-mono text-[0.7rem] text-[var(--color-carbon-500)]">
            bands: final points 5–95% · 25–75% · <span className="text-white">median</span> ·{" "}
            <span className="text-[var(--color-heat)]">today</span>
          </p>
        </div>
        <TitleFan rows={c.drivers} kind="driver" />
      </section>

      <section className="glass p-6">
        <h2 className="mb-5 font-display text-2xl font-bold">Constructors</h2>
        <TitleFan rows={c.constructors} kind="team" />
      </section>
    </div>
  );
}
