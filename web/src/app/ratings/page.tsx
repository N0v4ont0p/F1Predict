import { Sparkband } from "@/components/Charts";
import { getRatings } from "@/lib/data";
import { teamColour } from "@/lib/teams";

export const metadata = { title: "Ratings" };

export default function Ratings() {
  const r = getRatings();
  if (!r) return <p>No ratings published yet.</p>;
  const teamMu = r.teams.flatMap((t) => t.series.flatMap((s) => [s.mu - s.sd, s.mu + s.sd]));
  const [tmin, tmax] = [Math.min(...teamMu), Math.max(...teamMu)];
  const drvMu = r.drivers.flatMap((t) => t.series.flatMap((s) => [s.mu - s.sd, s.mu + s.sd]));
  const [dmin, dmax] = [Math.min(...drvMu), Math.max(...drvMu)];
  return (
    <div className="space-y-8">
      <header>
        <p className="eyebrow">{r.season} beliefs</p>
        <h1 className="font-display text-4xl font-bold">Car and driver ratings</h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--color-carbon-300)]">
          Every car&apos;s strength is a team-lineage performance plus a driver skill, each a
          Gaussian belief updated after every race (log-strength units; bands are ±1 sd). The fit
          says the car dominates: team beliefs move through the season, driver skill barely does.
        </p>
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        <section className="glass p-5">
          <h2 className="eyebrow mb-3">Teams</h2>
          <ol className="divide-y divide-white/5">
            {r.teams.map((t, i) => (
              <li key={t.lineage} className="flex items-center gap-3 py-2">
                <span className="tabular w-6 font-mono text-xs text-[var(--color-carbon-500)]">
                  {i + 1}
                </span>
                <span
                  className="h-6 w-1 rounded-full"
                  style={{ background: teamColour(t.lineage) }}
                />
                <span className="flex-1 font-display font-semibold">{t.team}</span>
                <Sparkband series={t.series} colour={teamColour(t.lineage)} min={tmin} max={tmax} />
                <span className="tabular w-24 text-right font-mono text-xs">
                  {t.mu.toFixed(2)}{" "}
                  <span className="text-[var(--color-carbon-500)]">±{t.sd.toFixed(2)}</span>
                </span>
              </li>
            ))}
          </ol>
        </section>
        <section className="glass p-5">
          <h2 className="eyebrow mb-3">Drivers</h2>
          <ol className="divide-y divide-white/5">
            {r.drivers.map((d, i) => (
              <li key={d.driver_id} className="flex items-center gap-3 py-2">
                <span className="tabular w-6 font-mono text-xs text-[var(--color-carbon-500)]">
                  {i + 1}
                </span>
                <span className="w-10 font-mono text-xs text-[var(--color-carbon-300)]">
                  {d.code}
                </span>
                <span className="flex-1 truncate font-display font-semibold">{d.name}</span>
                <Sparkband series={d.series} colour="var(--color-signal)" min={dmin} max={dmax} />
                <span className="tabular w-24 text-right font-mono text-xs">
                  {d.mu.toFixed(2)}{" "}
                  <span className="text-[var(--color-carbon-500)]">±{d.sd.toFixed(2)}</span>
                </span>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  );
}
