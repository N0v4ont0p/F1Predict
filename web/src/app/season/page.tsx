import Link from "next/link";
import { getIndex } from "@/lib/data";
import { heat, pct, shortDate } from "@/lib/format";

export const metadata = { title: "Season" };

export default function Season() {
  const idx = getIndex();
  return (
    <div className="space-y-6">
      <header>
        <p className="eyebrow">{idx.season} season</p>
        <h1 className="font-display text-4xl font-bold">Every round, forecast and scored</h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--color-carbon-300)]">
          The favourite before the weekend (C1) and after qualifying (C4), the winner, and the
          probability the model had given them. Forecasts use only what was public at the time.
        </p>
      </header>
      <div className="glass overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-white/5 text-left">
              {[
                "Round",
                "Grand Prix",
                "Date",
                "Favourite C1",
                "Favourite C4",
                "Winner",
                "P(winner) C1",
                "P(winner) C4",
              ].map((h) => (
                <th key={h} className="eyebrow px-4 py-3 font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.04]">
            {idx.events.map((e) => (
              <tr key={e.event_id} className="hover:bg-white/[0.02]">
                <td className="tabular px-4 py-2.5 font-mono text-xs text-[var(--color-carbon-300)]">
                  R{e.round}
                </td>
                <td className="px-4 py-2.5">
                  {e.status === "upcoming" ? (
                    <span className="text-[var(--color-carbon-500)]">{e.name}</span>
                  ) : (
                    <Link
                      href={`/e/${e.event_id}/`}
                      className="font-display font-semibold hover:text-[var(--color-signal)]"
                    >
                      {e.name}
                    </Link>
                  )}
                  {e.sprint && (
                    <span className="ml-2 font-mono text-[0.6rem] text-[var(--color-signal)]">
                      SPRINT
                    </span>
                  )}
                  {e.wet && (
                    <span className="ml-2 text-[var(--color-rain)]" title="wet race">
                      ☂
                    </span>
                  )}
                </td>
                <td className="px-4 py-2.5 text-[var(--color-carbon-300)]">
                  {shortDate(e.race_date)}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs">
                  {e.favourite_C1 ? `${e.favourite_C1.code} ${pct(e.favourite_C1.p_win, 0)}` : "—"}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs">
                  {e.favourite_C4 ? `${e.favourite_C4.code} ${pct(e.favourite_C4.p_win, 0)}` : "—"}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-[var(--color-heat)]">
                  {e.winner?.code ?? (e.status === "next" ? "next" : "")}
                </td>
                {(["p_winner_C1", "p_winner_C4"] as const).map((k) => (
                  <td key={k} className="px-4 py-2.5">
                    {e[k] !== undefined ? (
                      <span
                        className="tabular inline-block min-w-14 rounded-md px-2 py-0.5 text-center font-mono text-xs text-black"
                        style={{ background: heat(e[k] ?? 0) }}
                      >
                        {pct(e[k] ?? 0)}
                      </span>
                    ) : (
                      "—"
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
