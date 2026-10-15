import Link from "next/link";
import { heat, pct, shortDate } from "@/lib/format";
import type { CalendarRow } from "@/lib/types";

/** Every round of the season: how much probability the model gave the eventual winner. */
export function SeasonStrip({ events, active }: { events: CalendarRow[]; active?: string }) {
  return (
    <ol className="grid grid-cols-3 gap-2 sm:grid-cols-6 lg:grid-cols-8 xl:grid-cols-12">
      {events.map((e) => {
        const p = e.p_winner_C4 ?? e.p_winner_C1;
        const isActive = e.event_id === active;
        return (
          <li key={e.event_id}>
            <Link
              href={`/e/${e.event_id}/`}
              className={`group relative flex h-24 flex-col justify-between overflow-hidden rounded-xl border p-2 transition hover:-translate-y-0.5 ${
                isActive
                  ? "border-[var(--color-signal)] shadow-[0_0_24px_-6px_var(--color-signal)]"
                  : "border-white/5 hover:border-white/20"
              }`}
              style={{
                background:
                  p !== undefined
                    ? `linear-gradient(180deg, transparent 20%, color-mix(in oklch, ${heat(p)} 55%, transparent))`
                    : e.status === "next"
                      ? "linear-gradient(180deg, transparent, color-mix(in oklch, var(--color-signal) 22%, transparent))"
                      : "oklch(0.18 0.01 265 / 0.6)",
              }}
              title={
                e.winner
                  ? `${e.name}: won by ${e.winner.code}; model gave ${pct(p ?? 0)} (C4)`
                  : `${e.name} · ${e.status}`
              }
            >
              <span className="flex items-center justify-between font-mono text-[0.6rem] text-white/50">
                R{e.round}
                {e.sprint && <span className="text-[var(--color-signal)]">S</span>}
                {e.wet && <span className="text-[var(--color-rain)]">☂</span>}
              </span>
              <span className="truncate font-display text-[0.7rem] font-semibold leading-tight">
                {e.name}
              </span>
              <span className="font-mono text-[0.65rem]">
                {e.winner ? (
                  <>
                    <span className="text-[var(--color-heat)]">{e.winner.code}</span>
                    <span className="ml-1 text-white/60">{pct(p ?? 0, 0)}</span>
                  </>
                ) : (
                  <span
                    className={
                      e.status === "next"
                        ? "animate-pulse-soft text-[var(--color-signal)]"
                        : "text-white/40"
                    }
                  >
                    {e.status === "next" ? "NEXT" : shortDate(e.race_date)}
                  </span>
                )}
              </span>
            </Link>
          </li>
        );
      })}
    </ol>
  );
}
