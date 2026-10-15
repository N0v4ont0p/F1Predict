"use client";

import { LayoutGroup, motion } from "motion/react";
import { useMemo, useState } from "react";
import { pct } from "@/lib/format";
import { teamColour } from "@/lib/teams";
import type { Checkpoint, EventMart } from "@/lib/types";
import { HeatStrip } from "./HeatStrip";
import { ProbBar } from "./ProbBar";

const LABEL: Record<Checkpoint, string> = {
  C1: "Thursday · before the weekend",
  C4: "Sunday · after qualifying",
};

/** The probability timing tower: one row per car, re-ordering live between checkpoints. */
export function Tower({ event }: { event: EventMart }) {
  const available = (["C1", "C4"] as const).filter((c) => event.checkpoints[c]);
  const [cp, setCp] = useState<Checkpoint>(available.at(-1) ?? "C1");
  const fc = event.checkpoints[cp];
  const actual = useMemo(() => {
    const m = new Map<string, number | null>();
    for (const r of event.result ?? []) m.set(r.driver_id, r.classified ? r.position : null);
    return m;
  }, [event.result]);
  if (!fc) return null;
  const n = fc.cars.length;

  return (
    <section className="glass overflow-hidden" aria-label="Forecast timing tower">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/5 px-5 py-4">
        <div>
          <p className="eyebrow">Probability tower</p>
          <p className="mt-1 text-sm text-[var(--color-carbon-300)]">{LABEL[cp]}</p>
        </div>
        <div className="flex rounded-full bg-[var(--color-carbon-850)] p-1" role="tablist">
          {available.map((c) => (
            <button
              key={c}
              type="button"
              role="tab"
              aria-selected={c === cp}
              onClick={() => setCp(c)}
              className={`relative rounded-full px-4 py-1.5 font-mono text-xs tracking-widest transition-colors ${
                c === cp
                  ? "text-[var(--color-carbon-950)]"
                  : "text-[var(--color-carbon-300)] hover:text-white"
              }`}
            >
              {c === cp && (
                <motion.span
                  layoutId="cp-pill"
                  className="absolute inset-0 rounded-full bg-[var(--color-signal)]"
                  transition={{ type: "spring", stiffness: 420, damping: 34 }}
                />
              )}
              <span className="relative">{c}</span>
            </button>
          ))}
        </div>
      </header>

      <div className="hidden grid-cols-[2.5rem_minmax(9rem,1.1fr)_minmax(12rem,2fr)_7.5rem_6rem_6rem_4.5rem_3.5rem] items-center gap-3 px-5 pt-3 pb-2 lg:grid">
        {[
          "#",
          "Driver",
          `Finishing position P1 → P${n}`,
          "Win",
          "Podium",
          "Points",
          "Finish",
          "Result",
        ].map((h) => (
          <span key={h} className="eyebrow !text-[0.6rem]">
            {h}
          </span>
        ))}
      </div>

      <LayoutGroup>
        <ol className="divide-y divide-white/[0.04]">
          {fc.cars.map((car, i) => {
            const colour = teamColour(car.lineage);
            const pos = actual.get(car.driver_id);
            return (
              <motion.li
                layout
                key={car.driver_id}
                transition={{ type: "spring", stiffness: 260, damping: 30, mass: 0.8 }}
                className="group grid grid-cols-[2.5rem_1fr_auto] items-center gap-x-3 gap-y-2 px-5 py-2.5 hover:bg-white/[0.025] lg:grid-cols-[2.5rem_minmax(9rem,1.1fr)_minmax(12rem,2fr)_7.5rem_6rem_6rem_4.5rem_3.5rem]"
              >
                <span className="tabular font-display text-lg font-bold text-[var(--color-carbon-300)]">
                  {i + 1}
                </span>
                <div className="flex min-w-0 items-center gap-3">
                  <span
                    className="h-8 w-1 shrink-0 rounded-full"
                    style={{ background: colour, boxShadow: `0 0 10px ${colour}` }}
                  />
                  <div className="min-w-0">
                    <p className="truncate font-display text-[0.95rem] font-semibold leading-tight">
                      <span className="mr-2 font-mono text-xs text-[var(--color-carbon-300)]">
                        {car.code}
                      </span>
                      {car.name}
                    </p>
                    <p className="truncate text-xs text-[var(--color-carbon-500)]">{car.team}</p>
                  </div>
                </div>
                <div className="col-span-3 lg:col-span-1">
                  <HeatStrip probs={car.position_probs} actual={pos} label={car.name} />
                </div>
                <div className="col-span-3 grid grid-cols-[1.6fr_1fr_1fr] items-center gap-3 lg:contents">
                  <div>
                    <span className="eyebrow !text-[0.55rem] lg:hidden">Win</span>
                    <ProbBar p={car.p_win} colour={colour} />
                  </div>
                  <div className="tabular font-mono text-sm">
                    <span className="eyebrow block !text-[0.55rem] lg:hidden">Podium</span>
                    {pct(car.p_podium)}
                  </div>
                  <div className="tabular font-mono text-sm">
                    <span className="eyebrow block !text-[0.55rem] lg:hidden">Points</span>
                    {pct(car.p_points)}
                  </div>
                </div>
                <span className="tabular hidden font-mono text-xs text-[var(--color-carbon-300)] lg:block">
                  {pct(car.p_finish, 0)}
                </span>
                <span className="hidden text-right lg:block">
                  {event.result ? (
                    pos ? (
                      <span className="tabular rounded-md bg-[color-mix(in_oklch,var(--color-heat)_18%,transparent)] px-2 py-0.5 font-mono text-xs font-bold text-[var(--color-heat)]">
                        P{pos}
                      </span>
                    ) : (
                      <span className="rounded-md bg-[color-mix(in_oklch,var(--color-flag)_20%,transparent)] px-2 py-0.5 font-mono text-xs text-[var(--color-flag)]">
                        DNF
                      </span>
                    )
                  ) : (
                    <span className="font-mono text-xs text-[var(--color-carbon-500)]">
                      {car.expected_position.toFixed(1)}
                    </span>
                  )}
                </span>
              </motion.li>
            );
          })}
        </ol>
      </LayoutGroup>
    </section>
  );
}
