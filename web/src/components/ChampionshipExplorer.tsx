"use client";

import { useEffect, useMemo, useState } from "react";
import { HeatStrip } from "@/components/HeatStrip";
import { pct } from "@/lib/format";
import { teamColour } from "@/lib/teams";
import type {
  ChampionshipHistoryMart,
  ChampionshipMart,
  ChampionshipSnapshot,
  HistoryStanding,
} from "@/lib/types";

type Kind = "driver" | "constructor";

interface Row extends HistoryStanding {
  id: string;
  label: string;
  code?: string;
  team?: string;
  colour: string;
}

function idOf(s: { driver_id?: string; constructor_id?: string }, kind: Kind): string {
  return (kind === "driver" ? s.driver_id : s.constructor_id) ?? "";
}

/** The current mart as a snapshot, so the page renders before (or without) the history. */
function fromCurrent(c: ChampionshipMart): ChampionshipSnapshot {
  const conv = (rows: ChampionshipMart["drivers"]) =>
    rows.map((r, i) => ({
      ...r,
      p_top3: r.p_top3 ?? r.p_position.slice(0, 3).reduce((a, b) => a + b, 0),
      most_likely: r.most_likely ?? i + 1,
    }));
  return {
    after_round: c.after_round ?? 0,
    as_of: c.as_of ?? c.generated_at,
    remaining: c.remaining.length,
    drivers: conv(c.drivers),
    constructors: conv(c.constructors),
  };
}

function rowsFor(
  snap: ChampionshipSnapshot,
  kind: Kind,
  hist: ChampionshipHistoryMart | null,
  current: ChampionshipMart | null,
): Row[] {
  const src = kind === "driver" ? snap.drivers : snap.constructors;
  const fallback = new Map(
    (kind === "driver" ? (current?.drivers ?? []) : (current?.constructors ?? [])).map((r) => [
      idOf(r, kind),
      r,
    ]),
  );
  return src
    .map((s) => {
      const id = idOf(s, kind);
      const known = fallback.get(id);
      if (kind === "driver") {
        const who = hist?.drivers[id];
        const lineage = who?.lineage ?? known?.lineage ?? "";
        return {
          ...s,
          id,
          label: who?.name ?? known?.name ?? id,
          code: who?.code ?? known?.code,
          team: who?.team ?? known?.team,
          colour: teamColour(lineage),
        };
      }
      const team = hist?.constructors[id];
      const lineage = team?.lineage ?? known?.lineage ?? id;
      return {
        ...s,
        id,
        label: team?.team ?? known?.team ?? id,
        colour: teamColour(lineage),
      };
    })
    .sort((a, b) => a.most_likely - b.most_likely);
}

function laneMeta(hist: ChampionshipHistoryMart, kind: Kind, id: string) {
  if (id === "__field") return { label: "Field", colour: "oklch(0.3 0.012 265)" };
  if (kind === "driver") {
    const who = hist.drivers[id];
    return { label: who?.code ?? id, colour: teamColour(who?.lineage ?? "") };
  }
  const team = hist.constructors[id];
  return { label: team?.team ?? id, colour: teamColour(team?.lineage ?? id) };
}

const W = 1000;
const H = 240;

/** P(title) through the season as a river: lane width = title probability after each round. */
function TitleRiver({
  hist,
  kind,
  selected,
  onSelect,
}: {
  hist: ChampionshipHistoryMart;
  kind: Kind;
  selected: number;
  onSelect: (i: number) => void;
}) {
  const snaps = hist.snapshots;
  const n = snaps.length;
  const { lanes, labels } = useMemo(() => {
    const series = new Map<string, number[]>();
    snaps.forEach((s, i) => {
      for (const r of kind === "driver" ? s.drivers : s.constructors) {
        const id = idOf(r, kind);
        const arr = series.get(id) ?? new Array<number>(n).fill(0);
        arr[i] = r.p_title;
        series.set(id, arr);
      }
    });
    const final = snaps[n - 1];
    const finalRows = final ? (kind === "driver" ? final.drivers : final.constructors) : [];
    const finalOrder = new Map(finalRows.map((r) => [idOf(r, kind), r.most_likely]));
    const top = [...series.entries()]
      .filter(([, v]) => Math.max(...v) >= 0.02)
      .sort((a, b) => (finalOrder.get(a[0]) ?? 99) - (finalOrder.get(b[0]) ?? 99))
      .slice(0, 8);
    const rest = new Array<number>(n).fill(0).map((_, i) => {
      const used = top.reduce((s, [, v]) => s + (v[i] ?? 0), 0);
      return Math.max(0, 1 - used);
    });
    const x = (i: number) => (n === 1 ? W / 2 : (i / (n - 1)) * W);
    const base = new Array<number>(n).fill(0);
    const out: { id: string; d: string; colour: string; label: string; opacity: number }[] = [];
    const seen = new Set<string>(); // team-mates share a colour: the second lane is fainter
    const labs: { id: string; label: string; y: number; colour: string }[] = [];
    const all: [string, number[]][] = [...top, ["__field", rest]];
    for (const [id, v] of all) {
      const upper = base.map((b, i) => b + (v[i] ?? 0));
      const top_ = upper.map((u, i) => `${x(i).toFixed(1)},${(u * H).toFixed(1)}`);
      const bottom = base.map((b, i) => `${x(i).toFixed(1)},${(b * H).toFixed(1)}`).reverse();
      const { label, colour } = laneMeta(hist, kind, id);
      const opacity = id === "__field" ? 0.5 : seen.has(colour) ? 0.42 : 0.82;
      seen.add(colour);
      out.push({ id, d: `M${top_.join("L")}L${bottom.join("L")}Z`, colour, label, opacity });
      const end = v[n - 1] ?? 0;
      if (end >= 0.06) labs.push({ id, label, y: ((base[n - 1] ?? 0) + end / 2) * H, colour });
      for (let i = 0; i < n; i++) base[i] = upper[i] ?? 0;
    }
    return { lanes: out, labels: labs };
  }, [snaps, n, kind, hist]);

  const pick = (e: React.PointerEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const t = (e.clientX - box.left) / box.width;
    onSelect(Math.max(0, Math.min(n - 1, Math.round(t * (n - 1)))));
  };
  const cursor = n === 1 ? 50 : (selected / (n - 1)) * 100;

  return (
    <div className="relative pr-14">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="h-60 w-full cursor-crosshair touch-none overflow-visible rounded-lg"
        onPointerDown={pick}
        onPointerMove={(e) => e.buttons === 1 && pick(e)}
        role="img"
        aria-label="Title probability after each round"
      >
        {lanes.map((l) => (
          <path
            key={l.id}
            d={l.d}
            fill={l.colour}
            fillOpacity={l.opacity}
            stroke="var(--color-carbon-950)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          >
            <title>{l.label}</title>
          </path>
        ))}
      </svg>
      <div className="relative mt-1 h-4 font-mono text-[0.6rem] text-[var(--color-carbon-500)]">
        {snaps.map((s, i) =>
          i === 0 || i === n - 1 || s.after_round % 4 === 0 ? (
            <span
              key={s.after_round}
              className="absolute -translate-x-1/2"
              style={{ left: `${n === 1 ? 50 : (i / (n - 1)) * 100}%` }}
            >
              {s.after_round === 0 ? "Pre" : `R${s.after_round}`}
            </span>
          ) : null,
        )}
      </div>
      <div
        className="pointer-events-none absolute top-0 bottom-5 w-px bg-white/80 shadow-[0_0_10px_white]"
        style={{ left: `calc((100% - 3.5rem) * ${cursor / 100})` }}
      />
      {labels.map((l) => (
        <span
          key={l.id}
          className="absolute right-0 w-12 -translate-y-1/2 truncate font-mono text-[0.65rem] font-semibold"
          style={{ top: `calc((100% - 1.25rem) * ${l.y / H})`, color: l.colour }}
        >
          {l.label}
        </span>
      ))}
    </div>
  );
}

function Delta({ now, before }: { now: number; before: number | undefined }) {
  if (before === undefined) return null;
  const d = (now - before) * 100;
  if (Math.abs(d) < 0.5) return null;
  return (
    <span
      className={`ml-1 font-mono text-[0.6rem] ${d > 0 ? "text-[var(--color-green)]" : "text-[var(--color-flag)]"}`}
    >
      {d > 0 ? "▲" : "▼"}
      {Math.abs(d).toFixed(0)}
    </span>
  );
}

function ProjectedTable({
  rows,
  previous,
  kind,
}: {
  rows: Row[];
  previous: Map<string, number>;
  kind: Kind;
}) {
  return (
    <div className="-mx-6 overflow-x-auto">
      <table className="w-full min-w-[760px] text-sm">
        <thead>
          <tr className="border-b border-white/5 text-left">
            <th className="eyebrow py-2 pl-6 pr-2 text-right font-normal">#</th>
            <th className="eyebrow px-3 py-2 font-normal">
              {kind === "driver" ? "Driver" : "Team"}
            </th>
            <th className="eyebrow px-2 py-2 text-right font-normal">Now</th>
            <th className="eyebrow px-2 py-2 text-right font-normal">Expected</th>
            <th className="eyebrow px-2 py-2 text-right font-normal">Median</th>
            <th className="eyebrow px-2 py-2 text-right font-normal">50%</th>
            <th className="eyebrow px-2 py-2 text-right font-normal">90%</th>
            <th className="eyebrow px-2 py-2 text-right font-normal">Title</th>
            <th className="eyebrow px-2 py-2 text-right font-normal">Top 3</th>
            <th className="eyebrow py-2 pl-3 pr-6 font-normal">
              Final position P1 → P{rows.length}
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.04]">
          {rows.map((r) => {
            const [q05, q25, q50, q75, q95] = r.quantiles;
            return (
              <tr key={r.id} className="transition-colors hover:bg-white/[0.02]">
                <td className="tabular py-2 pl-6 pr-2 text-right font-display font-bold text-[var(--color-carbon-300)]">
                  {r.most_likely}
                </td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="h-5 w-1 shrink-0 rounded-full"
                      style={{ background: r.colour }}
                    />
                    <div className="min-w-0">
                      <p className="truncate font-display font-semibold leading-tight">
                        {r.code && (
                          <span className="mr-1.5 font-mono text-[0.7rem] text-[var(--color-carbon-300)]">
                            {r.code}
                          </span>
                        )}
                        {r.label}
                      </p>
                      {r.team && (
                        <p className="truncate text-[0.7rem] text-[var(--color-carbon-500)]">
                          {r.team}
                        </p>
                      )}
                    </div>
                  </div>
                </td>
                <td className="tabular px-2 py-2 text-right font-mono text-xs">
                  {r.current.toFixed(0)}
                </td>
                <td className="tabular px-2 py-2 text-right font-mono text-xs text-white">
                  {r.expected.toFixed(0)}
                </td>
                <td className="tabular px-2 py-2 text-right font-mono text-xs">
                  {q50?.toFixed(0)}
                </td>
                <td className="tabular px-2 py-2 text-right font-mono text-[0.7rem] text-[var(--color-carbon-300)]">
                  {q25?.toFixed(0)}–{q75?.toFixed(0)}
                </td>
                <td className="tabular px-2 py-2 text-right font-mono text-[0.7rem] text-[var(--color-carbon-500)]">
                  {q05?.toFixed(0)}–{q95?.toFixed(0)}
                </td>
                <td className="tabular whitespace-nowrap px-2 py-2 text-right font-display font-bold">
                  <span style={{ color: r.p_title > 0.005 ? r.colour : "var(--color-carbon-500)" }}>
                    {pct(r.p_title)}
                  </span>
                  <Delta now={r.p_title} before={previous.get(r.id)} />
                </td>
                <td className="tabular px-2 py-2 text-right font-mono text-xs">{pct(r.p_top3)}</td>
                <td className="w-[28%] py-2 pl-3 pr-6">
                  <HeatStrip probs={r.p_position} label={r.label} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The projected final table at any point of the season: a round slider over the published
 * snapshots (each made with only what was public right after that round), the title-odds
 * river, and the full table — most likely final order, points now, expected and median
 * final points, 50% and 90% ranges, P(title), P(top 3) and every final-position probability.
 */
export function ChampionshipExplorer({ current }: { current: ChampionshipMart | null }) {
  const [hist, setHist] = useState<ChampionshipHistoryMart | null>(null);
  const [kind, setKind] = useState<Kind>("driver");
  const [selected, setSelected] = useState<number | null>(null);

  useEffect(() => {
    let live = true;
    fetch("/data/championship_history.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((h: ChampionshipHistoryMart | null) => {
        if (live && h?.snapshots?.length) setHist(h);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  const snaps = hist?.snapshots ?? [];
  const last = snaps.length - 1;
  const index = selected ?? last;
  // the latest table comes from the current mart (more simulations) when it matches
  const latestFromCurrent =
    current !== null &&
    (index === last || !hist) &&
    (!hist || snaps[last]?.after_round === current.after_round);
  const snap: ChampionshipSnapshot | null = latestFromCurrent
    ? fromCurrent(current)
    : (snaps[index] ?? (current ? fromCurrent(current) : null));
  if (!snap) return null;

  const rows = rowsFor(snap, kind, hist, current);
  const prevSnap = index > 0 ? snaps[index - 1] : undefined;
  const previous = new Map(
    (prevSnap ? (kind === "driver" ? prevSnap.drivers : prevSnap.constructors) : []).map((r) => [
      idOf(r, kind),
      r.p_title,
    ]),
  );
  const roundName = (r: number) => hist?.rounds.find((x) => x.round === r)?.name ?? `Round ${r}`;
  const label = (r: number) => (r === 0 ? "Pre-season" : `After R${r} · ${roundName(r)}`);
  const asOf = new Date(snap.as_of).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });

  return (
    <section className="glass p-6">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Projected final table</p>
          <h2 className="mt-1 font-display text-2xl font-bold">
            {label(snap.after_round)}
            <span className="ml-3 font-mono text-xs font-normal text-[var(--color-carbon-500)]">
              data to {asOf} · {snap.remaining} rounds to go
            </span>
          </h2>
        </div>
        <div className="flex rounded-full bg-white/[0.04] p-1 text-sm" role="tablist">
          {(["driver", "constructor"] as const).map((k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={kind === k}
              onClick={() => setKind(k)}
              className={`rounded-full px-4 py-1.5 font-display font-semibold transition-colors ${
                kind === k
                  ? "bg-[var(--color-signal)] text-[var(--color-carbon-950)]"
                  : "text-[var(--color-carbon-300)] hover:text-white"
              }`}
            >
              {k === "driver" ? "Drivers" : "Constructors"}
            </button>
          ))}
        </div>
      </div>

      {hist && snaps.length > 1 && (
        <div className="mb-6 space-y-3">
          <TitleRiver hist={hist} kind={kind} selected={index} onSelect={setSelected} />
          <div className="flex items-center gap-4">
            <span className="eyebrow shrink-0">Pre-season</span>
            <input
              type="range"
              min={0}
              max={last}
              step={1}
              value={index}
              onChange={(e) => setSelected(Number(e.target.value))}
              className="w-full accent-[var(--color-signal)]"
              aria-label="Projection after round"
              aria-valuetext={label(snaps[index]?.after_round ?? 0)}
            />
            <span className="eyebrow shrink-0">Latest</span>
          </div>
          <p className="text-xs text-[var(--color-carbon-500)]">
            Each snapshot uses only what was public right after that round — drag back to see what
            the model believed then.
          </p>
        </div>
      )}

      <ProjectedTable rows={rows} previous={previous} kind={kind} />
      <p className="mt-4 text-xs text-[var(--color-carbon-500)]">
        # is the jointly most likely final order (an assignment over every position probability, not
        a sort by expected points). Equal points are split on countback. Ranges are of final points.
      </p>
    </section>
  );
}
