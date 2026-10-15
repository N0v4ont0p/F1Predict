"use client";

import { Reorder } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HeatStrip } from "@/components/HeatStrip";
import { ProbBar } from "@/components/ProbBar";
import { pct } from "@/lib/format";
import type { SampleInput, SampleOutput } from "@/lib/sampler";
import { teamColour } from "@/lib/teams";
import type { Car, Checkpoint, CheckpointForecast, EventMart } from "@/lib/types";

export interface LabEvent {
  event_id: string;
  name: string;
  round: number;
  status: string;
}

type Rain = "fc" | "dry" | "wet";
const SAMPLES = 20_000;
const BASE_LAP_S = 90; // seconds; converts a pace change into a share of the lap
const PACE_STEP = 0.1;

interface Scenario {
  rain: Rain;
  out: string[];
  pace: Record<string, number>;
  grid: string[] | null; // driver ids in grid order (C4), null = as qualified
}

const EMPTY: Scenario = { rain: "fc", out: [], pace: {}, grid: null };

const logit = (p: number) =>
  Math.log(Math.min(Math.max(p, 1e-4), 1 - 1e-4) / (1 - Math.min(Math.max(p, 1e-4), 1 - 1e-4)));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** How many units of log-strength one percent of lap time is worth in this field: the slope
 * of the cars' log-strengths on their race-pace signal (clipped to a sane range). */
function paceSlope(fc: CheckpointForecast): number {
  const w = fc.meta.w_race_pace;
  const xs: number[] = [];
  const ys: number[] = [];
  for (const c of fc.cars) {
    const part = c.why?.race_pace;
    if (w && Math.abs(w) > 1e-6 && part !== undefined && c.log_strength != null) {
      xs.push(part / w);
      ys.push(c.log_strength);
    }
  }
  if (xs.length < 5) return 2;
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const my = ys.reduce((a, b) => a + b, 0) / ys.length;
  let sxy = 0;
  let sxx = 0;
  xs.forEach((x, i) => {
    sxy += (x - mx) * ((ys[i] ?? my) - my);
    sxx += (x - mx) ** 2;
  });
  const k = sxx > 0 ? sxy / sxx : 2;
  return Math.min(Math.max(k, 0.5), 5);
}

function scenarioInput(fc: CheckpointForecast, cp: Checkpoint, sc: Scenario): SampleInput {
  const wGrid = cp === "C4" ? (fc.meta.w_grid ?? 0) : 0;
  const k = paceSlope(fc);
  const baseRain = fc.meta.rain ?? 0;
  const rain = sc.rain === "fc" ? baseRain : sc.rain === "wet" ? 1 : 0;
  const fRain = fc.meta.finish_rain ?? 0;
  const order = sc.grid;
  const logStrength = fc.cars.map((c) => {
    let ls = c.log_strength ?? 0;
    if (order && wGrid && c.grid_rank) {
      const slot = order.indexOf(c.driver_id) + 1;
      if (slot > 0) ls += wGrid * (-Math.log(slot) + Math.log(c.grid_rank));
    }
    const dt = sc.pace[c.driver_id] ?? 0; // seconds per lap, negative = faster
    ls += k * ((-dt / BASE_LAP_S) * 100);
    return ls;
  });
  const pFinish = fc.cars.map((c) =>
    sc.out.includes(c.driver_id) ? 0 : sigmoid(logit(c.p_finish) + fRain * (rain - baseRain)),
  );
  return {
    logStrength,
    pFinish,
    stageScales: fc.stage_scales ?? null,
    samples: SAMPLES,
    seed: 20260,
  };
}

function encode(sc: Scenario, codes: Map<string, string>): string {
  const p = new URLSearchParams();
  if (sc.rain !== "fc") p.set("rain", sc.rain);
  if (sc.out.length) p.set("out", sc.out.map((d) => codes.get(d) ?? d).join(","));
  const pace = Object.entries(sc.pace).filter(([, v]) => v !== 0);
  if (pace.length)
    p.set("pace", pace.map(([d, v]) => `${codes.get(d) ?? d}:${v.toFixed(1)}`).join(","));
  if (sc.grid) p.set("grid", sc.grid.map((d) => codes.get(d) ?? d).join(","));
  return p.toString();
}

function decode(p: URLSearchParams, byCode: Map<string, string>): Scenario {
  const id = (c: string) => byCode.get(c.toUpperCase()) ?? c;
  const rain = p.get("rain");
  const pace: Record<string, number> = {};
  for (const part of (p.get("pace") ?? "").split(",").filter(Boolean)) {
    const [c, v] = part.split(":");
    const x = Number(v);
    if (c && Number.isFinite(x)) pace[id(c)] = Math.max(-1, Math.min(1, x));
  }
  const grid = p.get("grid");
  return {
    rain: rain === "dry" || rain === "wet" ? rain : "fc",
    out: (p.get("out") ?? "").split(",").filter(Boolean).map(id),
    pace,
    grid: grid ? grid.split(",").filter(Boolean).map(id) : null,
  };
}

/** The What-if Lab: change the race and the forecast re-runs in the browser — the same
 * staged Plackett-Luce sampler as the Python model, in a Web Worker. */
export function WhatIfLab({ events, initial }: { events: LabEvent[]; initial: string }) {
  const [eventId, setEventId] = useState(initial);
  const [ev, setEv] = useState<EventMart | null>(null);
  const [cp, setCp] = useState<Checkpoint>("C4");
  const [sc, setSc] = useState<Scenario>(EMPTY);
  const [result, setResult] = useState<SampleOutput | null>(null);
  const [busy, setBusy] = useState(false);
  const worker = useRef<Worker | null>(null);
  const request = useRef(0);
  const hydrated = useRef(false);

  // read the scenario from the URL once
  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    const e = p.get("e");
    if (e && events.some((x) => x.event_id === e)) setEventId(e);
    const c = p.get("cp");
    if (c === "C1" || c === "C4") setCp(c);
  }, [events]);

  useEffect(() => {
    let live = true;
    fetch(`/data/events/${eventId}.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: EventMart | null) => {
        if (!live || !data) return;
        setEv(data);
        const available = (["C1", "C4"] as const).filter((c) => data.checkpoints[c]);
        setCp((cur) => (data.checkpoints[cur] ? cur : (available.at(-1) ?? "C1")));
      });
    return () => {
      live = false;
    };
  }, [eventId]);

  const fc = ev?.checkpoints[cp] ?? null;
  const codes = useMemo(() => new Map((fc?.cars ?? []).map((c) => [c.driver_id, c.code])), [fc]);
  const byCode = useMemo(
    () => new Map((fc?.cars ?? []).map((c) => [c.code.toUpperCase(), c.driver_id])),
    [fc],
  );

  // the scenario in the URL (after the first event has loaded)
  useEffect(() => {
    if (!fc) return;
    if (!hydrated.current) {
      hydrated.current = true;
      setSc(decode(new URLSearchParams(window.location.search), byCode));
      return;
    }
    const q = encode(sc, codes);
    const url = `${window.location.pathname}?e=${eventId}&cp=${cp}${q ? `&${q}` : ""}`;
    window.history.replaceState(null, "", url);
  }, [sc, eventId, cp, fc, codes, byCode]);

  useEffect(() => {
    const w = new Worker(new URL("../lib/sampler.worker.ts", import.meta.url), {
      type: "module",
    });
    w.onmessage = (e: MessageEvent<{ id: number; output: SampleOutput }>) => {
      if (e.data.id === request.current) {
        setResult(e.data.output);
        setBusy(false);
      }
    };
    worker.current = w;
    return () => w.terminate();
  }, []);

  useEffect(() => {
    if (!fc || !worker.current || fc.cars.some((c) => c.log_strength == null)) return;
    const id = request.current + 1;
    request.current = id;
    setBusy(true);
    const t = setTimeout(
      () => worker.current?.postMessage({ id, input: scenarioInput(fc, cp, sc) }),
      60,
    );
    return () => clearTimeout(t);
  }, [fc, cp, sc]);

  const changeEvent = useCallback((id: string) => {
    setEventId(id);
    setSc(EMPTY);
    setResult(null);
  }, []);

  if (!ev || !fc) {
    return <div className="glass p-8 text-[var(--color-carbon-300)]">Loading the forecast…</div>;
  }
  if (fc.cars.some((c) => c.log_strength == null)) {
    return (
      <div className="glass p-8 text-[var(--color-carbon-300)]">
        These forecasts were published before the Lab existed — run{" "}
        <code className="font-mono">f1p model publish</code> to refresh them.
      </div>
    );
  }

  const gridOrder =
    sc.grid ??
    [...fc.cars]
      .sort((a, b) => (a.grid_rank ?? 99) - (b.grid_rank ?? 99) || a.rank - b.rank)
      .map((c) => c.driver_id);
  const byId = new Map(fc.cars.map((c) => [c.driver_id, c]));
  const canDrag = cp === "C4" && fc.cars.some((c) => c.grid_rank);
  const rows = result
    ? fc.cars
        .map((c, i) => ({
          car: c,
          pWin: result.pWin[i] ?? 0,
          probs: result.positionProbs[i] ?? [],
        }))
        .sort((a, b) => b.pWin - a.pWin)
    : fc.cars.map((c) => ({ car: c, pWin: c.p_win, probs: c.position_probs }));
  const changed =
    sc.out.length +
    Object.values(sc.pace).filter(Boolean).length +
    (sc.grid ? 1 : 0) +
    (sc.rain !== "fc" ? 1 : 0);
  const available = (["C1", "C4"] as const).filter((c) => ev.checkpoints[c]);

  const setPace = (d: string, delta: number) =>
    setSc((s) => {
      const v = Math.round(((s.pace[d] ?? 0) + delta) * 10) / 10;
      return { ...s, pace: { ...s.pace, [d]: Math.max(-1, Math.min(1, v)) } };
    });
  const toggleOut = (d: string) =>
    setSc((s) => ({ ...s, out: s.out.includes(d) ? s.out.filter((x) => x !== d) : [...s.out, d] }));

  return (
    <div className="space-y-6">
      <header className="glass flex flex-wrap items-end justify-between gap-4 p-6">
        <div>
          <p className="eyebrow">What-if Lab</p>
          <h1 className="mt-2 font-display text-4xl font-bold leading-none tracking-tight">
            Change the race
          </h1>
          <p className="mt-2 max-w-xl text-sm text-[var(--color-carbon-300)]">
            Move the grid, find a few tenths, make it rain or put a car out. The forecast re-runs
            here in your browser: {SAMPLES.toLocaleString("en-GB")} races with the model&apos;s own
            sampler.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={eventId}
            onChange={(e) => changeEvent(e.target.value)}
            className="rounded-full border border-white/10 bg-[var(--color-carbon-850)] px-4 py-2 font-display text-sm"
            aria-label="Race"
          >
            {events.map((e) => (
              <option key={e.event_id} value={e.event_id}>
                R{e.round} · {e.name}
                {e.status === "next" ? " (next)" : ""}
              </option>
            ))}
          </select>
          <div
            className="flex rounded-full bg-[var(--color-carbon-850)] p-1"
            role="tablist"
            aria-label="Checkpoint"
          >
            {available.map((c) => (
              <button
                key={c}
                type="button"
                role="tab"
                aria-selected={c === cp}
                onClick={() => {
                  setCp(c);
                  setSc((s) => ({ ...s, grid: null }));
                }}
                className={`rounded-full px-3 py-1.5 font-mono text-xs ${c === cp ? "bg-[var(--color-signal)] text-[var(--color-carbon-950)]" : "text-[var(--color-carbon-300)]"}`}
              >
                {c}
              </button>
            ))}
          </div>
          <fieldset
            className="flex rounded-full bg-[var(--color-carbon-850)] p-1"
            aria-label="Weather"
          >
            {(
              [
                ["dry", "Dry"],
                ["fc", "Forecast"],
                ["wet", "Wet"],
              ] as const
            ).map(([k, label]) => (
              <button
                key={k}
                type="button"
                aria-pressed={sc.rain === k}
                onClick={() => setSc((s) => ({ ...s, rain: k }))}
                className={`rounded-full px-3 py-1.5 font-display text-xs font-semibold ${sc.rain === k ? "bg-[var(--color-rain)] text-[var(--color-carbon-950)]" : "text-[var(--color-carbon-300)]"}`}
              >
                {label}
              </button>
            ))}
          </fieldset>
          <button
            type="button"
            disabled={!changed}
            onClick={() => setSc(EMPTY)}
            className="rounded-full border border-white/10 px-4 py-1.5 font-display text-xs font-semibold text-[var(--color-carbon-300)] enabled:hover:text-white disabled:opacity-40"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={() => navigator.clipboard?.writeText(window.location.href)}
            className="rounded-full bg-white px-4 py-1.5 font-display text-xs font-semibold text-[var(--color-carbon-950)]"
          >
            Copy link
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.3fr)]">
        <section className="glass p-5" aria-label="Scenario">
          <p className="eyebrow">{canDrag ? "Grid · drag to reorder" : "Cars"}</p>
          <p className="mt-1 text-xs text-[var(--color-carbon-500)]">
            Pace in seconds per lap (− is faster). The ✕ takes a car out of the race.
          </p>
          <Reorder.Group
            axis="y"
            values={gridOrder}
            onReorder={(order: string[]) => canDrag && setSc((s) => ({ ...s, grid: order }))}
            className="mt-4 space-y-1.5"
          >
            {gridOrder.map((d, slot) => {
              const c = byId.get(d) as Car;
              if (!c) return null;
              const colour = teamColour(c.lineage);
              const out = sc.out.includes(d);
              const pace = sc.pace[d] ?? 0;
              const moved = canDrag && c.grid_rank && sc.grid ? slot + 1 - c.grid_rank : 0;
              return (
                <Reorder.Item
                  key={d}
                  value={d}
                  dragListener={canDrag}
                  className={`flex items-center gap-3 rounded-lg border border-white/5 bg-[var(--color-carbon-900)] px-3 py-2 ${canDrag ? "cursor-grab active:cursor-grabbing" : ""} ${out ? "opacity-40" : ""}`}
                >
                  <span className="tabular w-6 text-right font-display font-bold text-[var(--color-carbon-300)]">
                    {canDrag ? slot + 1 : ""}
                  </span>
                  <span className="h-6 w-1 rounded-full" style={{ background: colour }} />
                  <span className="min-w-0 flex-1 truncate font-display text-sm font-semibold">
                    <span className="mr-1.5 font-mono text-xs text-[var(--color-carbon-300)]">
                      {c.code}
                    </span>
                    {c.name}
                    {moved !== 0 && (
                      <span
                        className={`ml-2 font-mono text-[0.65rem] ${moved < 0 ? "text-[var(--color-green)]" : "text-[var(--color-flag)]"}`}
                      >
                        {moved < 0 ? `▲${-moved}` : `▼${moved}`}
                      </span>
                    )}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      aria-label={`${c.code} faster`}
                      onClick={() => setPace(d, -PACE_STEP)}
                      className="h-6 w-6 rounded bg-white/5 font-mono text-xs hover:bg-white/10"
                    >
                      −
                    </button>
                    <span
                      className={`tabular w-12 text-center font-mono text-xs ${pace < 0 ? "text-[var(--color-green)]" : pace > 0 ? "text-[var(--color-flag)]" : "text-[var(--color-carbon-500)]"}`}
                    >
                      {pace === 0 ? "±0.0" : `${pace > 0 ? "+" : ""}${pace.toFixed(1)}`}
                    </span>
                    <button
                      type="button"
                      aria-label={`${c.code} slower`}
                      onClick={() => setPace(d, PACE_STEP)}
                      className="h-6 w-6 rounded bg-white/5 font-mono text-xs hover:bg-white/10"
                    >
                      +
                    </button>
                    <button
                      type="button"
                      aria-label={out ? `put ${c.code} back in` : `take ${c.code} out`}
                      aria-pressed={out}
                      onClick={() => toggleOut(d)}
                      className={`ml-1 h-6 w-6 rounded font-mono text-xs ${out ? "bg-[var(--color-flag)] text-white" : "bg-white/5 text-[var(--color-carbon-300)] hover:bg-white/10"}`}
                    >
                      ✕
                    </button>
                  </div>
                </Reorder.Item>
              );
            })}
          </Reorder.Group>
        </section>

        <section className="glass p-5" aria-label="Outcome" aria-busy={busy}>
          <div className="flex items-center justify-between">
            <p className="eyebrow">Outcome {changed ? "· vs the published forecast" : ""}</p>
            <span
              className={`h-2 w-2 rounded-full ${busy ? "animate-pulse bg-[var(--color-signal)]" : "bg-[var(--color-green)]"}`}
              title={busy ? "simulating" : "up to date"}
            />
          </div>
          <ol className="mt-3 divide-y divide-white/[0.04]">
            {rows.map(({ car, pWin, probs }, i) => {
              const colour = teamColour(car.lineage);
              const delta = pWin - car.p_win;
              const podium = probs.slice(0, 3).reduce((a, b) => a + b, 0);
              const points = probs.slice(0, 10).reduce((a, b) => a + b, 0);
              return (
                <li
                  key={car.driver_id}
                  className="grid grid-cols-[1.5rem_minmax(0,7rem)_minmax(0,1fr)_3.5rem] items-center gap-3 py-2 sm:grid-cols-[1.5rem_minmax(0,8rem)_9rem_3.5rem_3.5rem_minmax(0,1fr)]"
                >
                  <span className="tabular font-display font-bold text-[var(--color-carbon-300)]">
                    {i + 1}
                  </span>
                  <span className="truncate font-display text-sm font-semibold">
                    <span className="mr-1.5 font-mono text-xs text-[var(--color-carbon-300)]">
                      {car.code}
                    </span>
                    {car.name}
                  </span>
                  <ProbBar p={pWin} colour={colour} wide />
                  <span
                    className={`tabular text-right font-mono text-[0.7rem] ${Math.abs(delta) < 0.0005 ? "text-[var(--color-carbon-500)]" : delta > 0 ? "text-[var(--color-green)]" : "text-[var(--color-flag)]"}`}
                  >
                    {Math.abs(delta) < 0.0005
                      ? "—"
                      : `${delta > 0 ? "+" : "−"}${(Math.abs(delta) * 100).toFixed(1)}`}
                  </span>
                  <span
                    className="tabular hidden text-right font-mono text-xs sm:block"
                    title="podium"
                  >
                    {pct(podium, 0)}
                  </span>
                  <div className="col-span-4 sm:col-span-1">
                    <HeatStrip probs={probs} label={car.name} />
                  </div>
                  <span className="sr-only">points {pct(points)}</span>
                </li>
              );
            })}
          </ol>
          <p className="mt-4 text-xs leading-relaxed text-[var(--color-carbon-500)]">
            Win odds with the change in points from the published forecast; strips show every
            finishing position. A tenth per lap is converted with this field&apos;s own slope of
            strength on race pace; rain changes the retirement odds as fitted on past races; moving
            the grid uses the fitted grid weight (C4 only). Swapping drivers between cars and the
            championship impact are not modelled here yet.
          </p>
        </section>
      </div>
    </div>
  );
}
