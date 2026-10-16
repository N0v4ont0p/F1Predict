"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_F1P_DEV_API ?? "http://127.0.0.1:8765";
const POLL_MS = 1500;

type Task = {
  id: string;
  group: string;
  title: string;
  detail: string;
  steps: string[][];
  after: string[];
  manual: string;
  confirm: string;
  hours: number;
  runnable: boolean;
};
type Job = {
  id: string;
  task: string;
  pipeline: string | null;
  status: "queued" | "running" | "ok" | "failed" | "cancelled";
  started: number | null;
  finished: number | null;
  elapsed: number;
  step: number;
  steps: number;
  command: string;
  returncode: number | null;
  progress: number | null;
  lines: number;
  last: string;
};
type Sample = {
  t: number;
  load: number;
  cpu: number;
  mem_used: number;
  mem_total: number;
  speed_limit: number | null;
};
type State = {
  version: string;
  parallel: number;
  tasks: Task[];
  pipelines: Record<string, string[]>;
  jobs: Job[];
  latest: Record<string, { id: string; status: Job["status"]; finished: number | null }>;
  system: { cpus: number; samples: Sample[] };
};

const PIPELINE_LABELS: Record<string, [string, string]> = {
  checks: ["Run all checks", "format · lint · types · boundaries · tests · smoke"],
  hardware: ["Hardware truth", "B1–B3, one evening"],
  m6: ["Full M6 run", "B1 → B11 in the plan's order; stops at the first failure"],
};

const STATUS_STYLE: Record<string, string> = {
  queued: "bg-white/5 text-[var(--color-carbon-300)]",
  running: "bg-[var(--color-signal)]/15 text-[var(--color-signal)]",
  ok: "bg-[var(--color-green)]/15 text-[var(--color-green)]",
  failed: "bg-[var(--color-flag)]/15 text-[var(--color-flag)]",
  cancelled: "bg-[var(--color-heat)]/15 text-[var(--color-heat)]",
};

async function post(path: string, body: unknown = {}) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-F1P-Dev": "1" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error ?? `HTTP ${r.status}`);
  return data;
}

function duration(s: number) {
  if (!s) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h
    ? `${h}h ${String(m).padStart(2, "0")}m`
    : m
      ? `${m}m ${String(sec).padStart(2, "0")}s`
      : `${sec}s`;
}

function Chip({ status }: { status: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-mono text-[0.68rem] uppercase tracking-wider ${STATUS_STYLE[status] ?? STATUS_STYLE.queued}`}
    >
      {status === "running" && (
        <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-current" />
      )}
      {status}
    </span>
  );
}

function Spark({ values, max }: { values: number[]; max: number }) {
  if (values.length < 2) return <div className="h-8" />;
  const w = 160;
  const h = 32;
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * w},${h - (Math.min(v, max) / max) * (h - 2) - 1}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-8 w-full" preserveAspectRatio="none" aria-hidden>
      <polyline
        points={pts}
        fill="none"
        stroke="var(--color-signal)"
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

function Machine({ system }: { system: State["system"] }) {
  const s = system.samples;
  const last = s[s.length - 1];
  if (!last) return null;
  const tiles = [
    {
      label: `CPU load · ${system.cpus} cores`,
      value: `${Math.round(last.cpu * 100)}%`,
      sub: `load ${last.load.toFixed(2)}`,
      spark: <Spark values={s.map((x) => x.cpu)} max={1} />,
    },
    {
      label: "Memory",
      value: `${last.mem_used.toFixed(1)} GB`,
      sub: `of ${last.mem_total.toFixed(0)} GB`,
      spark: <Spark values={s.map((x) => x.mem_used)} max={last.mem_total || 1} />,
    },
    {
      label: "Thermal",
      value: last.speed_limit === null ? "n/a" : `${last.speed_limit}%`,
      sub:
        last.speed_limit === null
          ? "macOS only"
          : last.speed_limit < 100
            ? "throttling"
            : "full speed",
      spark: <Spark values={s.map((x) => x.speed_limit ?? 100)} max={100} />,
    },
  ];
  return (
    <section className="grid gap-3 sm:grid-cols-3">
      {tiles.map((t) => (
        <div key={t.label} className="glass p-4">
          <p className="eyebrow">{t.label}</p>
          <p className="tabular mt-1 font-display text-2xl font-bold">
            {t.value}{" "}
            <span className="text-sm font-normal text-[var(--color-carbon-300)]">{t.sub}</span>
          </p>
          <div className="mt-2">{t.spark}</div>
        </div>
      ))}
    </section>
  );
}

function LogView({ job, onClose }: { job: Job; onClose: () => void }) {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState<string>(job.status);
  const [follow, setFollow] = useState(true);
  const [filter, setFilter] = useState("");
  const box = useRef<HTMLPreElement>(null);

  useEffect(() => {
    setLines([]);
    let next = 0;
    let es: EventSource | null = null;
    let closed = false;
    const open = () => {
      if (closed) return;
      es = new EventSource(`${API}/api/jobs/${job.id}/stream?from=${next}`);
      es.onmessage = (e) => {
        const d = JSON.parse(e.data) as { lines: string[]; next: number; status: string };
        next = d.next;
        setStatus(d.status);
        if (d.lines.length) setLines((prev) => [...prev, ...d.lines].slice(-20000));
        if (d.status !== "queued" && d.status !== "running") {
          closed = true;
          es?.close();
        }
      };
      es.onerror = () => {
        es?.close();
        if (!closed) setTimeout(open, 2000);
      };
    };
    open();
    return () => {
      closed = true;
      es?.close();
    };
  }, [job.id]);

  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  });

  const shown = filter
    ? lines.filter((l) => l.toLowerCase().includes(filter.toLowerCase()))
    : lines;

  return (
    <section className="glass flex flex-col overflow-hidden">
      <div className="flex flex-wrap items-center gap-3 border-b border-white/5 px-4 py-3">
        <Chip status={status} />
        <p className="min-w-0 flex-1 truncate font-mono text-xs text-[var(--color-carbon-300)]">
          {job.id}
        </p>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="filter"
          className="w-28 rounded-md border border-white/10 bg-black/30 px-2 py-1 font-mono text-xs"
        />
        <label className="flex items-center gap-1.5 text-xs text-[var(--color-carbon-300)]">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          follow
        </label>
        <a
          href={`${API}/api/jobs/${job.id}/log?from=0`}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-[var(--color-signal)] hover:underline"
        >
          raw
        </a>
        <button
          type="button"
          onClick={onClose}
          className="text-xs text-[var(--color-carbon-300)] hover:text-white"
        >
          close
        </button>
      </div>
      <pre
        ref={box}
        className="h-[28rem] overflow-auto bg-black/40 p-4 font-mono text-[0.72rem] leading-relaxed text-[var(--color-carbon-100)]"
      >
        {shown.map((l, i) => (
          <div
            key={i}
            className={
              l.startsWith("$ ")
                ? "mt-2 text-[var(--color-signal)]"
                : /error|failed|traceback|exit code/i.test(l)
                  ? "text-[var(--color-flag)]"
                  : /warn/i.test(l)
                    ? "text-[var(--color-heat)]"
                    : undefined
            }
          >
            {l || " "}
          </div>
        ))}
        {!shown.length && (
          <span className="text-[var(--color-carbon-500)]">waiting for output…</span>
        )}
      </pre>
      <p className="border-t border-white/5 px-4 py-2 font-mono text-[0.68rem] text-[var(--color-carbon-500)]">
        {lines.length.toLocaleString()} lines
      </p>
    </section>
  );
}

export function DevPanel() {
  const [state, setState] = useState<State | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/state`, { cache: "no-store" });
      setState(await r.json());
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const act = async (fn: () => Promise<{ job?: Job; jobs?: Job[] }>) => {
    setError(null);
    try {
      const res = await fn();
      const first = res.job ?? res.jobs?.[0];
      if (first) setSelected(first.id);
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const run = (t: Task) => {
    const latest = state?.latest ?? {};
    const missing = t.after.filter((a) => latest[a]?.status !== "ok");
    if (
      missing.length &&
      !window.confirm(
        `${t.id} should come after ${missing.join(", ")}, which have not passed yet. Run anyway?`,
      )
    )
      return;
    let confirm: string | undefined;
    if (t.confirm) {
      const typed = window.prompt(
        `${t.title}: this cannot be undone.\nType "${t.confirm}" to start.`,
      );
      if (typed === null) return;
      confirm = typed;
    }
    act(() => post("/api/run", { task: t.id, confirm }));
  };

  const groups = useMemo(() => {
    const g = new Map<string, Task[]>();
    for (const t of state?.tasks ?? []) g.set(t.group, [...(g.get(t.group) ?? []), t]);
    return [...g.entries()];
  }, [state?.tasks]);

  const selectedJob = state?.jobs.find((j) => j.id === selected) ?? null;
  const running = state?.jobs.filter((j) => j.status === "running") ?? [];
  const queued = state?.jobs.filter((j) => j.status === "queued") ?? [];

  if (online === false)
    return (
      <div className="glass space-y-3 p-8">
        <h2 className="font-display text-2xl font-bold">The dev server is not running</h2>
        <p className="text-sm text-[var(--color-carbon-300)]">
          Start it from the repository root, then this page connects on its own:
        </p>
        <pre className="rounded-lg bg-black/40 p-4 font-mono text-sm">
          just dev{"\n"}# or, separately:{"\n"}uv run f1p dev serve{"   "}# API on {API}
          {"\n"}just web
        </pre>
      </div>
    );
  if (!state) return <p className="eyebrow">connecting to {API}…</p>;

  return (
    <div className="space-y-6">
      <Machine system={state.system} />

      <section className="glass flex flex-wrap items-center gap-3 p-4">
        {Object.keys(state.pipelines).map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => act(() => post("/api/pipeline", { name }))}
            className="group rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-left transition hover:border-[var(--color-signal)]/50 hover:bg-[var(--color-signal)]/10"
          >
            <span className="block font-display font-bold">
              ▶ {PIPELINE_LABELS[name]?.[0] ?? name}
            </span>
            <span className="block text-xs text-[var(--color-carbon-300)]">
              {PIPELINE_LABELS[name]?.[1] ?? state.pipelines[name]?.join(" → ")}
            </span>
          </button>
        ))}
        <div className="ml-auto flex items-center gap-4 font-mono text-xs text-[var(--color-carbon-300)]">
          <span>
            {running.length} running · {queued.length} queued · {state.parallel} at a time
          </span>
          {queued.length > 0 && (
            <button
              type="button"
              onClick={() => act(() => post("/api/queue/clear"))}
              className="text-[var(--color-heat)] hover:underline"
            >
              clear queue
            </button>
          )}
        </div>
      </section>

      {error && (
        <p className="rounded-lg bg-[var(--color-flag)]/15 px-4 py-2 text-sm text-[var(--color-flag)]">
          {error}
        </p>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-6">
          {groups.map(([group, tasks]) => (
            <section key={group} className="glass overflow-hidden">
              <h2 className="eyebrow px-4 pt-4">
                {group === "M6" ? "M6 checklist (docs/plan/M6_FOLLOWUP_PLAN.md)" : group}
              </h2>
              <ul className="mt-2 divide-y divide-white/5">
                {tasks.map((t) => {
                  const latest = state.latest[t.id];
                  return (
                    <li key={t.id} className="flex items-start gap-3 px-4 py-3">
                      <span className="w-9 shrink-0 font-mono text-xs text-[var(--color-carbon-500)]">
                        {t.id.length <= 3 ? t.id : ""}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="flex flex-wrap items-center gap-2 text-sm font-semibold">
                          {t.title}
                          {latest && (
                            <button type="button" onClick={() => setSelected(latest.id)}>
                              <Chip status={latest.status} />
                            </button>
                          )}
                        </p>
                        <p className="text-xs text-[var(--color-carbon-300)]">
                          {t.manual ? `Manual: ${t.manual}` : t.detail}
                          {t.hours > 0 &&
                            ` · ~${t.hours < 1 ? `${Math.round(t.hours * 60)} min` : `${t.hours} h`}`}
                          {t.after.length > 0 && ` · after ${t.after.join(", ")}`}
                        </p>
                      </div>
                      <button
                        type="button"
                        disabled={!t.runnable}
                        onClick={() => run(t)}
                        title={t.steps.map((s) => s.join(" ")).join("\n")}
                        className={`shrink-0 rounded-lg px-3 py-1 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-30 ${
                          t.confirm
                            ? "border border-[var(--color-flag)]/50 text-[var(--color-flag)] hover:bg-[var(--color-flag)]/10"
                            : "bg-white/5 hover:bg-[var(--color-signal)]/15 hover:text-[var(--color-signal)]"
                        }`}
                      >
                        {t.confirm ? "Run once…" : "Run"}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </div>

        <div className="space-y-6">
          {selectedJob && (
            <LogView key={selectedJob.id} job={selectedJob} onClose={() => setSelected(null)} />
          )}
          <section className="glass overflow-hidden">
            <h2 className="eyebrow px-4 pt-4">Jobs</h2>
            {state.jobs.length === 0 && (
              <p className="px-4 py-6 text-sm text-[var(--color-carbon-300)]">
                Nothing has run yet. Start with “Run all checks”.
              </p>
            )}
            <ul className="mt-2 divide-y divide-white/5">
              {state.jobs.slice(0, 60).map((j) => (
                <li key={j.id}>
                  <div
                    className={`flex items-center gap-3 px-4 py-2.5 ${j.id === selected ? "bg-white/5" : ""}`}
                  >
                    <button
                      type="button"
                      onClick={() => setSelected(j.id)}
                      className="flex min-w-0 flex-1 items-center gap-3 text-left"
                    >
                      <Chip status={j.status} />
                      <span className="w-12 shrink-0 font-mono text-xs font-bold">{j.task}</span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-mono text-[0.7rem] text-[var(--color-carbon-300)]">
                          {j.status === "running"
                            ? j.last || j.command
                            : j.command || (j.pipeline ? `pipeline ${j.pipeline}` : "")}
                        </span>
                        {j.status === "running" && (
                          <span className="mt-1 block h-1 overflow-hidden rounded-full bg-white/5">
                            <span
                              className={`block h-full bg-[var(--color-signal)] transition-all ${j.progress === null ? "w-1/3 animate-pulse-soft" : ""}`}
                              style={
                                j.progress === null
                                  ? undefined
                                  : {
                                      width: `${((j.step - 1 + j.progress) / Math.max(1, j.steps)) * 100}%`,
                                    }
                              }
                            />
                          </span>
                        )}
                      </span>
                      <span className="tabular shrink-0 font-mono text-xs text-[var(--color-carbon-300)]">
                        {j.steps > 1 && `${j.step}/${j.steps} · `}
                        {duration(j.elapsed)}
                      </span>
                    </button>
                    {(j.status === "running" || j.status === "queued") && (
                      <button
                        type="button"
                        onClick={() => act(() => post(`/api/jobs/${j.id}/cancel`))}
                        className="shrink-0 text-xs text-[var(--color-flag)] hover:underline"
                      >
                        stop
                      </button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
