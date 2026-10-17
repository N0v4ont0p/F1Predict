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
  preset: string | null;
  step_elapsed: number;
  idle: number;
  cpu: number | null;
  rss_mb: number | null;
  cpu_hist: number[];
};
type Sample = {
  t: number;
  load: number;
  cpu: number;
  mem_used: number;
  mem_total: number;
  speed_limit: number | null;
};
type Preset = {
  id: string;
  title: string;
  detail: string;
  main: boolean;
  tasks: string[];
  hours: number;
};
type State = {
  version: string;
  parallel: number;
  paused: boolean;
  tasks: Task[];
  presets: Preset[];
  jobs: Job[];
  latest: Record<string, { id: string; status: Job["status"]; finished: number | null }>;
  system: { cpus: number; samples: Sample[] };
  reports: { path: string; modified: number }[];
};
type Act = (fn: () => Promise<{ job?: Job; jobs?: Job[] }>) => void;

// Long presets ask before starting; the rest start on one click.
const ASK_FIRST = new Set(["everything", "overnight", "retrain", "m6"]);

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
              l.startsWith("⋯")
                ? "italic text-[var(--color-carbon-500)]"
                : l.startsWith("$ ")
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

function hoursLabel(h: number) {
  if (h <= 0) return "";
  if (h < 1) return `~${Math.max(1, Math.round(h * 60))} min`;
  return `~${h % 1 === 0 ? h : h.toFixed(1)} h`;
}

const SEGMENT: Record<string, string> = {
  ok: "bg-[var(--color-green)]",
  failed: "bg-[var(--color-flag)]",
  cancelled: "bg-[var(--color-heat)]/70",
  running: "animate-pulse-soft bg-[var(--color-signal)]",
  queued: "bg-white/10",
};

/** The latest run of a preset: its jobs in pipeline order, and a summary. */
function latestRun(state: State, presetId: string) {
  const first = state.jobs.find((j) => j.preset === presetId);
  if (!first) return null;
  const jobs = state.jobs.filter((j) => j.pipeline === first.pipeline).reverse();
  const count = (s: Job["status"]) => jobs.filter((j) => j.status === s).length;
  const active = count("running") + count("queued") > 0;
  const started = Math.min(...jobs.map((j) => j.started ?? Number.POSITIVE_INFINITY));
  const ended = Math.max(...jobs.map((j) => j.finished ?? 0));
  return {
    tag: first.pipeline as string,
    jobs,
    active,
    running: jobs.find((j) => j.status === "running") ?? null,
    done: jobs.length - count("running") - count("queued"),
    ok: count("ok"),
    failed: count("failed"),
    cancelled: count("cancelled"),
    elapsed: Number.isFinite(started)
      ? (active ? Date.now() / 1000 : ended || Date.now() / 1000) - started
      : 0,
  };
}

function Btn({
  children,
  onClick,
  tone = "plain",
  big = false,
  disabled = false,
  title,
}: {
  children: React.ReactNode;
  onClick: () => void;
  tone?: "primary" | "plain" | "danger";
  big?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  const tones = {
    primary:
      "bg-[var(--color-signal)] text-[var(--color-carbon-950)] hover:brightness-110 font-bold",
    plain: "border border-white/15 bg-white/5 hover:bg-white/10",
    danger:
      "border border-[var(--color-flag)]/50 text-[var(--color-flag)] hover:bg-[var(--color-flag)]/10",
  };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`rounded-lg font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${
        big ? "px-5 py-2.5 text-base" : "px-3 py-1.5 text-sm"
      } ${tones[tone]}`}
    >
      {children}
    </button>
  );
}

function PresetCard({
  preset,
  state,
  act,
  onSelect,
  big = false,
}: {
  preset: Preset;
  state: State;
  act: Act;
  onSelect: (jobId: string) => void;
  big?: boolean;
}) {
  const run = latestRun(state, preset.id);
  const titles = Object.fromEntries(state.tasks.map((t) => [t.id, t.title]));
  const confirmed = (what: string) =>
    !ASK_FIRST.has(preset.id) ||
    window.confirm(
      `${what} "${preset.title}"? It takes ${hoursLabel(preset.hours) || "a while"}. ` +
        "Keep the Mac plugged in and the Terminal window open.",
    );
  const start = () => {
    if (confirmed("Start")) act(() => post("/api/pipeline", { name: preset.id }));
  };
  const stop = () => {
    if (run && window.confirm(`Stop "${preset.title}"? The current step is aborted.`))
      act(() => post("/api/pipeline/stop", { tag: run.tag }));
  };
  const restart = () => {
    if (run && window.confirm(`Abort "${preset.title}" and start it again from the beginning?`))
      act(() => post("/api/pipeline/restart", { tag: run.tag }));
  };
  const retry = () => {
    if (run) act(() => post("/api/pipeline/retry", { tag: run.tag }));
  };
  const pause = () => act(() => post("/api/pause", { paused: !state.paused }));

  let status = "Not run yet";
  let tone = "text-[var(--color-carbon-300)]";
  if (run?.active) {
    status = state.paused ? "Paused" : "Running";
    tone = state.paused ? "text-[var(--color-heat)]" : "text-[var(--color-signal)]";
  } else if (run) {
    if (run.failed) {
      status = `Finished · ${run.failed} failed`;
      tone = "text-[var(--color-flag)]";
    } else if (run.cancelled) {
      status = run.ok ? `Stopped · ${run.ok} done` : "Stopped";
      tone = "text-[var(--color-heat)]";
    } else {
      status = "Done";
      tone = "text-[var(--color-green)]";
    }
  }

  return (
    <section
      className={`glass relative flex flex-col overflow-hidden ${big ? "p-6" : "p-5"} ${
        run?.active ? "ring-1 ring-[var(--color-signal)]/50" : ""
      }`}
    >
      {big && <span className="absolute inset-y-0 left-0 w-1 bg-[var(--color-signal)]" />}
      <h3 className={`font-display font-bold ${big ? "text-2xl" : "text-lg"}`}>{preset.title}</h3>
      <p className="mt-0.5 font-mono text-xs uppercase tracking-wider">
        <span className={tone}>
          {status}
          {run && run.elapsed > 0 && ` · ${duration(run.elapsed)}`}
        </span>
        <span className="text-[var(--color-carbon-500)]">
          {" "}
          · {preset.tasks.length} steps · {hoursLabel(preset.hours)}
        </span>
      </p>
      <p className={`mt-2 text-[var(--color-carbon-300)] ${big ? "text-sm" : "text-xs"}`}>
        {preset.detail}
      </p>

      {run && (
        <div className="mt-4">
          <div className="flex h-2.5 gap-0.5 overflow-hidden rounded-full">
            {run.jobs.map((j) => (
              <button
                key={j.id}
                type="button"
                onClick={() => onSelect(j.id)}
                title={`${titles[j.task] ?? j.task}: ${j.status}`}
                className={`flex-1 ${SEGMENT[j.status] ?? SEGMENT.queued}`}
              />
            ))}
          </div>
          <p className="mt-1.5 truncate text-xs text-[var(--color-carbon-300)]">
            {run.running
              ? `Step ${run.jobs.indexOf(run.running) + 1} of ${run.jobs.length}: ${
                  titles[run.running.task] ?? run.running.task
                }`
              : [
                  run.ok && `${run.ok} done`,
                  run.failed && `${run.failed} failed`,
                  run.cancelled && `${run.cancelled} stopped or skipped`,
                ]
                  .filter(Boolean)
                  .join(" · ")}
          </p>
        </div>
      )}

      <div className="mt-auto flex flex-wrap gap-2 pt-4">
        {run?.active ? (
          <>
            <Btn onClick={pause} tone="plain" big={big}>
              {state.paused ? "▶ Resume" : "⏸ Pause"}
            </Btn>
            <Btn onClick={stop} tone="danger" big={big}>
              ■ Stop
            </Btn>
            <Btn onClick={restart} tone="plain" big={big}>
              ⟲ Restart
            </Btn>
          </>
        ) : (
          <>
            <Btn onClick={start} tone="primary" big={big}>
              ▶ Start
            </Btn>
            {run && run.ok < run.jobs.length && (
              <Btn
                onClick={retry}
                tone="plain"
                big={big}
                title="Run again only the steps that failed or were stopped"
              >
                ↻ Retry failed
              </Btn>
            )}
          </>
        )}
      </div>
    </section>
  );
}

function health(job: Job): { label: string; detail: string; tone: string } {
  const cpu = job.cpu ?? 0;
  if (cpu >= 10 || job.idle < 90)
    return {
      label: "Working",
      detail: cpu >= 10 ? "busy computing" : "printing output",
      tone: "bg-[var(--color-green)]/15 text-[var(--color-green)]",
    };
  if (job.idle < 600)
    return {
      label: "Waiting",
      detail: "quiet and idle: probably downloading or waiting on a rate limit",
      tone: "bg-[var(--color-heat)]/15 text-[var(--color-heat)]",
    };
  return {
    label: "Possibly stuck",
    detail: `no output and no CPU for ${duration(job.idle)}; Stop and Retry if it stays like this`,
    tone: "bg-[var(--color-flag)]/15 text-[var(--color-flag)]",
  };
}

function CpuSpark({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const max = Math.max(100, ...values);
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * 100},${30 - (v / max) * 28}`)
    .join(" ");
  return (
    <svg viewBox="0 0 100 30" className="h-6 w-28" preserveAspectRatio="none" aria-hidden>
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

function Activity({ job, state }: { job: Job; state: State }) {
  const h = health(job);
  const run = job.preset ? latestRun(state, job.preset) : null;
  const index = run ? run.jobs.findIndex((j) => j.id === job.id) : -1;
  const cell = "flex flex-col gap-0.5";
  const label = "eyebrow !text-[0.6rem]";
  return (
    <div className="glass flex flex-wrap items-center gap-x-8 gap-y-3 p-4">
      <div className={cell}>
        <span className={label}>Health</span>
        <span className={`rounded-full px-2.5 py-0.5 text-sm font-semibold ${h.tone}`}>
          {h.label}
        </span>
      </div>
      {index >= 0 && run && (
        <div className={cell}>
          <span className={label}>Step</span>
          <span className="tabular font-mono text-sm">
            {index + 1} of {run.jobs.length}
            {job.steps > 1 && ` · part ${job.step}/${job.steps}`}
          </span>
        </div>
      )}
      <div className={cell}>
        <span className={label}>This step</span>
        <span className="tabular font-mono text-sm">{duration(job.step_elapsed)}</span>
      </div>
      <div className={cell}>
        <span className={label}>Last output</span>
        <span className="tabular font-mono text-sm">
          {job.idle < 2 ? "just now" : `${duration(job.idle)} ago`}
        </span>
      </div>
      <div className={cell}>
        <span className={label}>CPU</span>
        <span className="flex items-center gap-2 tabular font-mono text-sm">
          {job.cpu === null ? "…" : `${Math.round(job.cpu)}%`}
          <CpuSpark values={job.cpu_hist} />
        </span>
      </div>
      <div className={cell}>
        <span className={label}>Memory</span>
        <span className="tabular font-mono text-sm">
          {job.rss_mb === null
            ? "…"
            : job.rss_mb >= 1024
              ? `${(job.rss_mb / 1024).toFixed(1)} GB`
              : `${Math.round(job.rss_mb)} MB`}
        </span>
      </div>
      {job.progress !== null && (
        <div className={cell}>
          <span className={label}>Progress</span>
          <span className="tabular font-mono text-sm">{Math.round(job.progress * 100)}%</span>
        </div>
      )}
      <p className="basis-full text-xs text-[var(--color-carbon-300)]">
        {h.detail}. Command: <code className="text-[var(--color-carbon-100)]">{job.command}</code>
      </p>
    </div>
  );
}

function StatusBar({ state, act }: { state: State; act: Act }) {
  const last = state.system.samples[state.system.samples.length - 1];
  const running = state.jobs.filter((j) => j.status === "running").length;
  const queued = state.jobs.filter((j) => j.status === "queued").length;
  const pill = "rounded-full bg-white/5 px-3 py-1";
  return (
    <section className="flex flex-wrap items-center gap-2 font-mono text-xs text-[var(--color-carbon-300)]">
      <span className={`${pill} text-[var(--color-green)]`}>● connected</span>
      {last && (
        <>
          <span className={pill}>CPU {Math.round(last.cpu * 100)}%</span>
          <span className={pill}>
            RAM {last.mem_used.toFixed(1)}/{last.mem_total.toFixed(0)} GB
          </span>
          {last.speed_limit !== null && (
            <span className={`${pill} ${last.speed_limit < 100 ? "text-[var(--color-heat)]" : ""}`}>
              {last.speed_limit < 100 ? `throttled ${last.speed_limit}%` : "no throttling"}
            </span>
          )}
        </>
      )}
      <span className={pill}>
        {running} running · {queued} waiting
      </span>
      <span className="ml-auto" />
      {(running > 0 || queued > 0) && (
        <Btn onClick={() => act(() => post("/api/pause", { paused: !state.paused }))} tone="plain">
          {state.paused ? "▶ Resume all" : "⏸ Pause all"}
        </Btn>
      )}
    </section>
  );
}

function NowRunning({
  state,
  selected,
  onClear,
}: {
  state: State;
  selected: string | null;
  onClear: () => void;
}) {
  const titles = Object.fromEntries(state.tasks.map((t) => [t.id, t.title]));
  const running = state.jobs.find((j) => j.status === "running") ?? null;
  const job = state.jobs.find((j) => j.id === selected) ?? running;
  const report = state.reports[0];
  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="font-display text-xl font-bold">
          {job
            ? `${job.status === "running" ? "Now running" : "Output"}: ${titles[job.task] ?? job.task}`
            : "Nothing running"}
        </h2>
        {selected && running && selected !== running.id && (
          <button
            type="button"
            onClick={onClear}
            className="text-xs text-[var(--color-signal)] hover:underline"
          >
            back to the running step
          </button>
        )}
      </div>
      {job?.status === "running" && <Activity job={job} state={state} />}
      {job ? (
        <LogView key={job.id} job={job} onClose={onClear} />
      ) : (
        <p className="glass p-6 text-sm text-[var(--color-carbon-300)]">
          Pick a preset above and press Start. Its live output appears here.
        </p>
      )}
      {report && (
        <div className="glass p-4 text-sm">
          <span className="font-semibold text-[var(--color-green)]">Latest report:</span>{" "}
          <code>{report.path}</code>
          <p className="mt-1 text-xs text-[var(--color-carbon-300)]">
            Share it:{" "}
            <code className="text-[var(--color-carbon-100)]">
              git add docs/runs && git commit -m &quot;Run report&quot; && git push
            </code>
          </p>
        </div>
      )}
    </section>
  );
}

function Advanced({
  state,
  act,
  onSelect,
}: {
  state: State;
  act: Act;
  onSelect: (jobId: string) => void;
}) {
  const groups = useMemo(() => {
    const g = new Map<string, Task[]>();
    for (const t of state.tasks) g.set(t.group, [...(g.get(t.group) ?? []), t]);
    return [...g.entries()];
  }, [state.tasks]);

  const runOne = (t: Task) => {
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

  return (
    <details className="glass group overflow-hidden">
      <summary className="cursor-pointer select-none px-5 py-4 font-display font-bold">
        Advanced: smaller presets, single steps, history
      </summary>
      <div className="space-y-6 border-t border-white/5 p-5">
        <div className="grid gap-3 md:grid-cols-3">
          {state.presets
            .filter((p) => !p.main)
            .map((p) => (
              <PresetCard key={p.id} preset={p} state={state} act={act} onSelect={onSelect} />
            ))}
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-4">
            <h3 className="eyebrow">Single steps</h3>
            {groups.map(([group, tasks]) => (
              <div key={group}>
                <p className="mb-1 text-xs font-semibold text-[var(--color-carbon-300)]">
                  {group === "M6" ? "M6 checklist" : group}
                </p>
                <ul className="divide-y divide-white/5 rounded-lg bg-black/20">
                  {tasks.map((t) => {
                    const latest = state.latest[t.id];
                    return (
                      <li key={t.id} className="flex items-center gap-3 px-3 py-2">
                        <span className="w-9 shrink-0 font-mono text-[0.7rem] text-[var(--color-carbon-500)]">
                          {t.id.length <= 3 ? t.id : ""}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-sm" title={t.detail}>
                          {t.title}
                        </span>
                        {latest && (
                          <button type="button" onClick={() => onSelect(latest.id)}>
                            <Chip status={latest.status} />
                          </button>
                        )}
                        <Btn
                          onClick={() => runOne(t)}
                          disabled={!t.runnable}
                          tone={t.confirm ? "danger" : "plain"}
                          title={t.manual || t.steps.map((x) => x.join(" ")).join("\n")}
                        >
                          {t.confirm ? "Run once…" : "Run"}
                        </Btn>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </div>

          <div>
            <h3 className="eyebrow mb-2">History</h3>
            <ul className="divide-y divide-white/5 rounded-lg bg-black/20">
              {state.jobs.length === 0 && (
                <li className="px-3 py-4 text-sm text-[var(--color-carbon-300)]">
                  Nothing has run yet.
                </li>
              )}
              {state.jobs.slice(0, 80).map((j) => (
                <li key={j.id} className="flex items-center gap-3 px-3 py-2">
                  <button
                    type="button"
                    onClick={() => onSelect(j.id)}
                    className="flex min-w-0 flex-1 items-center gap-3 text-left"
                  >
                    <Chip status={j.status} />
                    <span className="min-w-0 flex-1 truncate text-sm">
                      {state.tasks.find((t) => t.id === j.task)?.title ?? j.task}
                    </span>
                    <span className="shrink-0 font-mono text-[0.7rem] text-[var(--color-carbon-500)]">
                      {j.preset ?? "single"} · {duration(j.elapsed)}
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
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </details>
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

  const act: Act = async (fn) => {
    setError(null);
    try {
      await fn();
      setSelected(null); // follow whatever runs next
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  if (online === false)
    return (
      <div className="glass space-y-3 p-8">
        <h2 className="font-display text-2xl font-bold">Not connected to this Mac</h2>
        <p className="text-sm text-[var(--color-carbon-300)]">
          The job server is not running. In Terminal, from the project folder, run the line below;
          this page connects by itself once it is up (the first start can take a few minutes).
        </p>
        <pre className="rounded-lg bg-black/40 p-4 font-mono text-sm">./f1p-dev.sh</pre>
      </div>
    );
  if (!state) return <p className="eyebrow">connecting…</p>;

  const main = state.presets.filter((p) => p.main);
  const [hero, ...rest] = main;

  return (
    <div className="space-y-6">
      <StatusBar state={state} act={act} />

      {state.paused && (
        <p className="rounded-lg bg-[var(--color-heat)]/15 px-4 py-3 text-sm text-[var(--color-heat)]">
          Paused: the current step finishes, then nothing new starts until you press Resume.
        </p>
      )}
      {error && (
        <p className="rounded-lg bg-[var(--color-flag)]/15 px-4 py-3 text-sm text-[var(--color-flag)]">
          {error}
        </p>
      )}

      {hero && <PresetCard preset={hero} state={state} act={act} onSelect={setSelected} big />}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {rest.map((p) => (
          <PresetCard key={p.id} preset={p} state={state} act={act} onSelect={setSelected} />
        ))}
      </div>

      <NowRunning state={state} selected={selected} onClear={() => setSelected(null)} />

      <Advanced state={state} act={act} onSelect={setSelected} />
    </div>
  );
}
