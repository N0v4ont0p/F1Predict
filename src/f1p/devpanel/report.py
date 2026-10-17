"""The run report: one Markdown file (plus the small result files) summarising a dev-panel run.

Written to ``docs/runs/`` so it can be committed and pushed from the machine that ran it; it
carries the numbers the gate decisions need (each job's status and the end of its output) and
copies the small JSON results the run produced. Nothing is adopted by writing it.
"""

from __future__ import annotations

import contextlib
import json
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f1p.devpanel.tasks import BY_ID, TASKS

TAIL_LINES = 60
MAX_LINE = 220
MAX_COPY_BYTES = 256_000
# result files worth copying; logs, caches and the lake are not
RESULT_DIRS = ("bench", "eval", "sim_v2", "tuning", "stack", "lapseq", "ledger", "posteriors")


@dataclass(frozen=True)
class Run:
    pipeline: str | None
    jobs: list[dict[str, Any]]  # latest record per task, in catalogue order
    started: float
    finished: float


def load_jobs(dev_dir: Path) -> list[dict[str, Any]]:
    path = dev_dir / "jobs.jsonl"
    if not path.exists():
        return []
    out = []
    for raw in path.read_text().splitlines():
        with contextlib.suppress(ValueError):
            out.append(json.loads(raw))
    return out


def select_run(records: list[dict[str, Any]], pipeline: str | None = None) -> Run | None:
    """The jobs of ``pipeline`` (default: the latest pipeline run), else the latest per task."""
    done = [r for r in records if r.get("task") != "report" and r.get("finished")]
    if not done:
        return None
    if pipeline is None:
        pipeline = max(done, key=lambda r: r["finished"]).get("pipeline")
    chosen = [r for r in done if r.get("pipeline") == pipeline] if pipeline else done
    latest: dict[str, dict[str, Any]] = {}
    for r in sorted(chosen, key=lambda r: r["finished"]):
        latest[r["task"]] = r
    order = {t.id: i for i, t in enumerate(TASKS)}
    jobs = sorted(latest.values(), key=lambda r: order.get(r["task"], len(order)))
    started = min((r.get("started") or r["finished"]) for r in jobs)
    return Run(pipeline, jobs, started, max(r["finished"] for r in jobs))


def _fmt_time(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip()


def _machine() -> str:
    try:
        from f1p.silicon.device import detect

        d = detect()
        cores = ", ".join(f"{c.cores}x {c.name}" for c in d.clusters)
        return f"{d.chip} · {cores} · {d.memory_gb:g} GB · GPU cores {d.gpu_cores or '-'}"
    except Exception:  # the report must still be written
        return f"{platform.platform()} / {platform.machine()}"


def _tail(path: Path, n: int = TAIL_LINES) -> list[str]:
    if not path.exists():
        return ["(no log)"]
    lines = path.read_text(errors="replace").splitlines()
    return [ln if len(ln) <= MAX_LINE else ln[: MAX_LINE - 1] + "…" for ln in lines[-n:]]


def _results_since(artifacts: Path, since: float) -> list[Path]:
    found: list[Path] = []
    for name in RESULT_DIRS:
        base = artifacts / name
        if not base.exists():
            continue
        found += [
            p
            for p in base.rglob("*")
            if p.is_file()
            and p.suffix in (".json", ".jsonl", ".csv", ".md", ".txt")
            and p.stat().st_mtime >= since
        ]
    return sorted(found)


def write(
    dev_dir: Path, artifacts: Path, out_dir: Path, repo: Path, pipeline: str | None = None
) -> Path | None:
    """Write the report for a run; returns its path (None when nothing has run yet)."""
    run = select_run(load_jobs(dev_dir), pipeline)
    if run is None:
        return None
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(run.finished))
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"run-{stamp}.md"
    files_dir = out_dir / f"run-{stamp}"

    counts: dict[str, int] = {}
    for r in run.jobs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    commit = _git(repo, "rev-parse", "--short", "HEAD") or "unknown"
    dirty = " (with local changes)" if _git(repo, "status", "--porcelain", "--", "src") else ""

    lines = [
        f"# Run report — {_fmt_time(run.finished)}",
        "",
        f"- **Run:** {run.pipeline or 'individual tasks'}, "
        f"{_fmt_time(run.started)} → {_fmt_time(run.finished)} "
        f"({_duration(run.finished - run.started)})",
        "- **Result:** " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
        f"- **Code:** `{commit}`{dirty}",
        f"- **Machine:** {_machine()}",
        "",
        "Nothing is adopted by this run. Push this file; each result is then checked against",
        "its pre-registered gate (`docs/plan/M6_FOLLOWUP_PLAN.md`) before anything changes.",
        "",
        "| Task | What | Status | Time | Finished |",
        "|---|---|---|---|---|",
    ]
    for r in run.jobs:
        task = BY_ID.get(r["task"])
        title = task.title if task else r["task"]
        elapsed = _duration(r.get("elapsed") or 0) if r.get("started") else "-"
        lines.append(
            f"| {r['task']} | {title} | {r['status']} | {elapsed} | {_fmt_time(r['finished'])} |"
        )

    copied: list[str] = []
    for p in _results_since(artifacts, run.started):
        if p.stat().st_size > MAX_COPY_BYTES:
            continue
        dest = files_dir / p.relative_to(artifacts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        copied.append(str(dest.relative_to(out_dir)))
    if copied:
        lines += ["", "## Result files", "", *(f"- `{c}`" for c in copied)]

    for r in run.jobs:
        task = BY_ID.get(r["task"])
        lines += ["", f"## {r['task']} — {task.title if task else ''} ({r['status']})", ""]
        if task and task.steps:
            lines += ["```", *(" ".join(s) for s in task.steps), "```", ""]
        lines += [
            f"Last {TAIL_LINES} lines of the output:",
            "",
            "```text",
            *_tail(dev_dir / "logs" / f"{r['id']}.log"),
            "```",
        ]
    md.write_text("\n".join(lines) + "\n")
    return md
