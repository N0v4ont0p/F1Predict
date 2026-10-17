"""Job execution for the dev panel: a queue, subprocesses, live logs and machine stats."""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from f1p.devpanel.tasks import BY_ID, KEEP_GOING, PIPELINES, Task, resolve

MAX_LINES = 20_000  # kept in memory per job; the full log is on disk
_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%")
_FRAC = re.compile(r"\b(\d+)\s?/\s?(\d+)\b")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass
class Job:
    id: str
    task: str
    pipeline: str | None = None  # the run's tag, e.g. "overnight-031500-002"
    preset: str | None = None  # the pipeline it belongs to, e.g. "overnight"
    keep_going: bool = False  # on a failure, skip only the pipeline's dependent tasks
    status: str = "queued"  # queued, running, ok, failed, cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    step: int = 0
    steps: int = 0
    command: str = ""
    returncode: int | None = None
    progress: float | None = None  # 0-1, parsed from the output when it shows one
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    first_line: int = 0  # index of lines[0] in the whole log
    proc: subprocess.Popen[bytes] | None = None
    cancel_requested: bool = False

    @property
    def line_count(self) -> int:
        return self.first_line + len(self.lines)

    def summary(self) -> dict[str, Any]:
        end = self.finished or time.time()
        return {
            "id": self.id,
            "task": self.task,
            "pipeline": self.pipeline,
            "preset": self.preset,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "elapsed": (end - self.started) if self.started else 0.0,
            "step": self.step,
            "steps": self.steps,
            "command": self.command,
            "returncode": self.returncode,
            "progress": self.progress,
            "lines": self.line_count,
            "last": self.lines[-1] if self.lines else "",
        }


def parse_progress(line: str) -> float | None:
    """A 0-1 progress value from a progress-bar or ``k/n`` line, if the line shows one."""
    if m := _PCT.search(line):
        v = float(m.group(1))
        if 0 <= v <= 100:
            return v / 100
    if m := _FRAC.search(line):
        k, n = int(m.group(1)), int(m.group(2))
        if 0 < n <= 100_000 and 0 <= k <= n:
            return k / n
    return None


class JobManager:
    """Runs queued jobs, at most ``parallel`` at a time, and records them under ``root``."""

    def __init__(self, root: Path, cwd: Path, parallel: int = 1) -> None:
        self.root = root
        self.cwd = cwd
        self.parallel = max(1, parallel)
        self.logs = root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.history_path = root / "jobs.jsonl"
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.cond = threading.Condition()
        self._seq = 0
        self._stopping = False
        self.paused = False  # when set, running jobs finish but no queued job starts
        self._load_history()
        self._worker = threading.Thread(target=self._loop, name="devpanel-jobs", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ history
    def _load_history(self) -> None:
        if not self.history_path.exists():
            return
        for raw in self.history_path.read_text().splitlines():
            with contextlib.suppress(ValueError, KeyError, TypeError):
                rec = json.loads(raw)
                job = Job(
                    id=rec["id"],
                    task=rec["task"],
                    pipeline=rec.get("pipeline"),
                    preset=rec.get("preset"),
                    status=rec["status"],
                    created=rec["created"],
                    started=rec.get("started"),
                    finished=rec.get("finished"),
                    step=rec.get("step", 0),
                    steps=rec.get("steps", 0),
                    command=rec.get("command", ""),
                    returncode=rec.get("returncode"),
                    progress=rec.get("progress"),
                )
                job.first_line = rec.get("lines", 0)
                self.jobs[job.id] = job
                self.order.append(job.id)

    def _record(self, job: Job) -> None:
        rec = job.summary()
        rec.pop("last", None)
        with self.history_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ API
    def submit(
        self,
        task_id: str,
        pipeline: str | None = None,
        keep_going: bool = False,
        preset: str | None = None,
    ) -> Job:
        task = BY_ID.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id!r}")
        if not task.runnable:
            raise ValueError(f"{task_id} cannot be run from the panel: {task.manual}")
        with self.cond:
            self._seq += 1
            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{self._seq:03d}-{task_id}"
            job = Job(
                id=job_id,
                task=task_id,
                pipeline=pipeline,
                preset=preset,
                keep_going=keep_going,
                steps=len(task.steps),
            )
            self.jobs[job_id] = job
            self.order.append(job_id)
            self.cond.notify_all()
        return job

    def submit_pipeline(self, name: str, tasks: tuple[str, ...] | None = None) -> list[Job]:
        """Queue a pipeline (or the given subset of its tasks, in order) as one run."""
        if name not in PIPELINES:
            raise KeyError(f"unknown pipeline {name!r}")
        with self.cond:
            self._seq += 1
            tag = f"{name}-{time.strftime('%H%M%S')}-{self._seq:03d}"
        keep = name in KEEP_GOING
        self.set_paused(False)  # pressing Start means "run it"
        return [
            self.submit(t, pipeline=tag, keep_going=keep, preset=name)
            for t in (tasks if tasks is not None else PIPELINES[name])
        ]

    def run_jobs(self, tag: str) -> list[Job]:
        with self.cond:
            return [self.jobs[i] for i in self.order if self.jobs[i].pipeline == tag]

    def stop_pipeline(self, tag: str) -> int:
        """Abort a run: cancel its queued jobs and terminate its running one."""
        jobs = self.run_jobs(tag)
        if not jobs:
            raise KeyError(f"unknown run {tag!r}")
        n = 0
        for j in jobs:
            if j.status in ("queued", "running"):
                self.cancel(j.id)
                n += 1
        with self.cond:
            if not any(j.status == "queued" for j in self.jobs.values()):
                self.paused = False  # nothing left to hold back
        return n

    def retry_pipeline(self, tag: str) -> list[Job]:
        """Run again the tasks of a finished run that did not succeed, as a new run."""
        jobs = self.run_jobs(tag)
        if not jobs or jobs[0].preset is None:
            raise KeyError(f"unknown run {tag!r}")
        if any(j.status in ("queued", "running") for j in jobs):
            raise ValueError("the run is still going: stop it first")
        todo = tuple(j.task for j in jobs if j.status != "ok" or BY_ID[j.task].always)
        if not todo:
            raise ValueError("every step of this run succeeded")
        return self.submit_pipeline(jobs[0].preset, todo)

    def restart_pipeline(self, tag: str) -> list[Job]:
        """Abort a run and start its pipeline again from the beginning."""
        jobs = self.run_jobs(tag)
        if not jobs or jobs[0].preset is None:
            raise KeyError(f"unknown run {tag!r}")
        self.stop_pipeline(tag)
        return self.submit_pipeline(jobs[0].preset)

    def set_paused(self, paused: bool) -> None:
        with self.cond:
            self.paused = paused
            self.cond.notify_all()

    def cancel(self, job_id: str) -> Job:
        with self.cond:
            job = self.jobs[job_id]
            if job.status == "queued":
                self._finish(job, "cancelled")
            elif job.status == "running":
                job.cancel_requested = True
                if job.proc is not None:
                    _terminate(job.proc)
            self.cond.notify_all()
        return job

    def clear_queue(self) -> int:
        with self.cond:
            queued = [j for j in self.jobs.values() if j.status == "queued"]
            for j in queued:
                self._finish(j, "cancelled")
            self.cond.notify_all()
        return len(queued)

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.cond:
            return [self.jobs[i].summary() for i in reversed(self.order[-limit:])]

    def last_status(self) -> dict[str, dict[str, Any]]:
        """The latest job per task (for the checklist's ticks)."""
        out: dict[str, dict[str, Any]] = {}
        with self.cond:
            for i in self.order:
                j = self.jobs[i]
                out[j.task] = {"id": j.id, "status": j.status, "finished": j.finished}
        return out

    def lines(self, job_id: str, start: int) -> tuple[list[str], int, str]:
        """Log lines from index ``start``, the next index, and the job's status."""
        with self.cond:
            job = self.jobs[job_id]
            if job.lines or job.status in ("queued", "running"):
                begin = max(start, job.first_line)
                chunk = list(job.lines)[begin - job.first_line :]
                return chunk, begin + len(chunk), job.status
        # a job from an earlier session: read its log file
        path = self.logs / f"{job_id}.log"
        all_lines = path.read_text(errors="replace").splitlines() if path.exists() else []
        return all_lines[start:], max(start, len(all_lines)), job.status

    def wait(self, job_id: str, index: int, timeout: float) -> None:
        """Block until the job has lines past ``index`` or changes state, or ``timeout``."""
        with self.cond:
            job = self.jobs[job_id]
            status = job.status
            self.cond.wait_for(
                lambda: job.line_count > index or job.status != status or self._stopping,
                timeout=timeout,
            )

    def shutdown(self) -> None:
        with self.cond:
            self._stopping = True
            for j in self.jobs.values():
                if j.status == "running" and j.proc is not None:
                    j.cancel_requested = True
                    _terminate(j.proc)
            self.cond.notify_all()

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self._stopping or self._next() is not None)
                if self._stopping:
                    return
                job = self._next()
                if job is None:
                    continue
                job.status = "running"
                job.started = time.time()
                self.cond.notify_all()
            threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def _next(self) -> Job | None:
        if self.paused:
            return None
        running = sum(1 for j in self.jobs.values() if j.status == "running")
        if running >= self.parallel:
            return None
        return next((self.jobs[i] for i in self.order if self.jobs[i].status == "queued"), None)

    def _append(self, job: Job, text: str) -> None:
        line = _ANSI.sub("", text.rstrip("\r\n"))
        # progress bars redraw with \r: keep the last frame only
        line = line.rsplit("\r", 1)[-1]
        with self.cond:
            if len(job.lines) == job.lines.maxlen:
                job.first_line += 1
            job.lines.append(line)
            if (p := parse_progress(line)) is not None:
                job.progress = p
            self.cond.notify_all()

    def _run(self, job: Job) -> None:
        task: Task = BY_ID[job.task]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "140", "TERM": "dumb"}
        status = "ok"
        with (self.logs / f"{job.id}.log").open("w") as log:
            for k, argv in enumerate(task.steps, start=1):
                if job.cancel_requested:
                    status = "cancelled"
                    break
                cmd = resolve(argv)
                with self.cond:
                    job.step, job.command, job.progress = k, " ".join(argv), None
                header = f"$ {' '.join(argv)}   [step {k}/{len(task.steps)}]"
                self._append(job, header)
                log.write(header + "\n")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=self.cwd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except OSError as exc:
                    self._append(job, f"could not start: {exc}")
                    log.write(f"could not start: {exc}\n")
                    status = "failed"
                    break
                with self.cond:
                    job.proc = proc
                assert proc.stdout is not None
                for raw in proc.stdout:
                    text = raw.decode(errors="replace")
                    self._append(job, text)
                    log.write(_ANSI.sub("", text))
                    log.flush()
                rc = proc.wait()
                with self.cond:
                    job.proc, job.returncode = None, rc
                if job.cancel_requested:
                    status = "cancelled"
                    break
                if rc != 0:
                    status = "failed"
                    self._append(job, f"exit code {rc}")
                    log.write(f"exit code {rc}\n")
                    break
        with self.cond:
            self._finish(job, status)
            if status != "ok" and job.pipeline:
                self._skip_after_failure(job)
            self.cond.notify_all()

    def _skip_after_failure(self, failed: Job) -> None:
        """Cancel the queued jobs of ``failed``'s pipeline that can no longer run usefully.

        A normal pipeline stops. A keep-going one (the overnight run) skips only the tasks
        that depend on the failed one, directly or through another skipped task, and
        ``always`` tasks (the report) run regardless.
        """
        queued = [
            j for j in self.jobs.values() if j.pipeline == failed.pipeline and j.status == "queued"
        ]
        if not failed.keep_going or BY_ID[failed.task].critical:
            for j in queued:
                if not BY_ID[j.task].always:
                    self._finish(j, "cancelled")
            return
        broken = {failed.task}
        changed = True
        while changed:
            changed = False
            for j in queued:
                task = BY_ID[j.task]
                if j.status == "queued" and not task.always and broken & set(task.after):
                    self._finish(j, "cancelled")
                    broken.add(j.task)
                    changed = True

    def _finish(self, job: Job, status: str) -> None:
        job.status = status
        job.finished = time.time()
        if status == "ok":
            job.progress = 1.0
        self._record(job)


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the job's process group, then SIGKILL it if it is still alive after 10 s."""

    def kill() -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)

    threading.Thread(target=kill, daemon=True).start()


# ---------------------------------------------------------------------- machine stats
def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def memory() -> tuple[float, float]:
    """(used GB, total GB) of physical memory."""
    if sys.platform == "darwin":
        total = int(_run_cmd(["sysctl", "-n", "hw.memsize"]).strip() or 0)
        out = _run_cmd(["vm_stat"])
        page = int(m.group(1)) if (m := re.search(r"page size of (\d+)", out)) else 16384

        def pages(name: str) -> int:
            m = re.search(rf"{name}:\s+(\d+)", out)
            return int(m.group(1)) if m else 0

        used = (
            pages("Pages active")
            + pages("Pages wired down")
            + pages("Pages occupied by compressor")
        ) * page
        return used / 1024**3, total / 1024**3
    info: dict[str, int] = {}
    with contextlib.suppress(OSError):
        for row in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = row.partition(":")
            info[key] = int(rest.split()[0]) * 1024
    total = info.get("MemTotal", 0)
    return (total - info.get("MemAvailable", total)) / 1024**3, total / 1024**3


def thermal() -> int | None:
    """macOS CPU speed limit in percent (100 = not throttled); None elsewhere."""
    if sys.platform != "darwin":
        return None
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", _run_cmd(["pmset", "-g", "therm"]))
    return int(m.group(1)) if m else 100


class SystemMonitor:
    """Samples load, memory and throttling every ``every`` seconds; keeps the last 10 min."""

    def __init__(self, every: float = 2.0) -> None:
        self.every = every
        self.samples: deque[dict[str, Any]] = deque(maxlen=int(600 / every))
        self.cpus = os.cpu_count() or 1
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, name="devpanel-sys", daemon=True).start()

    def _loop(self) -> None:
        while True:
            load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
            used, total = memory()
            sample = {
                "t": time.time(),
                "load": load,
                "cpu": min(1.0, load / self.cpus),
                "mem_used": used,
                "mem_total": total,
                "speed_limit": thermal(),
            }
            with self._lock:
                self.samples.append(sample)
            time.sleep(self.every)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"cpus": self.cpus, "samples": list(self.samples)}
