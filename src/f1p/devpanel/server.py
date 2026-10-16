"""The dev panel's HTTP API on 127.0.0.1 (standard library only).

Routes (JSON unless noted):

- ``GET  /api/health``                   liveness and version
- ``GET  /api/state``                    tasks, pipelines, jobs, latest status per task, machine
- ``GET  /api/jobs/<id>/log?from=N``     log lines from N
- ``GET  /api/jobs/<id>/stream?from=N``  the same, live, as server-sent events
- ``POST /api/run``                      ``{"task": id, "confirm": phrase?}``
- ``POST /api/pipeline``                 ``{"name": checks|hardware|m6}``
- ``POST /api/jobs/<id>/cancel``
- ``POST /api/queue/clear``

Requests are accepted only from localhost pages (Origin and Host checked, and a custom header
on every POST so a foreign page cannot send one without a CORS preflight that fails).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from f1p import __version__
from f1p.devpanel.jobs import JobManager, SystemMonitor
from f1p.devpanel.tasks import BY_ID, PIPELINES, TASKS

_LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
_LOCAL_HOST = re.compile(r"^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
_JOB = re.compile(r"^/api/jobs/([\w.-]+)/(log|stream|cancel)$")
HEADER = "X-F1P-Dev"
STREAM_SECONDS = 600  # the browser reconnects after this


class Panel:
    def __init__(self, manager: JobManager, monitor: SystemMonitor) -> None:
        self.manager = manager
        self.monitor = monitor

    def state(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "now": time.time(),
            "parallel": self.manager.parallel,
            "tasks": [{**asdict(t), "runnable": t.runnable} for t in TASKS],
            "pipelines": PIPELINES,
            "jobs": self.manager.list(),
            "latest": self.manager.last_status(),
            "system": self.monitor.snapshot(),
        }


def make_handler(panel: Panel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "f1p-devpanel"

        def log_message(self, format: str, *args: Any) -> None:  # quiet
            return

        # ------------------------------------------------------------ helpers
        def _origin_ok(self) -> bool:
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            return bool(_LOCAL_HOST.match(host)) and (
                origin is None or bool(_LOCAL_ORIGIN.match(origin))
            )

        def _cors(self) -> None:
            origin = self.headers.get("Origin")
            if origin and _LOCAL_ORIGIN.match(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _json(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            data = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
            return data if isinstance(data, dict) else {}

        # ------------------------------------------------------------ verbs
        def do_OPTIONS(self) -> None:
            if not self._origin_ok():
                self.send_response(403)
                self.end_headers()
                return
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", f"Content-Type, {HEADER}")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:
            if not self._origin_ok():
                self._json(403, {"error": "localhost only"})
                return
            url = urlparse(self.path)
            query = parse_qs(url.query)
            start = int((query.get("from") or ["0"])[0] or 0)
            if url.path == "/api/health":
                self._json(200, {"ok": True, "version": __version__})
            elif url.path == "/api/state":
                self._json(200, panel.state())
            elif (m := _JOB.match(url.path)) and m.group(2) in ("log", "stream"):
                job_id = m.group(1)
                if job_id not in panel.manager.jobs:
                    self._json(404, {"error": "no such job"})
                elif m.group(2) == "log":
                    lines, nxt, status = panel.manager.lines(job_id, start)
                    self._json(200, {"lines": lines, "next": nxt, "status": status})
                else:
                    self._stream(job_id, start)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._origin_ok() or self.headers.get(HEADER) != "1":
                self._json(403, {"error": f"localhost pages only, with the {HEADER} header"})
                return
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/run":
                    task_id = str(body.get("task", ""))
                    task = BY_ID.get(task_id)
                    if task and task.confirm and body.get("confirm") != task.confirm:
                        self._json(409, {"error": f'type "{task.confirm}" to start {task_id}'})
                        return
                    job = panel.manager.submit(task_id)
                    self._json(200, {"job": job.summary()})
                elif path == "/api/pipeline":
                    jobs = panel.manager.submit_pipeline(str(body.get("name", "")))
                    self._json(200, {"jobs": [j.summary() for j in jobs]})
                elif path == "/api/queue/clear":
                    self._json(200, {"cancelled": panel.manager.clear_queue()})
                elif (m := _JOB.match(path)) and m.group(2) == "cancel":
                    self._json(200, {"job": panel.manager.cancel(m.group(1)).summary()})
                else:
                    self._json(404, {"error": "not found"})
            except KeyError as exc:
                self._json(404, {"error": str(exc.args[0]) if exc.args else "not found"})
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def _stream(self, job_id: str, start: int) -> None:
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            index = start
            deadline = time.time() + STREAM_SECONDS
            try:
                while time.time() < deadline:
                    lines, index, status = panel.manager.lines(job_id, index)
                    payload = json.dumps({"lines": lines, "next": index, "status": status})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    if status not in ("queued", "running"):
                        break
                    panel.manager.wait(job_id, index, timeout=15)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def serve(manager: JobManager, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Start the server in a background thread and return it (``shutdown()`` to stop)."""
    panel = Panel(manager, SystemMonitor())
    server = ThreadingHTTPServer((host, port), make_handler(panel))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="devpanel-http", daemon=True).start()
    return server
