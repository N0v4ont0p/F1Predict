from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from f1p.devpanel import jobs as J
from f1p.devpanel import server as S
from f1p.devpanel import tasks as T


def _fake_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "ok": T.Task("ok", "Test", "ok", "", (("print", "a"), ("print", "b"))),
        "bad": T.Task("bad", "Test", "bad", "", (("fail",), ("print", "never"))),
        "slow": T.Task("slow", "Test", "slow", "", (("sleep",),)),
        "guarded": T.Task("guarded", "Test", "g", "", (("print", "x"),), confirm="YES"),
        "manual": T.Task("manual", "Test", "m", "", manual="by hand"),
    }
    scripts = {
        "print": "import sys; print(sys.argv[1]); print('3/4 done')",
        "fail": "import sys; print('boom'); sys.exit(3)",
        "sleep": "import time; time.sleep(30)",
    }

    def resolve(argv: tuple[str, ...]) -> list[str]:
        return [sys.executable, "-c", scripts[argv[0]], *argv[1:]]

    monkeypatch.setattr(J, "BY_ID", fake)
    monkeypatch.setattr(S, "BY_ID", fake)
    monkeypatch.setattr(J, "resolve", resolve)
    monkeypatch.setattr(J, "PIPELINES", {"p": ("bad", "ok")})


def _wait(manager: J.JobManager, job_id: str, timeout: float = 20) -> str:
    end = time.time() + timeout
    while time.time() < end:
        status = manager.jobs[job_id].status
        if status not in ("queued", "running"):
            return status
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_catalogue_is_consistent() -> None:
    assert len(T.BY_ID) == len(T.TASKS)
    for ids in T.PIPELINES.values():
        assert all(T.BY_ID[i].runnable for i in ids)
    # the holdout is never part of a pipeline and always needs a typed confirmation
    assert all("B12" not in ids for ids in T.PIPELINES.values())
    assert T.BY_ID["B12"].confirm


def test_resolve_runs_f1p_with_this_interpreter() -> None:
    assert T.resolve(("f1p", "version")) == [sys.executable, "-m", "f1p.cli.main", "version"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [("  45%|####", 0.45), ("race 3/12", 0.25), ("no progress here", None), ("150%", None)],
)
def test_parse_progress(line: str, expected: float | None) -> None:
    assert J.parse_progress(line) == expected


def test_jobs_run_log_and_persist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_tasks(monkeypatch)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    job = m.submit("ok")
    assert _wait(m, job.id) == "ok"
    lines, nxt, status = m.lines(job.id, 0)
    assert "a" in lines
    assert "b" in lines
    assert status == "ok"
    assert nxt == len(lines)
    assert m.jobs[job.id].progress == 1.0
    assert (tmp_path / "logs" / f"{job.id}.log").read_text().count("3/4 done") == 2
    m.shutdown()

    # a new manager reloads the history and reads old logs from disk
    m2 = J.JobManager(tmp_path, cwd=tmp_path)
    assert m2.jobs[job.id].status == "ok"
    assert "a" in m2.lines(job.id, 0)[0]
    assert m2.last_status()["ok"]["status"] == "ok"
    m2.shutdown()


def test_failure_stops_the_task_and_its_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_tasks(monkeypatch)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    bad, ok = m.submit_pipeline("p")
    assert _wait(m, bad.id) == "failed"
    assert _wait(m, ok.id) == "cancelled"
    lines = m.lines(bad.id, 0)[0]
    assert "boom" in lines
    assert "never" not in lines
    assert m.jobs[bad.id].returncode == 3
    m.shutdown()


def test_cancel_running_and_refuse_manual(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_tasks(monkeypatch)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    with pytest.raises(ValueError, match="by hand"):
        m.submit("manual")
    with pytest.raises(KeyError):
        m.submit("rm -rf /")
    job = m.submit("slow")
    while m.jobs[job.id].proc is None:
        time.sleep(0.05)
    m.cancel(job.id)
    assert _wait(m, job.id) == "cancelled"
    m.shutdown()


def _request(
    port: int, path: str, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_guards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_tasks(monkeypatch)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    server = S.serve(m, 0)
    port = server.server_address[1]
    dev = {"X-F1P-Dev": "1", "Content-Type": "application/json"}
    try:
        assert _request(port, "/api/health")[0] == 200
        assert _request(port, "/api/state", headers={"Origin": "https://evil.example"})[0] == 403
        # a POST without the custom header (what a foreign page could send) is refused
        assert _request(port, "/api/run", {"task": "ok"})[0] == 403
        assert _request(port, "/api/run", {"task": "guarded"}, dev)[0] == 409
        code, body = _request(port, "/api/run", {"task": "guarded", "confirm": "YES"}, dev)
        assert code == 200
        assert _wait(m, body["job"]["id"]) == "ok"
        code, state = _request(port, "/api/state", headers={"Origin": "http://localhost:3000"})
        assert code == 200
        assert state["jobs"][0]["status"] == "ok"
        code, log = _request(port, f"/api/jobs/{body['job']['id']}/log?from=0")
        assert code == 200
        assert "x" in log["lines"]
    finally:
        server.shutdown()
        m.shutdown()
