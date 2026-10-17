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
from f1p.devpanel import report as R
from f1p.devpanel import server as S
from f1p.devpanel import tasks as T


def _fake_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "ok": T.Task("ok", "Test", "ok", "", (("print", "a"), ("print", "b"))),
        "bad": T.Task("bad", "Test", "bad", "", (("fail",), ("print", "never"))),
        "slow": T.Task("slow", "Test", "slow", "", (("sleep",),)),
        "guarded": T.Task("guarded", "Test", "g", "", (("print", "x"),), confirm="YES"),
        "manual": T.Task("manual", "Test", "m", "", manual="by hand"),
        "dep": T.Task("dep", "Test", "dep", "", (("print", "d"),), after=("bad",)),
        "dep2": T.Task("dep2", "Test", "dep2", "", (("print", "e"),), after=("dep",)),
        "end": T.Task("end", "Test", "end", "", (("print", "z"),), always=True),
        "crit": T.Task("crit", "Test", "crit", "", (("fail",),), critical=True),
        "nap": T.Task("nap", "Test", "nap", "", (("nap",),)),
    }
    scripts = {
        "print": "import sys; print(sys.argv[1]); print('3/4 done')",
        "fail": "import sys; print('boom'); sys.exit(3)",
        "sleep": "import time; time.sleep(30)",
        "nap": "import time; print('start', flush=True); time.sleep(2)",
    }

    def resolve(argv: tuple[str, ...]) -> list[str]:
        return [sys.executable, "-c", scripts[argv[0]], *argv[1:]]

    monkeypatch.setattr(J, "BY_ID", fake)
    monkeypatch.setattr(S, "BY_ID", fake)
    monkeypatch.setattr(J, "resolve", resolve)
    monkeypatch.setattr(
        J,
        "PIPELINES",
        {
            "p": ("bad", "ok", "end"),
            "night": ("bad", "dep", "ok", "dep2", "end"),
            "setup": ("crit", "ok", "end"),
            "slowrun": ("slow", "ok", "end"),
        },
    )
    monkeypatch.setattr(J, "KEEP_GOING", frozenset({"night", "setup"}))


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
    [
        ("  45%|####", 0.45),
        ("race 3/12", 0.25),
        ("weather 6/24: 2026-R06", 0.25),
        ("[10/17/26 09:57:18] INFO fetched x", None),
        ("no progress here", None),
        ("150%", None),
    ],
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
    bad, ok, end = m.submit_pipeline("p")
    assert _wait(m, bad.id) == "failed"
    assert _wait(m, ok.id) == "cancelled"
    assert _wait(m, end.id) == "ok"  # an `always` task still runs
    lines = m.lines(bad.id, 0)[0]
    assert "boom" in lines
    assert "never" not in lines
    assert m.jobs[bad.id].returncode == 3
    m.shutdown()


def test_keep_going_pipeline_skips_only_dependents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_tasks(monkeypatch)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    bad, dep, ok, dep2, end = m.submit_pipeline("night")
    assert _wait(m, bad.id) == "failed"
    assert _wait(m, dep.id) == "cancelled"
    assert _wait(m, ok.id) == "ok"
    assert _wait(m, dep2.id) == "cancelled"  # depends on bad through dep
    assert _wait(m, end.id) == "ok"
    m.shutdown()


def test_critical_failure_stops_a_keep_going_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_tasks(monkeypatch)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    crit, ok, end = m.submit_pipeline("setup")
    assert _wait(m, crit.id) == "failed"
    assert _wait(m, ok.id) == "cancelled"
    assert _wait(m, end.id) == "ok"
    m.shutdown()


def test_retry_reruns_only_what_did_not_succeed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_tasks(monkeypatch)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    first = m.submit_pipeline("night")
    for j in first:
        _wait(m, j.id)
    tag = first[0].pipeline
    assert tag is not None
    again = m.retry_pipeline(tag)
    # bad, dep, dep2 did not succeed; end always runs; ok is not repeated
    assert [j.task for j in again] == ["bad", "dep", "dep2", "end"]
    assert {j.preset for j in again} == {"night"}
    assert again[0].pipeline != tag
    for j in again:
        _wait(m, j.id)
    m.shutdown()


def test_stop_restart_and_pause(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_tasks(monkeypatch)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    slow, ok, end = m.submit_pipeline("slowrun")
    while m.jobs[slow.id].proc is None:
        time.sleep(0.05)
    assert slow.pipeline is not None
    with pytest.raises(ValueError, match="still going"):
        m.retry_pipeline(slow.pipeline)
    fresh = m.restart_pipeline(slow.pipeline)
    for j in (slow, ok, end):
        assert _wait(m, j.id) == "cancelled"
    assert [j.task for j in fresh] == ["slow", "ok", "end"]
    assert fresh[0].pipeline is not None
    assert m.stop_pipeline(fresh[0].pipeline) == 3
    for j in fresh:
        assert _wait(m, j.id) == "cancelled"

    # paused: nothing new starts until released
    m.set_paused(True)
    held = m.submit("ok")
    time.sleep(0.5)
    assert m.jobs[held.id].status == "queued"
    m.set_paused(False)
    assert _wait(m, held.id) == "ok"
    m.shutdown()


def test_quiet_step_gets_heartbeats_and_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_tasks(monkeypatch)
    monkeypatch.setattr(J, "HEARTBEAT_S", 0.5)
    monkeypatch.setattr(J, "SAMPLE_S", 0.2)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    job = m.submit("nap")
    while m.jobs[job.id].status != "running":
        time.sleep(0.05)
    time.sleep(1.2)
    summary = m.jobs[job.id].summary()
    assert summary["cpu"] is not None
    assert summary["rss_mb"] is not None
    assert summary["idle"] > 0.3
    assert _wait(m, job.id) == "ok"
    lines = m.lines(job.id, 0)[0]
    assert any(line.startswith("⋯ still working") for line in lines)
    m.shutdown()


def test_presets_cover_real_pipelines() -> None:
    assert {p.id for p in T.PRESETS} <= set(T.PIPELINES)
    assert T.PRESETS[0].id == "everything"
    assert "B12" not in T.PIPELINES["everything"]
    assert T.PIPELINES["everything"][-1] == "report"
    assert T.BY_ID["install"].critical


def test_report_summarises_the_latest_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_tasks(monkeypatch)
    dev, artifacts, out = tmp_path / "dev", tmp_path / "artifacts", tmp_path / "runs"
    m = J.JobManager(dev, cwd=tmp_path)
    jobs = m.submit_pipeline("night")
    for j in jobs:
        _wait(m, j.id)
    m.shutdown()
    (artifacts / "bench").mkdir(parents=True)
    (artifacts / "bench" / "result.json").write_text('{"x": 1}')

    md = R.write(dev, artifacts, out, tmp_path)
    assert md is not None
    text = md.read_text()
    assert "1 failed" in text
    assert "2 ok" in text
    assert "boom" in text  # the failed task's output is in the report
    assert "bench/result.json" in text
    assert (out / md.stem / "bench" / "result.json").exists()
    assert R.write(tmp_path / "empty", artifacts, out, tmp_path) is None


def test_overnight_never_scores_the_holdout() -> None:
    night = T.PIPELINES["overnight"]
    assert "B12" not in night
    assert night[-1] == "report"
    assert T.BY_ID["report"].always


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


def test_era5_key_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_tasks(monkeypatch)
    rc = tmp_path / "home" / ".cdsapirc"
    monkeypatch.setenv("CDSAPI_RC", str(rc))
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    m = J.JobManager(tmp_path, cwd=tmp_path)
    server = S.serve(m, 0)
    port = server.server_address[1]
    dev = {"X-F1P-Dev": "1", "Content-Type": "application/json"}
    key = "0123abcd-4567-89ef-0123-456789abcdef"
    try:
        assert _request(port, "/api/state")[1]["era5"]["key_saved"] is False
        assert _request(port, "/api/era5/key", {"key": key})[0] == 403
        code, body = _request(port, "/api/era5/key", {"key": "nope"}, dev)
        assert code == 400
        assert not rc.exists()
        code, body = _request(port, "/api/era5/key", {"key": f"  key: {key}\n"}, dev)
        assert code == 200
        assert body["era5"]["key_saved"] is True
        assert rc.read_text() == f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n"
        assert rc.stat().st_mode & 0o777 == 0o600
        assert key not in json.dumps(_request(port, "/api/state")[1])
        assert key not in json.dumps(body)
    finally:
        server.shutdown()
        m.shutdown()
