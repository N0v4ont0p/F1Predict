from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest
from typer.testing import CliRunner

from f1p.cli.main import app
from f1p.core.config import MemoryBudget
from f1p.eval.forecast import position_matrix, sample_orders
from f1p.silicon import kernels
from f1p.silicon.budget import BudgetError, MemoryLedger, batch_size
from f1p.silicon.device import detect
from f1p.silicon.qos import Lane, QoS, lane_pool, lane_size, set_thread_qos


def test_detect_describes_this_machine() -> None:
    d = detect()
    assert d.logical_cpus >= 1
    assert d.strong_cores >= 1
    assert d.memory_gb > 0
    assert d.clusters
    assert "strong_cores" in d.to_dict()
    if sys.platform != "darwin":
        assert not d.apple_silicon
        assert not d.neural_engine


@pytest.mark.apple
@pytest.mark.skipif(sys.platform != "darwin", reason="Apple silicon only")
def test_apple_clusters_and_qos() -> None:  # pragma: no cover - runs on macOS CI / the M6
    d = detect()
    assert d.apple_silicon
    assert d.clusters
    if "Virtual" not in d.chip:  # CI VMs expose one "Standard" cluster
        assert d.efficiency_cores > 0
    assert set_thread_qos(QoS.USER_INITIATED)


def test_qos_is_a_noop_off_macos() -> None:
    if sys.platform != "darwin":
        assert set_thread_qos(QoS.UTILITY) is False
    assert Lane.BACKGROUND.qos is QoS.UTILITY
    assert lane_size(Lane.COMPUTE) >= 1
    with lane_pool(Lane.BACKGROUND, max_workers=2) as pool:
        assert list(pool.map(lambda x: x * 2, range(4))) == [0, 2, 4, 6]


def test_ledger_admits_within_budget_and_blocks_beyond() -> None:
    ledger = MemoryLedger(MemoryBudget(total_gb=24, reserved_gb=8, working_gb=16))
    with ledger.reserve(10, "mcmc"):
        assert ledger.available_gb == pytest.approx(6)
        with pytest.raises(TimeoutError), ledger.reserve(8, "mlx", timeout=0.05):
            pass
        released = threading.Event()

        def later() -> None:
            with ledger.reserve(8, "mlx", timeout=5):
                released.set()

        t = threading.Thread(target=later)
        t.start()
        time.sleep(0.05)
        assert not released.is_set()
    t.join(timeout=5)
    assert released.is_set()
    assert ledger.reserved_gb == 0
    with pytest.raises(BudgetError), ledger.reserve(17, "too big"):
        pass


def test_batch_size() -> None:
    assert batch_size(1024**2, 1.0) == 1024
    assert batch_size(10**12, 0.001) == 1


def test_sampler_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F1P_SAMPLER", "numpy")
    assert kernels.preferred_sampler() == "numpy"
    monkeypatch.setenv("F1P_SAMPLER", "mlx")
    assert kernels.preferred_sampler() == ("mlx" if kernels.mlx_available() else "numpy")


@pytest.mark.skipif(not kernels.mlx_available(), reason="MLX not installed")
def test_mlx_sampler_matches_numpy_distribution() -> None:
    rng = np.random.default_rng(0)
    ls = rng.normal(size=12) * 1.5
    pf = np.full(12, 0.85)
    sc = np.exp(-0.15 * np.arange(26))
    o1, f1 = kernels.staged_orders_mlx(ls, 60_000, 3, p_finish=pf, stage_scales=sc)
    o2, f2 = sample_orders(ls, 60_000, rng, p_finish=pf, stage_scales=sc)
    assert o1.shape == o2.shape == (60_000, 12)
    assert np.all(np.sort(o1, axis=1) == np.arange(12))  # every row is a permutation
    np.testing.assert_allclose(f1.mean(), f2.mean(), atol=0.01)
    np.testing.assert_allclose(position_matrix(o1, 12), position_matrix(o2, 12), atol=0.012)
    # retirements always sit behind the classified cars
    rows = np.arange(o1.shape[0])[:, None]
    fin_sorted = f1[rows, o1]
    assert np.all(np.diff(fin_sorted.astype(int), axis=1) <= 0)


def test_cli_info_and_quick_bench() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["silicon", "info"])
    assert res.exit_code == 0, res.stdout
    assert "race sampler" in res.stdout
    res = runner.invoke(app, ["silicon", "bench", "--quick", "--no-save"])
    assert res.exit_code == 0, res.stdout
    assert "race_sampler" in res.stdout
