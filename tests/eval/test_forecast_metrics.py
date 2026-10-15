from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from f1p.eval.bootstrap import paired_bootstrap
from f1p.eval.forecast import (
    Forecast,
    pl_log_likelihood,
    pl_sample_orders,
    pl_win_probabilities,
    position_matrix,
)
from f1p.eval.metrics import (
    Outcome,
    expected_calibration_error,
    reliability_table,
    score,
    spearman,
)

strengths = st.lists(st.floats(-3, 3, allow_nan=False), min_size=2, max_size=8)


@given(strengths)
def test_pl_likelihoods_of_all_orders_sum_to_one(values: list[float]) -> None:
    import itertools

    s = np.array(values[:5])
    total = sum(
        math.exp(pl_log_likelihood(s, np.array(p))) for p in itertools.permutations(range(len(s)))
    )
    assert total == pytest.approx(1.0, rel=1e-9)


@given(strengths)
def test_marginal_win_probability_matches_first_choice(values: list[float]) -> None:
    s = np.array(values)
    p = pl_win_probabilities(s)
    assert p.sum() == pytest.approx(1.0)
    for i in range(len(s)):
        assert math.exp(pl_log_likelihood(s, np.array([i]))) == pytest.approx(1.0)
        assert p[i] == pytest.approx(math.exp(s[i]) / np.exp(s).sum())


@settings(max_examples=20, deadline=None)
@given(strengths)
def test_sampled_positions_match_exact_win_probability(values: list[float]) -> None:
    s = np.array(values)
    orders = pl_sample_orders(s, 40_000, np.random.default_rng(1))
    pm = position_matrix(orders, len(s))
    assert np.allclose(pm.sum(axis=0), 1.0)
    assert np.allclose(pm.sum(axis=1), 1.0)
    assert np.allclose(pm[:, 0], pl_win_probabilities(s), atol=0.015)


def test_forecast_validates_shapes() -> None:
    with pytest.raises(ValueError, match="must be"):
        Forecast("e", "C4", "m", ("a", "b"), np.eye(3))
    with pytest.raises(ValueError, match="sum to 1"):
        Forecast("e", "C4", "m", ("a", "b"), np.array([[0.5, 0.0], [0.5, 1.0]]))


def _fc(log_strength: list[float]) -> Forecast:
    drivers = tuple("abcdefghijkl"[: len(log_strength)])
    return Forecast.from_log_strength(
        "e", "C4", "m", drivers, np.array(log_strength), n_samples=5000
    )


def test_score_rewards_the_right_favourite() -> None:
    good = score(_fc([3.0, 1.0, 0.0, -1.0]), Outcome("e", ("a", "b", "c", "d")))
    bad = score(_fc([-1.0, 0.0, 1.0, 3.0]), Outcome("e", ("a", "b", "c", "d")))
    assert good["winner_logloss"] < bad["winner_logloss"]
    assert good["winner_hit"] == 1.0
    assert bad["winner_hit"] == 0.0
    assert good["spearman"] == pytest.approx(1.0)
    assert bad["spearman"] == pytest.approx(-1.0)
    assert good["finisher_pl_loglik"] > bad["finisher_pl_loglik"]


def test_flat_forecast_gets_fractional_hit_and_no_rank_signal() -> None:
    m = score(_fc([0.0] * 4), Outcome("e", ("c", "a", "b", "d")))
    assert m["winner_hit"] == pytest.approx(0.25)
    assert m["spearman"] == 0.0
    assert m["winner_logloss"] == pytest.approx(math.log(4))


def test_retirements_are_censored_not_penalised() -> None:
    m = score(_fc([2.0, 1.0, 0.0]), Outcome("e", ("a", "c")))  # b retired
    assert m["n_classified"] == 2
    assert np.isfinite(m["finisher_pl_loglik"])


def test_score_rejects_unknown_finisher() -> None:
    with pytest.raises(ValueError, match="no entry"):
        score(_fc([0.0, 0.0]), Outcome("e", ("zz",)))


def test_spearman_basic() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert spearman(np.ones(4), a) == 0.0


def test_ece_perfect_and_terrible() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.01
    assert expected_calibration_error(np.full(1000, 0.9), np.zeros(1000)) == pytest.approx(0.9)
    table = reliability_table(p, y)
    assert sum(c for _, _, c in table) == p.size


def test_paired_bootstrap_detects_real_difference() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(size=300)
    a = b - 0.5 + rng.normal(scale=0.1, size=300)
    cmp = paired_bootstrap(a, b)
    assert cmp.significant
    assert cmp.hi < 0
    same = paired_bootstrap(b, b)
    assert same.mean_diff == 0.0
    assert not same.significant


def test_forecast_with_retirements() -> None:
    from f1p.eval.forecast import pl_win_probabilities

    ls = np.array([2.0, 1.0, 0.0, -1.0])
    pf = np.array([0.5, 0.95, 0.95, 0.95])
    fc = Forecast.from_log_strength("e", "C4", "m", ("a", "b", "c", "d"), ls, p_finish=pf)
    plain = pl_win_probabilities(ls)
    assert fc.p_win.sum() == pytest.approx(1.0)
    assert fc.p_win[0] < plain[0]
    assert fc.p_win[1] > plain[1]
    # Rao-Blackwellised P(win) agrees with the sampled winners
    np.testing.assert_allclose(fc.p_win, fc.position_probs[:, 0], atol=0.015)
    # the unreliable favourite spends half its races in the retirement slots
    assert fc.position_probs[0, 1:].sum() > 0.5 - 0.02
