from __future__ import annotations

import math

import numpy as np
import pytest

from f1p.models import reset_study as rs
from f1p.models.ratings import RaceObs, RatingParams


def _history(
    seasons: range, *, scramble: bool, seed: int = 0, spread: float = 1.2
) -> list[RaceObs]:
    """Six two-car teams. Team strength persists between ordinary seasons; at a reset season
    (2009, 2014, 2017) it is drawn afresh when ``scramble``."""
    from f1p.core.regulations import MAJOR_RESETS

    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1.2, 6)
    out = []
    for s in seasons:
        if scramble and s in MAJOR_RESETS:
            strength = rng.normal(0, spread, 6)
        else:
            strength = 0.95 * strength + rng.normal(0, 0.1, 6)
        for r in range(1, 11):
            eta = np.repeat(strength, 2) + rng.gumbel(size=12)
            out.append(
                RaceObs(
                    f"{s}-R{r:02d}",
                    s,
                    r,
                    tuple(f"d{i}" for i in range(12)),
                    tuple(f"t{i // 2}" for i in range(12)),
                    np.argsort(-eta).astype(np.intp),
                )
            )
    return out


def test_variants_are_the_pre_registered_ones() -> None:
    assert set(rs.VARIANTS) == {"H1", "H2", "H3", "H1+H3", "H2+H3"}
    assert set(rs.SIMPLICITY) == set(rs.VARIANTS)
    for names in rs.VARIANTS.values():
        assert set(names) <= set(RatingParams.RESET_FIELDS)


def test_scrambling_resets_are_found() -> None:
    """When resets do re-draw team strength, the fit lowers the carryover across them and
    forecasts the reset seasons better than the base filter."""
    races = _history(range(2004, 2020), scramble=True)
    base = RatingParams()
    fit = rs.fit_variant(races, "H3", base, score_from=2005, train_until=2019)
    assert fit.params.reset_carryover_ratio < 0.7
    mask = rs.scored_mask(races, score_from=2005, train_until=2019)
    assert fit.loglik > float(rs.race_logliks(races, base)[mask].sum())


def test_each_fold_never_sees_its_reset() -> None:
    """A fold's parameters are exactly the fit with its own reset season left out."""
    races = _history(range(2004, 2020), scramble=True)
    base = RatingParams()
    folds = rs.leave_one_reset_out(
        races, "H3", base, resets=[2009, 2014, 2017], score_from=2005, train_until=2019
    )
    assert [f.held_out for f in folds] == [2009, 2014, 2017]
    for f in folds:
        alone = rs.fit_variant(
            races, "H3", base, score_from=2005, train_until=2019, exclude=(f.held_out,)
        )
        assert f.params == alone.params
        assert f.block == rs.reset_blocks(races, f.params, base, [f.held_out])[0]


def test_held_out_resets_gain_when_resets_scramble() -> None:
    """With six resets that clearly re-draw the order, held-out resets gain on the whole."""
    races = _history(range(1990, 2024), scramble=True, spread=2.0)
    base = RatingParams()
    resets = rs.resets_between(1991, 2023)
    folds = rs.leave_one_reset_out(
        races, "H3", base, resets=resets, score_from=1991, train_until=2023
    )
    gains = [f.block.season_total for f in folds]
    assert sum(gains) > 0
    assert sum(g > 0 for g in gains) >= len(gains) // 2


def test_without_scrambling_the_prior_keeps_no_change() -> None:
    """When resets change nothing, the shrinkage keeps the reset parameters near 1."""
    races = _history(range(2004, 2020), scramble=False)
    fit = rs.fit_variant(races, "H3", RatingParams(), score_from=2005, train_until=2019)
    assert abs(math.log(fit.params.reset_carryover_ratio)) < 0.35


def test_nothing_scored_means_no_change() -> None:
    """With every reset season excluded and none in range, the penalty alone decides: 1."""
    races = _history(range(2010, 2014), scramble=False)  # no reset in 2010-2013
    fit = rs.fit_variant(races, "H1+H3", RatingParams(), score_from=2010, train_until=2013)
    assert fit.params.reset_pu_ratio == pytest.approx(1.0, abs=0.02)
    assert fit.params.reset_carryover_ratio == pytest.approx(1.0, abs=0.02)


def test_fits_never_run_past_train_until() -> None:
    races = _history(range(2004, 2020), scramble=True)
    early = rs.fit_variant(races, "H3", RatingParams(), score_from=2005, train_until=2012)
    # identical to fitting on the truncated history: later races are never used
    trunc = [r for r in races if r.season <= 2012]
    again = rs.fit_variant(trunc, "H3", RatingParams(), score_from=2005, train_until=2012)
    assert early.params == again.params


def test_reset_blocks_compare_against_the_base() -> None:
    races = _history(range(2012, 2016), scramble=True)
    base = RatingParams()
    same = rs.reset_blocks(races, base, base, [2014])[0]
    assert (same.early, same.season_total, same.next_season) == (0.0, 0.0, 0.0)
    other = rs.reset_blocks(races, RatingParams(reset_carryover_ratio=0.3), base, [2014])[0]
    assert other.season_total != 0.0
    assert rs.resets_between(1980, 2023) == [1983, 1989, 1995, 1998, 2009, 2014, 2017, 2022]
