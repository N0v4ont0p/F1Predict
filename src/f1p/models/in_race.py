"""In-race baseline with the pit cycle (plan M6 §D.2, the second ``C5`` baseline).

The running order flatters cars that have not stopped yet. Here each running car's gap to
the leader gets the pit losses it still owes added:

    adjusted_i = gap_i + pit_loss · E[stops still to make_i],

with the circuit's pit loss and its distribution of stops per race from earlier visits
(:func:`f1p.models.race_sim.circuit_profile`, cut at the start). The stops still owed are
the expectation over that distribution given the stops already made. The score is
``adjusted / sqrt(laps left)``: a gap matters more the less race remains to close it (lap
times add up like a random walk). β and P(classified | running) are fitted as for the
running order (:class:`f1p.eval.baselines.InRacePL`).
"""

from __future__ import annotations

import numpy as np

from f1p.eval.baselines import InRacePL, InRaceSignal
from f1p.eval.context import EventContext
from f1p.eval.in_race import race_state
from f1p.models.race_sim import circuit_profile


def stops_owed(made: np.ndarray, p_stops: np.ndarray) -> np.ndarray:
    """E[K - made | K >= made] for K distributed as ``p_stops`` (0, 1, 2, 3 stops); none
    owed once ``made`` reaches the most stops the distribution allows."""
    k = np.arange(p_stops.size, dtype=np.float64)
    out = np.zeros(made.size)
    for i, m in enumerate(made):
        w = np.where(k >= m, p_stops, 0.0)
        out[i] = float(((k - m) * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    return out


def pit_cycle(ctx: EventContext) -> InRaceSignal | None:
    st = race_state(ctx)
    if not st.running.any():
        return None
    prof = circuit_profile(ctx.known("laps"), ctx.tables["events"], ctx.circuit_id)
    gap = st.gap_s.copy()
    timed = st.running & np.isfinite(gap)
    if not timed.any():
        return None
    # running cars without timing at lap k: just behind the last timed car
    gap = np.where(st.running & ~timed, float(np.max(gap[timed])) + 1.0, gap)
    owed = stops_owed(st.stops.astype(np.float64), np.asarray(prof.stops))
    adjusted = gap + prof.pit_loss_s * owed
    score = adjusted / np.sqrt(max(st.laps_left, 1))
    return InRaceSignal(np.where(st.running, score, np.nan), st.running)


def pit_cycle_baseline() -> InRacePL:
    return InRacePL("pit_cycle", pit_cycle)
