"""Formula 1 championship points systems across eras.

Points systems changed several times. We support the major historical schemes and
default to the modern (2010+) system. ``fastest_lap`` adds 1 point (2019+) when the
driver finished in the top 10.
"""
from __future__ import annotations

# Mapping: finishing position -> points, per era keyword.
POINTS_SYSTEMS: dict[str, dict[int, float]] = {
    # 2010+ : 25-18-15-12-10-8-6-4-2-1
    "modern": {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1},
    # 2003-2009 : 10-8-6-5-4-3-2-1
    "points_2003": {1: 10, 2: 8, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1},
    # 1991-2002 : 10-6-4-3-2-1
    "points_1991": {1: 10, 2: 6, 3: 4, 4: 3, 5: 2, 6: 1},
    # 1960-1990 : 9-6-4-3-2-1
    "points_1960": {1: 9, 2: 6, 3: 4, 4: 3, 5: 2, 6: 1},
    # 1950-1959 : 8-6-4-3-2 (+1 fastest lap)
    "points_1950": {1: 8, 2: 6, 3: 4, 4: 3, 5: 2},
}

# Sprint race points (2022+): top 8 score.
SPRINT_POINTS: dict[int, float] = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def system_for_season(season: int) -> str:
    if season >= 2010:
        return "modern"
    if season >= 2003:
        return "points_2003"
    if season >= 1991:
        return "points_1991"
    if season >= 1960:
        return "points_1960"
    return "points_1950"


def points_for_position(
    position: int,
    season: int = 2025,
    session: str = "R",
    fastest_lap: bool = False,
) -> float:
    """Return championship points for a finishing ``position`` in ``season``.

    ``session`` ``"S"`` uses the sprint table. ``fastest_lap`` adds 1 point when the
    driver finished in the top 10 (2019+).
    """
    if position is None or position < 1:
        return 0.0
    table = SPRINT_POINTS if session.upper() == "S" else POINTS_SYSTEMS[system_for_season(season)]
    pts = float(table.get(int(position), 0.0))
    if fastest_lap and session.upper() == "R" and season >= 2019 and position <= 10:
        pts += 1.0
    return pts
