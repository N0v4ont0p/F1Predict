"""Circuit metadata: track-type clusters and characteristics.

Track *type* is a strong prior for which driver/car traits matter (street circuits reward
precision & qualifying; high-speed power circuits reward engine & aero efficiency; technical
circuits reward downforce & mechanical grip). We map every circuit_id we expect to see to a
cluster plus a few continuous characteristics. Unknown circuits fall back to ``balanced``.

These are **static**, regulation-independent properties of the venue, so they are perfectly
safe to use as features (no leakage).
"""
from __future__ import annotations

# cluster, overtaking_difficulty (0 easy .. 1 very hard), avg_speed (0 slow .. 1 fast),
# power_sensitivity (0 low .. 1 high)
_TRACKS: dict[str, tuple[str, float, float, float]] = {
    # --- street circuits (qualifying-critical, hard to overtake) -------------------
    "monaco":        ("street", 0.98, 0.20, 0.25),
    "monte_carlo":   ("street", 0.98, 0.20, 0.25),
    "marina_bay":    ("street", 0.85, 0.30, 0.40),
    "baku":          ("street", 0.55, 0.70, 0.85),
    "jeddah":        ("street", 0.50, 0.85, 0.80),
    "albert_park":   ("street", 0.60, 0.62, 0.60),
    "miami":         ("street", 0.55, 0.66, 0.62),
    "vegas":         ("street", 0.45, 0.88, 0.90),
    "las_vegas":     ("street", 0.45, 0.88, 0.90),
    "madring":       ("street", 0.62, 0.58, 0.55),   # Madrid (2026+)
    "valencia":      ("street", 0.80, 0.55, 0.55),
    "detroit":       ("street", 0.90, 0.30, 0.35),
    # --- high-speed / power circuits ----------------------------------------------
    "monza":         ("high_speed", 0.40, 0.98, 0.95),
    "spa":           ("high_speed", 0.45, 0.92, 0.90),
    "silverstone":   ("high_speed", 0.50, 0.88, 0.78),
    "red_bull_ring": ("high_speed", 0.45, 0.82, 0.80),
    "americas":      ("high_speed", 0.48, 0.80, 0.72),
    "interlagos":    ("high_speed", 0.45, 0.78, 0.75),
    "suzuka":        ("high_speed", 0.55, 0.84, 0.70),
    "shanghai":      ("high_speed", 0.50, 0.74, 0.70),
    "villeneuve":    ("high_speed", 0.50, 0.76, 0.82),
    "rodriguez":     ("high_speed", 0.55, 0.72, 0.88),  # altitude, power-sensitive
    # --- technical / downforce circuits -------------------------------------------
    "hungaroring":   ("technical", 0.88, 0.42, 0.40),
    "catalunya":     ("technical", 0.72, 0.60, 0.58),
    "zandvoort":     ("technical", 0.78, 0.58, 0.55),
    "imola":         ("technical", 0.80, 0.62, 0.60),
    "portimao":      ("technical", 0.65, 0.64, 0.60),
    "mugello":       ("technical", 0.62, 0.80, 0.68),
    "yas_marina":    ("technical", 0.70, 0.62, 0.62),
    "losail":        ("technical", 0.60, 0.72, 0.60),   # Qatar
    "lusail":        ("technical", 0.60, 0.72, 0.60),
    "bahrain":       ("technical", 0.45, 0.66, 0.72),
    "sakhir":        ("technical", 0.45, 0.66, 0.72),
    "sochi":         ("technical", 0.70, 0.58, 0.55),
    "istanbul":      ("technical", 0.55, 0.70, 0.62),
    "nurburgring":   ("technical", 0.58, 0.70, 0.66),
    "hockenheim":    ("technical", 0.55, 0.72, 0.70),
    "paul_ricard":   ("technical", 0.68, 0.72, 0.66),
    "kyalami":       ("technical", 0.55, 0.74, 0.70),
}

_CLUSTERS = ["street", "high_speed", "technical", "balanced"]


# ---------------------------------------------------------------------------------------
# Rich, researched per-circuit environment & physical stats. These are **static** venue
# properties (no leakage) used both as model features and to seed the simulation's
# tyre-stress / chaos terms. Sources: circuit length & layout (FIA), surface abrasion &
# degradation (Pirelli tyre-allocation notes / historical stint data), geographic
# coordinates (for the weather API), and race-month climatology (typical conditions).
#
# Fields per circuit:
#   lat, lon            geographic coordinates (decimal degrees) — feed the weather API
#   degradation         tyre degradation severity   (0 low .. 1 very high)
#   abrasion            surface abrasiveness         (0 smooth .. 1 abrasive)
#   lap_km              circuit length in kilometres
#   downforce           aero downforce demand        (0 low .. 1 maximum)
#   altitude_m          elevation above sea level (affects PU & cooling)
#   clim_temp_c         typical race-day air temperature (°C)
#   clim_humidity       typical race-day relative humidity (0..1)
#   clim_rain_prob      historical probability of a wet/mixed race (0..1)
_TRACK_ENV: dict[str, dict[str, float]] = {
    "monaco":        {"lat": 43.7347, "lon": 7.4206,  "degradation": 0.20, "abrasion": 0.25, "lap_km": 3.337, "downforce": 0.98, "altitude_m": 7,    "clim_temp_c": 21, "clim_humidity": 0.66, "clim_rain_prob": 0.25},
    "monte_carlo":   {"lat": 43.7347, "lon": 7.4206,  "degradation": 0.20, "abrasion": 0.25, "lap_km": 3.337, "downforce": 0.98, "altitude_m": 7,    "clim_temp_c": 21, "clim_humidity": 0.66, "clim_rain_prob": 0.25},
    "marina_bay":    {"lat": 1.2914,  "lon": 103.8640,"degradation": 0.45, "abrasion": 0.45, "lap_km": 4.940, "downforce": 0.95, "altitude_m": 2,    "clim_temp_c": 29, "clim_humidity": 0.84, "clim_rain_prob": 0.45},
    "baku":          {"lat": 40.3725, "lon": 49.8533, "degradation": 0.35, "abrasion": 0.40, "lap_km": 6.003, "downforce": 0.35, "altitude_m": -20,  "clim_temp_c": 24, "clim_humidity": 0.60, "clim_rain_prob": 0.15},
    "jeddah":        {"lat": 21.6319, "lon": 39.1044, "degradation": 0.40, "abrasion": 0.45, "lap_km": 6.174, "downforce": 0.40, "altitude_m": 8,    "clim_temp_c": 30, "clim_humidity": 0.62, "clim_rain_prob": 0.05},
    "albert_park":   {"lat": -37.8497,"lon": 144.968, "degradation": 0.45, "abrasion": 0.45, "lap_km": 5.278, "downforce": 0.60, "altitude_m": 10,   "clim_temp_c": 20, "clim_humidity": 0.60, "clim_rain_prob": 0.30},
    "miami":         {"lat": 25.9581, "lon": -80.2389,"degradation": 0.50, "abrasion": 0.50, "lap_km": 5.412, "downforce": 0.62, "altitude_m": 2,    "clim_temp_c": 29, "clim_humidity": 0.72, "clim_rain_prob": 0.35},
    "vegas":         {"lat": 36.1147, "lon": -115.173,"degradation": 0.35, "abrasion": 0.40, "lap_km": 6.201, "downforce": 0.20, "altitude_m": 620,  "clim_temp_c": 15, "clim_humidity": 0.35, "clim_rain_prob": 0.10},
    "las_vegas":     {"lat": 36.1147, "lon": -115.173,"degradation": 0.35, "abrasion": 0.40, "lap_km": 6.201, "downforce": 0.20, "altitude_m": 620,  "clim_temp_c": 15, "clim_humidity": 0.35, "clim_rain_prob": 0.10},
    "madring":       {"lat": 40.4637, "lon": -3.6000, "degradation": 0.45, "abrasion": 0.50, "lap_km": 5.470, "downforce": 0.58, "altitude_m": 600,  "clim_temp_c": 26, "clim_humidity": 0.45, "clim_rain_prob": 0.15},
    "valencia":      {"lat": 39.4589, "lon": -0.3316, "degradation": 0.40, "abrasion": 0.45, "lap_km": 5.419, "downforce": 0.55, "altitude_m": 5,    "clim_temp_c": 27, "clim_humidity": 0.60, "clim_rain_prob": 0.10},
    "detroit":       {"lat": 42.3314, "lon": -83.0458,"degradation": 0.40, "abrasion": 0.50, "lap_km": 4.000, "downforce": 0.85, "altitude_m": 180,  "clim_temp_c": 23, "clim_humidity": 0.62, "clim_rain_prob": 0.25},
    "monza":         {"lat": 45.6156, "lon": 9.2811,  "degradation": 0.55, "abrasion": 0.55, "lap_km": 5.793, "downforce": 0.10, "altitude_m": 162,  "clim_temp_c": 26, "clim_humidity": 0.58, "clim_rain_prob": 0.25},
    "spa":           {"lat": 50.4372, "lon": 5.9714,  "degradation": 0.55, "abrasion": 0.55, "lap_km": 7.004, "downforce": 0.45, "altitude_m": 401,  "clim_temp_c": 18, "clim_humidity": 0.74, "clim_rain_prob": 0.55},
    "silverstone":   {"lat": 52.0786, "lon": -1.0169, "degradation": 0.65, "abrasion": 0.60, "lap_km": 5.891, "downforce": 0.65, "altitude_m": 153,  "clim_temp_c": 19, "clim_humidity": 0.70, "clim_rain_prob": 0.40},
    "red_bull_ring": {"lat": 47.2197, "lon": 14.7647, "degradation": 0.45, "abrasion": 0.45, "lap_km": 4.318, "downforce": 0.55, "altitude_m": 678,  "clim_temp_c": 22, "clim_humidity": 0.62, "clim_rain_prob": 0.40},
    "americas":      {"lat": 30.1328, "lon": -97.6411,"degradation": 0.50, "abrasion": 0.50, "lap_km": 5.513, "downforce": 0.70, "altitude_m": 180,  "clim_temp_c": 24, "clim_humidity": 0.64, "clim_rain_prob": 0.30},
    "interlagos":    {"lat": -23.7036,"lon": -46.6997,"degradation": 0.55, "abrasion": 0.55, "lap_km": 4.309, "downforce": 0.68, "altitude_m": 785,  "clim_temp_c": 22, "clim_humidity": 0.72, "clim_rain_prob": 0.50},
    "suzuka":        {"lat": 34.8431, "lon": 136.541, "degradation": 0.60, "abrasion": 0.55, "lap_km": 5.807, "downforce": 0.72, "altitude_m": 45,   "clim_temp_c": 20, "clim_humidity": 0.68, "clim_rain_prob": 0.45},
    "shanghai":      {"lat": 31.3389, "lon": 121.220, "degradation": 0.55, "abrasion": 0.55, "lap_km": 5.451, "downforce": 0.62, "altitude_m": 4,    "clim_temp_c": 18, "clim_humidity": 0.72, "clim_rain_prob": 0.40},
    "villeneuve":    {"lat": 45.5000, "lon": -73.5228,"degradation": 0.45, "abrasion": 0.40, "lap_km": 4.361, "downforce": 0.45, "altitude_m": 13,   "clim_temp_c": 22, "clim_humidity": 0.62, "clim_rain_prob": 0.35},
    "rodriguez":     {"lat": 19.4042, "lon": -99.0907,"degradation": 0.40, "abrasion": 0.45, "lap_km": 4.304, "downforce": 0.75, "altitude_m": 2240, "clim_temp_c": 21, "clim_humidity": 0.55, "clim_rain_prob": 0.40},
    "hungaroring":   {"lat": 47.5789, "lon": 19.2486, "degradation": 0.50, "abrasion": 0.45, "lap_km": 4.381, "downforce": 0.90, "altitude_m": 250,  "clim_temp_c": 27, "clim_humidity": 0.55, "clim_rain_prob": 0.30},
    "catalunya":     {"lat": 41.5700, "lon": 2.2611,  "degradation": 0.65, "abrasion": 0.60, "lap_km": 4.657, "downforce": 0.72, "altitude_m": 132,  "clim_temp_c": 24, "clim_humidity": 0.62, "clim_rain_prob": 0.20},
    "zandvoort":     {"lat": 52.3888, "lon": 4.5409,  "degradation": 0.50, "abrasion": 0.50, "lap_km": 4.259, "downforce": 0.80, "altitude_m": 5,    "clim_temp_c": 19, "clim_humidity": 0.74, "clim_rain_prob": 0.40},
    "imola":         {"lat": 44.3439, "lon": 11.7167, "degradation": 0.50, "abrasion": 0.50, "lap_km": 4.909, "downforce": 0.75, "altitude_m": 37,   "clim_temp_c": 19, "clim_humidity": 0.66, "clim_rain_prob": 0.35},
    "portimao":      {"lat": 37.2270, "lon": -8.6267, "degradation": 0.55, "abrasion": 0.55, "lap_km": 4.653, "downforce": 0.68, "altitude_m": 75,   "clim_temp_c": 19, "clim_humidity": 0.68, "clim_rain_prob": 0.25},
    "mugello":       {"lat": 43.9975, "lon": 11.3719, "degradation": 0.60, "abrasion": 0.60, "lap_km": 5.245, "downforce": 0.78, "altitude_m": 292,  "clim_temp_c": 27, "clim_humidity": 0.58, "clim_rain_prob": 0.25},
    "yas_marina":    {"lat": 24.4672, "lon": 54.6031, "degradation": 0.40, "abrasion": 0.45, "lap_km": 5.281, "downforce": 0.70, "altitude_m": 3,    "clim_temp_c": 28, "clim_humidity": 0.58, "clim_rain_prob": 0.03},
    "losail":        {"lat": 25.4900, "lon": 51.4542, "degradation": 0.55, "abrasion": 0.45, "lap_km": 5.419, "downforce": 0.72, "altitude_m": 12,   "clim_temp_c": 27, "clim_humidity": 0.55, "clim_rain_prob": 0.03},
    "lusail":        {"lat": 25.4900, "lon": 51.4542, "degradation": 0.55, "abrasion": 0.45, "lap_km": 5.419, "downforce": 0.72, "altitude_m": 12,   "clim_temp_c": 27, "clim_humidity": 0.55, "clim_rain_prob": 0.03},
    "bahrain":       {"lat": 26.0325, "lon": 50.5106, "degradation": 0.60, "abrasion": 0.65, "lap_km": 5.412, "downforce": 0.55, "altitude_m": 7,    "clim_temp_c": 27, "clim_humidity": 0.50, "clim_rain_prob": 0.05},
    "sakhir":        {"lat": 26.0325, "lon": 50.5106, "degradation": 0.60, "abrasion": 0.65, "lap_km": 5.412, "downforce": 0.55, "altitude_m": 7,    "clim_temp_c": 27, "clim_humidity": 0.50, "clim_rain_prob": 0.05},
    "sochi":         {"lat": 43.4057, "lon": 39.9578, "degradation": 0.35, "abrasion": 0.35, "lap_km": 5.848, "downforce": 0.55, "altitude_m": 3,    "clim_temp_c": 21, "clim_humidity": 0.66, "clim_rain_prob": 0.20},
    "istanbul":      {"lat": 40.9517, "lon": 29.4050, "degradation": 0.55, "abrasion": 0.55, "lap_km": 5.338, "downforce": 0.62, "altitude_m": 130,  "clim_temp_c": 18, "clim_humidity": 0.70, "clim_rain_prob": 0.35},
    "nurburgring":   {"lat": 50.3356, "lon": 6.9475,  "degradation": 0.50, "abrasion": 0.50, "lap_km": 5.148, "downforce": 0.66, "altitude_m": 578,  "clim_temp_c": 16, "clim_humidity": 0.74, "clim_rain_prob": 0.45},
    "hockenheim":    {"lat": 49.3278, "lon": 8.5658,  "degradation": 0.50, "abrasion": 0.50, "lap_km": 4.574, "downforce": 0.60, "altitude_m": 103,  "clim_temp_c": 23, "clim_humidity": 0.62, "clim_rain_prob": 0.35},
    "paul_ricard":   {"lat": 43.2506, "lon": 5.7917,  "degradation": 0.45, "abrasion": 0.50, "lap_km": 5.842, "downforce": 0.60, "altitude_m": 432,  "clim_temp_c": 26, "clim_humidity": 0.58, "clim_rain_prob": 0.15},
    "kyalami":       {"lat": -25.9986,"lon": 28.0767, "degradation": 0.55, "abrasion": 0.55, "lap_km": 4.522, "downforce": 0.66, "altitude_m": 1560, "clim_temp_c": 21, "clim_humidity": 0.55, "clim_rain_prob": 0.30},
}

# Field-wide defaults for an unknown circuit (a neutral, balanced venue).
_ENV_DEFAULT: dict[str, float] = {
    "lat": 0.0, "lon": 0.0, "degradation": 0.50, "abrasion": 0.50, "lap_km": 5.0,
    "downforce": 0.65, "altitude_m": 100, "clim_temp_c": 23, "clim_humidity": 0.60,
    "clim_rain_prob": 0.25,
}
# Reasonable normalisation ranges so engineered features land in ~[0,1].
_LAP_KM_RANGE = (3.0, 7.1)
_ALT_RANGE = (-25.0, 2300.0)
_TEMP_RANGE = (10.0, 35.0)


def _norm(x: float, lo: float, hi: float) -> float:
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def track_env(circuit_id: str) -> dict[str, float]:
    """Raw researched environment/physical stats for a circuit (with defaults)."""
    return {**_ENV_DEFAULT, **_TRACK_ENV.get(circuit_id, {})}


def track_coords(circuit_id: str) -> tuple[float, float]:
    """(latitude, longitude) for the circuit — used to query the weather API."""
    e = track_env(circuit_id)
    return float(e["lat"]), float(e["lon"])


def track_climate(circuit_id: str) -> dict[str, float]:
    """Race-month climatology (offline fallback when no measured weather is available)."""
    e = track_env(circuit_id)
    return {
        "wx_temp_c": float(e["clim_temp_c"]),
        "wx_humidity": float(e["clim_humidity"]),
        "wx_rain_prob": float(e["clim_rain_prob"]),
        "wx_wind_kmh": 12.0,
    }


def track_stats(circuit_id: str) -> dict[str, float]:
    """Normalised static track-physics features (degradation, abrasion, length, etc.)."""
    e = track_env(circuit_id)
    return {
        "trk_degradation": float(e["degradation"]),
        "trk_abrasion": float(e["abrasion"]),
        "trk_lap_km": _norm(e["lap_km"], *_LAP_KM_RANGE),
        "trk_downforce": float(e["downforce"]),
        "trk_altitude": _norm(e["altitude_m"], *_ALT_RANGE),
    }


def track_profile(circuit_id: str) -> dict[str, float]:
    """Return cluster one-hots + continuous characteristics for a circuit."""
    cluster, overtake, speed, power = _TRACKS.get(
        circuit_id, ("balanced", 0.6, 0.65, 0.65)
    )
    prof = {f"trk_is_{c}": 1.0 if c == cluster else 0.0 for c in _CLUSTERS}
    prof["trk_overtake_difficulty"] = overtake
    prof["trk_avg_speed"] = speed
    prof["trk_power_sensitivity"] = power
    prof.update(track_stats(circuit_id))
    return prof


def track_cluster(circuit_id: str) -> str:
    return _TRACKS.get(circuit_id, ("balanced", 0, 0, 0))[0]
