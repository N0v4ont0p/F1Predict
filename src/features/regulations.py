"""Formula 1 regulation eras & rule-change awareness.

Major regulation resets scramble the competitive order: a team that dominated one era can
fall to the midfield overnight (e.g. 2014 hybrid switch, 2022 ground-effect, and especially
the **2026 overhaul** — all-new power units with 50% electrical power + active aerodynamics).

We expose:

* :func:`regulation_era` — a stable era id for a season (used for an embedding-style one-hot
  and for *down-weighting* cross-era history).
* :func:`seasons_since_reset` — how many seasons into the current rule-set we are. Early in a
  new rule-set, prior-car performance is a weak signal, so the model should lean on driver
  skill (ELO) more than constructor history. This is the key mechanism that stops the model
  from blindly assuming 2025's pecking order holds in the 2026 reset.
* :func:`is_regulation_reset_year` — first year of a new rule-set.
* :data:`MAJOR_RESETS` — the seasons that began a brand-new aero/PU rule-set.
"""
from __future__ import annotations

# Season that *started* each major aero/power-unit regulation reset.
MAJOR_RESETS = [2009, 2014, 2017, 2022, 2026]

# Named eras (lo, hi inclusive) — coarse buckets for an era one-hot.
_ERAS: dict[str, tuple[int, int]] = {
    "v8":          (2006, 2013),
    "hybrid_v1":   (2014, 2016),
    "hybrid_wide": (2017, 2021),
    "ground_fx":   (2022, 2025),
    "reg_2026":    (2026, 2099),
}

_ERA_KEYS = list(_ERAS)


def regulation_era(season: int) -> str:
    for era, (lo, hi) in _ERAS.items():
        if lo <= season <= hi:
            return era
    return "pre_2006"


def seasons_since_reset(season: int) -> int:
    """Number of completed seasons since the most recent rule-set began (0 == reset year)."""
    applicable = [r for r in MAJOR_RESETS if r <= season]
    if not applicable:
        return 5  # deep into a stable rule-set
    return season - max(applicable)


def is_regulation_reset_year(season: int) -> int:
    return 1 if season in MAJOR_RESETS else 0


def reset_weight(season: int) -> float:
    """Weight (0..1) for how much *constructor* history should be trusted this season.

    In a reset year, prior-car pace barely transfers, so we shrink it hard (0.25) and let
    driver ELO dominate; trust recovers as the rule-set matures.
    """
    s = seasons_since_reset(season)
    return min(1.0, 0.25 + 0.25 * s)


def era_onehot(season: int) -> dict[str, float]:
    era = regulation_era(season)
    feats = {f"reg_era_{k}": 1.0 if k == era else 0.0 for k in _ERA_KEYS}
    feats["reg_seasons_since_reset"] = float(seasons_since_reset(season))
    feats["reg_is_reset_year"] = float(is_regulation_reset_year(season))
    feats["reg_constructor_trust"] = reset_weight(season)
    return feats


# --------------------------------------------------------------------------- rule registry
# A compact, curated knowledge base of what each era's rules *do to a race*. The numeric
# modifiers feed the Monte-Carlo engine so simulations are regulation-native rather than a
# one-size-fits-all Gaussian. Values are expert priors (calibrated against the historical
# DNF/upset record), not learned parameters — see docs/DESIGN.md for the rationale.
#
# Modifier meanings (all multiplicative around 1.0 unless noted):
#   chaos               variance multiplier on race order (>1 == more upsets)
#   reliability         DNF-rate multiplier (>1 == more retirements; reset/new-PU years bite)
#   overtake_difficulty 0..1 — how hard it is to pass (higher == grid order sticks)
#   safety_car_rate     P(at least one order-bunching safety car) per race
#   tire_wear_var       extra strategy-driven variance folded into chaos
_RULES: dict[str, dict] = {
    "v8": {
        "label": "V8 naturally-aspirated (2006–2013)",
        "changes": [
            "2.4L V8 NA engines, frozen development",
            "2009 aero overhaul: wide front / narrow rear wings, slick tyres return, KERS debuts",
            "2011 DRS + Pirelli high-deg tyres introduced (more pit-stop strategy)",
            "2010 refuelling ban — race fuel loads, heavier starts",
        ],
        "chaos": 1.05, "reliability": 1.10, "overtake_difficulty": 0.55,
        "safety_car_rate": 0.45, "tire_wear_var": 1.15,
    },
    "hybrid_v1": {
        "label": "Hybrid V6 turbo — first generation (2014–2016)",
        "changes": [
            "1.6L V6 turbo-hybrid power units (MGU-H/MGU-K), huge early reliability spread",
            "Mercedes PU dominance; large field stratification",
            "Token-based PU development restrictions",
            "Brake-by-wire, complex energy deployment",
        ],
        "chaos": 1.15, "reliability": 1.35, "overtake_difficulty": 0.50,
        "safety_car_rate": 0.45, "tire_wear_var": 1.10,
    },
    "hybrid_wide": {
        "label": "Wide-car hybrid (2017–2021)",
        "changes": [
            "Wider cars & tyres, much higher downforce, faster cornering",
            "Overtaking harder (dirty air) — grid position more decisive",
            "2019 simplified front wing to aid following",
            "Mature, reliable power units",
        ],
        "chaos": 0.95, "reliability": 0.95, "overtake_difficulty": 0.70,
        "safety_car_rate": 0.40, "tire_wear_var": 1.00,
    },
    "ground_fx": {
        "label": "Ground-effect (2022–2025)",
        "changes": [
            "Ground-effect floors — cars can follow closely, easier overtaking",
            "18-inch wheels, simplified aero, $135M cost cap tightens",
            "Porpoising / floor regs evolution",
            "Red Bull then McLaren competitive cycles",
        ],
        "chaos": 1.00, "reliability": 0.90, "overtake_difficulty": 0.45,
        "safety_car_rate": 0.40, "tire_wear_var": 1.05,
    },
    "reg_2026": {
        "label": "2026 overhaul — new PUs + active aero",
        "changes": [
            "All-new power units: ~50% electrical power, no MGU-H, 100% sustainable fuel",
            "Active aerodynamics (movable front & rear wings), nimbler/lighter cars",
            "Manufacturer reshuffle (Audi, Honda-Aston, Red Bull Ford) — wide PU spread",
            "Expect first-year reliability volatility and a scrambled pecking order",
        ],
        "chaos": 1.30, "reliability": 1.45, "overtake_difficulty": 0.40,
        "safety_car_rate": 0.50, "tire_wear_var": 1.20,
    },
    "pre_2006": {
        "label": "Pre-2006 (V10 era and earlier)",
        "changes": ["High-rev V10 engines, frequent mechanical retirements"],
        "chaos": 1.15, "reliability": 1.30, "overtake_difficulty": 0.50,
        "safety_car_rate": 0.45, "tire_wear_var": 1.20,
    },
}

_MODIFIER_KEYS = ("chaos", "reliability", "overtake_difficulty", "safety_car_rate", "tire_wear_var")


def era_modifiers(season: int) -> dict[str, float]:
    """Regulation-driven simulation modifiers for a season.

    The base era priors are amplified in/just after a reset year: a brand-new rule-set is
    less reliable and more chaotic until teams converge. ``seasons_since_reset`` decays that
    amplification back toward the era baseline.
    """
    era = regulation_era(season)
    base = {k: float(_RULES[era][k]) for k in _MODIFIER_KEYS}
    s = seasons_since_reset(season)
    # Reset amplification: +0 at 5+ seasons in, up to +35% chaos / +30% DNF in year 0.
    amp = max(0.0, 1.0 - s / 4.0)
    base["chaos"] *= 1.0 + 0.35 * amp
    base["reliability"] *= 1.0 + 0.30 * amp
    return base


def era_summary(season: int) -> dict:
    """A structured, display-ready summary of a season's regulatory context."""
    era = regulation_era(season)
    rule = _RULES[era]
    return {
        "season": season,
        "era": era,
        "label": rule["label"],
        "seasons_since_reset": seasons_since_reset(season),
        "is_reset_year": bool(is_regulation_reset_year(season)),
        "constructor_trust": round(reset_weight(season), 3),
        "changes": list(rule["changes"]),
        "modifiers": {k: round(v, 3) for k, v in era_modifiers(season).items()},
    }
