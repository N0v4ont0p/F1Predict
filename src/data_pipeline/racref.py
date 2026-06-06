"""Race-reference parser — turn human shorthand into a concrete ``(season, round)``.

Users want to type natural shorthand instead of remembering round numbers:

* ``abudhabi2026``, ``abu_dhabi_2026``, ``yasmarina26`` → 2026 Abu Dhabi GP
* ``Uae2027``                                            → 2027 Abu Dhabi GP (country code)
* ``suzuka26`` / ``japan27`` / ``jpn26``                 → Japanese GP
* ``madring26``                                          → 2026 Madrid GP (NOT Barcelona)
* ``barca26`` / ``catalunya26`` / ``spain25``           → Barcelona/Catalunya
* ``vegas26`` / ``lasvegas2026``                         → Las Vegas GP
* ``monaco26`` / ``monte_carlo26`` / ``mc27``            → Monaco GP

The parser is **alias-rich and forgiving**: it strips separators/case, splits off a 2- or
4-digit year (2-digit ⇒ 20xx), normalises the remaining token through a large alias map to a
canonical ``circuit_id``, then resolves the round against that season's schedule (from the
master dataset for completed seasons, or the live jolpica calendar for future ones).

Disambiguation note: Spain hosts two 2026 events — ``catalunya`` (Barcelona GP, round 7) and
``madring`` (Madrid, the "Spanish Grand Prix", round 14). ``spainXX`` resolves to Madrid from
2026 onward (the official Spanish GP moved to Madrid) and to Barcelona before then; use
``barca``/``catalunya`` or ``madrid``/``madring`` to be explicit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ..config import Config, load_config
from ..utils.logging import get_logger

log = get_logger()


@dataclass
class RaceRef:
    season: int
    round: int | None
    circuit_id: str
    label: str

    def is_resolved(self) -> bool:
        return self.round is not None


# alias token -> canonical circuit_id. Keys are normalised (lowercase, no separators).
_ALIASES: dict[str, str] = {}


def _add(circuit_id: str, *aliases: str) -> None:
    _ALIASES[circuit_id] = circuit_id
    for a in aliases:
        _ALIASES[re.sub(r"[^a-z0-9]", "", a.lower())] = circuit_id


# --- current / recent circuits with country, code, city and GP-name aliases ----------
_add("bahrain", "sakhir", "bhr", "bahraingp")
_add("jeddah", "saudi", "saudiarabia", "ksa", "sau", "jed", "saudiarabiangp")
_add("albert_park", "australia", "melbourne", "aus", "australiangp", "albertpark")
_add("suzuka", "japan", "jpn", "jp", "japanesegp", "suz")
_add("shanghai", "china", "chn", "chinagp", "chinesegp")
_add("miami", "mia", "miamigp")
_add("imola", "emiliaromagna", "emilia", "ita_imola", "imolagp")
_add("monaco", "montecarlo", "mc", "mco", "monacogp", "monte")
_add("catalunya", "barcelona", "barca", "spaincatalunya", "barcelonagp", "cat")
_add("madring", "madrid", "madridgp", "madringp")
_add("villeneuve", "canada", "montreal", "can", "canadiangp", "gillesvilleneuve")
_add("red_bull_ring", "austria", "spielberg", "aut", "redbullring", "austriangp", "rbr")
_add("silverstone", "britain", "british", "uk", "gbr", "england", "britishgp", "silver")
_add("hungaroring", "hungary", "hun", "budapest", "hungariangp")
_add("spa", "belgium", "bel", "spafrancorchamps", "francorchamps", "belgiangp")
_add("zandvoort", "netherlands", "dutch", "ned", "nld", "holland", "dutchgp")
_add("monza", "italy", "ita", "italiangp", "monzagp")
_add("baku", "azerbaijan", "aze", "azerbaijangp")
_add("marina_bay", "singapore", "sin", "sgp", "singaporegp", "marinabay")
_add("americas", "usa", "austin", "cota", "usgp", "unitedstates", "unitedstatesgp", "texas")
_add("rodriguez", "mexico", "mex", "mexicocity", "mexicogp", "mexicocitygp", "hermanosrodriguez")
_add("interlagos", "brazil", "bra", "saopaulo", "braziliangp", "saopaulogp", "sp")
_add("vegas", "lasvegas", "lv", "vegasgp", "lasvegasgp", "las_vegas")
_add("losail", "qatar", "qat", "lusail", "qatargp", "doha")
_add("yas_marina", "abudhabi", "uae", "are", "abu", "yasmarina", "abudhabigp", "yas")
# --- historical venues (so older seasons resolve too) --------------------------------
_add("portimao", "portugal", "por", "algarve", "portuguesegp")
_add("mugello", "tuscany", "tuscangp")
_add("istanbul", "turkey", "tur", "turkishgp", "istanbulpark")
_add("sochi", "russia", "rus", "russiangp")
_add("nurburgring", "eifel", "eifelgp", "nurburg")
_add("hockenheim", "germany", "ger", "germangp", "hockenheimring")
_add("paul_ricard", "france", "fra", "frenchgp", "paulricard", "lecastellet")
_add("sepang", "malaysia", "mys", "malaysiangp")
_add("kyalami", "southafrica", "rsa", "southafricangp")


def _split_year(token: str) -> tuple[str, int | None]:
    """Pull a trailing 2- or 4-digit year off the token."""
    m = re.search(r"(\d{4}|\d{2})$", token)
    if not m:
        return token, None
    raw = m.group(1)
    year = int(raw)
    if len(raw) == 2:
        year += 2000
    name = token[: m.start()]
    return name, year


def parse_racref(ref: str, cfg: Config | None = None,
                 schedule: pd.DataFrame | None = None) -> RaceRef:
    """Parse a shorthand reference into a :class:`RaceRef` and resolve its round.

    ``schedule`` (columns ``season, round, circuit_id``) may be supplied to avoid a network
    call; otherwise the master dataset / live calendar is consulted as needed.
    """
    cfg = cfg or load_config()
    raw = ref.strip()
    token = re.sub(r"[^a-z0-9]", "", raw.lower())
    name, year = _split_year(token)
    name = name.strip("_")

    if not name:
        raise ValueError(f"could not find a circuit name in {ref!r}")
    circuit_id = _ALIASES.get(name)
    if circuit_id is None and len(name) >= 4:
        # Fuzzy: prefix match only, and only between reasonably-long tokens (avoids a short
        # country code like 'are' spuriously matching inside an unrelated word).
        cands = [cid for alias, cid in _ALIASES.items()
                 if len(alias) >= 4 and (alias.startswith(name) or name.startswith(alias))]
        if cands:
            circuit_id = max(set(cands), key=cands.count)
    if circuit_id is None:
        raise ValueError(
            f"unknown circuit {name!r} in {ref!r}. Try a country, city, circuit or GP name "
            f"(e.g. 'monaco26', 'japan27', 'abudhabi2026', 'madring26')."
        )

    # Spain disambiguation: 'spain'/'esp' => Madrid from 2026, Barcelona before.
    if name in ("spain", "esp", "spanish", "spaingp", "spanishgp"):
        circuit_id = "madring" if (year or 2026) >= 2026 else "catalunya"

    if year is None:
        year = _default_year(cfg)

    rnd = _resolve_round(cfg, year, circuit_id, schedule)
    label = f"{circuit_id} {year}" + (f" · R{rnd}" if rnd else " (round TBD)")
    return RaceRef(season=int(year), round=rnd, circuit_id=circuit_id, label=label)


def _default_year(cfg: Config) -> int:
    return int(cfg.get("data.end_season", 2026))


def _resolve_round(cfg: Config, season: int, circuit_id: str,
                   schedule: pd.DataFrame | None) -> int | None:
    """Find the round number for ``circuit_id`` in ``season``."""
    # 1) explicit schedule passed in.
    if schedule is not None and len(schedule):
        m = schedule[(schedule["season"] == season) & (schedule["circuit_id"] == circuit_id)]
        if len(m):
            return int(m["round"].iloc[0])

    # 2) master dataset (completed seasons).
    try:
        from .build import load_master
        master = load_master(cfg)
        m = master[(master["season"] == season) & (master["circuit_id"] == circuit_id)]
        if len(m):
            return int(m["round"].iloc[0])
    except Exception:  # pragma: no cover - master may be absent in odd setups
        pass

    # 3) live calendar (future seasons not yet in master).
    try:
        from .jolpica import fetch_schedule
        cal = fetch_schedule(season, cfg.get("data.jolpica_base_url"),
                             int(cfg.get("data.request_timeout")))
        if len(cal):
            m = cal[cal["circuit_id"] == circuit_id]
            if len(m):
                return int(m["round"].iloc[0])
    except Exception as exc:  # pragma: no cover - offline / future-unknown
        log.warning(f"could not fetch {season} calendar: {exc}")
    return None
