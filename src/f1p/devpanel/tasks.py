"""The fixed catalogue of tasks the dev panel can run.

Each task is a list of steps, each step one command. Commands use the tools of the running
environment (``f1p``, ``pytest`` …), so the server must be started with ``uv run``. The M6
checklist mirrors ``docs/plan/M6_FOLLOWUP_PLAN.md`` part B; items that need new code or a
manual action are listed but cannot be run (``manual``).
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

LAPSEQ_SEASONS = tuple(range(2019, 2027))
STACK_MEMBERS = "world_v1,world_v1_lapv2,race_sim_v2,tabicl_pointwise,lapseq,lgbm_rank"


@dataclass(frozen=True)
class Task:
    id: str
    group: str
    title: str
    detail: str
    steps: tuple[tuple[str, ...], ...] = ()
    after: tuple[str, ...] = ()  # tasks that should have succeeded first (advisory)
    manual: str = ""  # why it cannot be run from the panel
    confirm: str = ""  # a phrase the user must type to start it
    hours: float = 0.0  # rough expected duration, for the display
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def runnable(self) -> bool:
        return not self.manual and bool(self.steps)


def _f1p(*args: str) -> tuple[str, ...]:
    return ("f1p", *args)


TASKS: tuple[Task, ...] = (
    # ------------------------------------------------------------------ checks
    Task(
        "format",
        "Checks",
        "Format check",
        "ruff format --check",
        (("ruff", "format", "--check", "."),),
    ),
    Task("lint", "Checks", "Lint", "ruff check", (("ruff", "check", "."),)),
    Task("types", "Checks", "Types", "basedpyright", (("basedpyright",),)),
    Task("imports", "Checks", "Module boundaries", "lint-imports", (("lint-imports",),)),
    Task("tests", "Checks", "Test suite", "pytest, the full suite", (("pytest",),)),
    Task(
        "smoke",
        "Checks",
        "CLI smoke test",
        "The CI's smoke commands",
        (
            _f1p("version"),
            _f1p("doctor"),
            _f1p("silicon", "info"),
            _f1p("silicon", "bench", "--quick", "--no-save"),
        ),
    ),
    # ------------------------------------------------------------------ M6 checklist
    Task(
        "B1",
        "M6",
        "Sustained benchmark baseline",
        "Throughput and throttling under load: the timing basis for everything below",
        (_f1p("silicon", "bench", "--sustained", "5"),),
        hours=1.0,
    ),
    Task(
        "B2",
        "M6",
        "MLX simulator on the GPU",
        "Race-sim timing (target: 100k races < 2 s) and NumPy↔MLX parity",
        (_f1p("silicon", "bench"), ("pytest", "tests/silicon")),
        hours=0.2,
    ),
    Task(
        "B3",
        "M6",
        "lapseq GPU parity",
        "test_gpu_matches_cpu runs only on Apple silicon",
        (("pytest", "tests/models/test_lapseq.py"),),
        hours=0.2,
    ),
    Task(
        "B4",
        "M6",
        "lap_v2 posteriors, sequential",
        "4-chain MCMC per race over tune then select; run overnight",
        (
            _f1p("lap", "validate", "--split", "tune", "--sequential", "--store", "default"),
            _f1p("lap", "validate", "--split", "select", "--sequential", "--store", "default"),
        ),
        hours=12.0,
    ),
    Task(
        "B5",
        "M6",
        "lap_v2 gates on select",
        "Gate 3: does Bayesian pace help ranking?",
        (_f1p("eval", "run", "--models", "world_v1,world_v1_lapv2", "--split", "select"),),
        after=("B4",),
        hours=1.0,
    ),
    Task(
        "B6",
        "M6",
        "Simulator calibration and checks",
        "SMM calibration on tune, then predictive checks on select",
        (
            _f1p("sim", "calibrate", "--split", "tune", "--every", "1", "--sims", "2000"),
            _f1p("sim", "ppc", "--split", "select"),
        ),
        hours=6.0,
    ),
    Task(
        "B7",
        "M6",
        "Tuning campaigns",
        "filter 300, combiner 100, quali 100, race_sim 60 trials; one-SE choice on select",
        tuple(
            step
            for space, trials in (
                ("filter", 300),
                ("combiner", 100),
                ("quali", 100),
                ("race_sim", 60),
            )
            for step in (
                _f1p("tune", "run", space, "--trials", str(trials)),
                _f1p("tune", "select", space),
            )
        ),
        hours=10.0,
    ),
    Task(
        "B8",
        "M6",
        "Full-size lapseq",
        f"Train the full network for {LAPSEQ_SEASONS[0]}-{LAPSEQ_SEASONS[-1]}",
        tuple(_f1p("lapseq", "train", str(s), "--size", "full") for s in LAPSEQ_SEASONS),
        after=("B4",),
        hours=8.0,
    ),
    Task(
        "B9",
        "M6",
        "Pairwise TabICL",
        "Compare with pointwise on select",
        (
            _f1p(
                "eval", "run", "--models", "tabicl_pairwise", "--split", "select", "--workers", "1"
            ),
        ),
        hours=3.0,
    ),
    Task(
        "B10",
        "M6",
        "TabICL fine-tuning",
        "LoRA/full fine-tune on tune seasons",
        manual="Needs new code first.",
    ),
    Task(
        "B17",
        "M6",
        "SSM lapseq decision run",
        "Small SSM vs attention lapseq on select (A3's gate)",
        (
            _f1p(
                "eval",
                "run",
                "--models",
                "lapseq_ssm,lapseq,pit_cycle",
                "--split",
                "select",
                "--checkpoints",
                "C4,C5a,C5,C5b,C5c",
                "--workers",
                "1",
                "--reference",
                "lapseq",
            ),
            _f1p("lapseq", "nextlap", "2023", "--temporal", "ssm"),
        ),
        hours=4.0,
    ),
    Task(
        "B18",
        "M6",
        "SSM lapseq at full size",
        "Only if B17 adopts the SSM",
        manual="Decided by B17's result.",
    ),
    Task(
        "B11",
        "M6",
        "Stack v2 on select",
        "All members; freeze the member list and settings afterwards",
        (
            _f1p(
                "eval",
                "stack",
                "--split",
                "select",
                "--checkpoints",
                "C1,C4,C5",
                "--members",
                STACK_MEMBERS,
            ),
        ),
        after=("B4", "B5", "B6", "B7", "B8", "B9", "B17"),
        hours=2.0,
    ),
    Task(
        "B12",
        "M6",
        "Holdout, once",
        "Scores 2024-2026 once; recorded in the ledger. Nothing may change afterwards.",
        (
            _f1p("eval", "stack", "--split", "holdout", "--checkpoints", "C1,C4"),
            _f1p("eval", "run", "--split", "holdout", "--checkpoints", "C5"),
        ),
        after=("B11",),
        confirm="SCORE HOLDOUT",
        hours=2.0,
        tags=("holdout",),
    ),
    Task(
        "B13",
        "M6",
        "Overnight automation",
        "launchd plist running `just overnight <event>` after each race weekend",
        manual="Install the launchd plist by hand.",
    ),
)

PIPELINES: dict[str, tuple[str, ...]] = {
    "checks": ("format", "lint", "types", "imports", "tests", "smoke"),
    "hardware": ("B1", "B2", "B3"),
    # The plan's order up to the freeze; the holdout (B12) is never part of a pipeline.
    "m6": ("B1", "B2", "B3", "B4", "B6", "B7", "B5", "B8", "B9", "B17", "B11"),
}

BY_ID: dict[str, Task] = {t.id: t for t in TASKS}


def resolve(argv: tuple[str, ...]) -> list[str]:
    """The argv to execute: tools from this environment's ``bin`` first, then ``PATH``."""
    name, *rest = argv
    if name == "f1p":
        return [sys.executable, "-m", "f1p.cli.main", *rest]
    if name == "pytest":
        return [sys.executable, "-m", "pytest", *rest]
    local = Path(sys.executable).parent / name
    if local.exists():
        return [str(local), *rest]
    return [shutil.which(name) or name, *rest]
