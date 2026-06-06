"""Reusable Rich rendering helpers for the CLI (tables, panels, banners)."""
from __future__ import annotations

import pandas as pd
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..utils.logging import console

ACCENT = "bright_red"


def banner(subtitle: str = "") -> None:
    title = Text("  f1predict  ", style=f"bold white on {ACCENT}")
    sub = Text(f"  {subtitle}", style="dim")
    console.print(Panel(Text.assemble(title, sub), border_style=ACCENT, expand=False))


def _pos_style(rank: int) -> str:
    return {1: "bold yellow", 2: "bold white", 3: "bold #cd7f32"}.get(rank, "white")


def race_table(df: pd.DataFrame, title: str = "Predicted Result") -> Table:
    t = Table(title=title, header_style=f"bold {ACCENT}", expand=False, title_style="bold")
    t.add_column("#", justify="right")
    t.add_column("Driver")
    t.add_column("Team", style="dim")
    t.add_column("Grid", justify="right")
    t.add_column("P(win)", justify="right")
    t.add_column("P(podium)", justify="right")
    t.add_column("xPts", justify="right")
    for i, row in enumerate(df.itertuples(), start=1):
        t.add_row(
            Text(str(i), style=_pos_style(i)),
            str(getattr(row, "driver_name", "")),
            str(getattr(row, "constructor_name", "")),
            str(int(getattr(row, "grid", 0))),
            _bar(getattr(row, "p_win", 0.0)),
            _bar(getattr(row, "p_podium", 0.0)),
            f"{getattr(row, 'expected_points', getattr(row,'exp_points',0.0)):.1f}",
        )
    return t


def _bar(p: float, width: int = 10) -> Text:
    filled = int(round(p * width))
    bar = "█" * filled + "·" * (width - filled)
    color = "green" if p > 0.5 else "yellow" if p > 0.2 else "red"
    return Text(f"{bar} {p:5.1%}", style=color)


def metrics_panel(metrics: dict, title: str = "Metrics") -> Panel:
    lines = []
    pretty = {
        "mae": "MAE (positions)", "spearman": "Spearman ρ",
        "winner_accuracy": "Winner accuracy", "podium_accuracy": "Podium overlap",
        "n_races": "Races evaluated",
    }
    for k, label in pretty.items():
        if k in metrics:
            v = metrics[k]
            lines.append(f"[bold]{label}:[/bold] {v:.3f}" if isinstance(v, float) else f"[bold]{label}:[/bold] {v}")
    return Panel("\n".join(lines), title=title, border_style=ACCENT, expand=False)


def df_table(df: pd.DataFrame, title: str = "", max_rows: int = 25) -> Table:
    t = Table(title=title, header_style=f"bold {ACCENT}", title_style="bold")
    for col in df.columns:
        t.add_column(str(col))
    for row in df.head(max_rows).itertuples(index=False):
        t.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    return t


_PRESET_STYLE = {
    "test": "green", "light": "cyan", "medium": "yellow",
    "deep": "magenta", "max": "bold red",
}


def presets_overview(rows: list[dict], title: str = "Training presets") -> Table:
    """A beautiful overview of every training preset and what it costs you in time."""
    t = Table(
        title=title, header_style=f"bold {ACCENT}", title_style="bold",
        expand=False, show_lines=True,
    )
    t.add_column("Preset", justify="left")
    t.add_column("Est. time", justify="left")
    t.add_column("Ensemble", justify="left", style="dim")
    t.add_column("Trials", justify="right")
    t.add_column("CV", justify="right")
    t.add_column("Best for")
    for r in rows:
        style = _PRESET_STYLE.get(r["preset"], "white")
        t.add_row(
            Text(r["preset"], style=f"bold {style}"),
            Text(r["eta"], style=style),
            str(r["ensemble"]),
            str(r["trials"]),
            str(r["cv_folds"]),
            str(r["best_for"]),
        )
    return t


def pick_preset(rows: list[dict], default: str = "test") -> str:
    """Show the overview and force the user to pick a preset (interactive)."""
    from rich.prompt import Prompt

    console.print(presets_overview(rows))
    console.print(
        "[dim]Bigger presets train longer but predict more accurately. "
        "You can always re-train later.[/dim]"
    )
    choices = [r["preset"] for r in rows]
    return Prompt.ask(
        "[bold]Pick a training preset[/bold]",
        choices=choices, default=default, show_choices=True,
    )


def models_table(models: list[dict], title: str = "Trained models") -> Table:
    """A beautiful overview of every trained model in the store."""
    t = Table(
        title=title, header_style=f"bold {ACCENT}", title_style="bold",
        expand=False, show_lines=False,
    )
    t.add_column("", justify="center", width=3)        # active marker
    t.add_column("Name")
    t.add_column("Family", style="dim")
    t.add_column("Preset")
    t.add_column("Winner", justify="right")
    t.add_column("Podium", justify="right")
    t.add_column("MAE", justify="right")
    t.add_column("Size", justify="right", style="dim")
    t.add_column("Trained", style="dim")
    for m in models:
        met = m.get("metrics", {})
        marker = Text("●", style="bold green") if m["is_production"] else Text("·", style="dim")
        name_style = "bold green" if m["is_production"] else "bold cyan"
        wa = met.get("winner_accuracy")
        pa = met.get("podium_accuracy")
        mae = met.get("mae")
        t.add_row(
            marker,
            Text(m["name"], style=name_style),
            str(m["family"]),
            str(m["preset"]),
            f"{wa:.0%}" if isinstance(wa, (int, float)) else "—",
            f"{pa:.0%}" if isinstance(pa, (int, float)) else "—",
            f"{mae:.2f}" if isinstance(mae, (int, float)) else "—",
            f"{m['size_mb']:.0f} MB",
            m["modified"].strftime("%b %d %H:%M") if hasattr(m["modified"], "strftime") else str(m["modified"]),
        )
    return t
