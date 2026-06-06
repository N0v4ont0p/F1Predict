"""Interactive REPL shell for f1predict — the "Claude Code" smooth-CLI experience.

Most CLIs make you retype the program name and full flags every single time. Tools like
Claude Code feel smooth because they're a *persistent session*: you launch once and then
issue short commands, with history, tab-completion and instant responses because heavy
state stays warm in memory.

This module turns f1predict into exactly that. Launching ``f1predict`` with no arguments
(or ``f1predict shell``) drops you into a session where you type bare verbs:

    f1predict ❯ predict suzuka26
    f1predict ❯ simulate suzuka26 --weather rain
    f1predict ❯ model
    f1predict ❯ /help

Design choices
--------------
* **prompt_toolkit** powers history (persisted to ``~/.f1predict/history``), fish-style
  auto-suggestions from history, and a context-aware tab completer that knows commands,
  sub-commands, options, race references, presets and model names.
* **Warm caches.** ``load_master`` / ``get_feature_frame`` / ``load_production`` are now
  mtime-cached, so the first ``predict`` pays the cost once and every later command in the
  session is instant — the single biggest contributor to "smooth".
* **No process churn.** Each line is parsed with :func:`shlex.split` and dispatched into the
  existing Typer/Click app with ``standalone_mode=False`` so a command (or a bad flag, or
  ``--help``) never tears down the session — errors are caught and printed, and the loop
  continues.
* **Meta commands** use a leading ``/`` (``/help``, ``/exit``, ``/clear``, ``/reload``) so
  they never collide with real commands.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..utils.logging import console
from . import render

ACCENT = render.ACCENT

# Commands that take a race reference as their first argument → offer venue completions.
_RACE_COMMANDS = {"predict", "simulate", "whatif", "weather"}
_PRESETS = ["test", "light", "medium", "deep", "max"]
_META = {
    "/help": "Show this help",
    "/exit": "Leave the shell (also /quit, /q, or Ctrl-D)",
    "/clear": "Clear the screen",
    "/reload": "Drop cached data/model so the next command reloads fresh",
    "/model": "List trained models (shortcut for `model`)",
}


def _history_path() -> Path:
    d = Path.home() / ".f1predict"
    d.mkdir(parents=True, exist_ok=True)
    return d / "history"


def _click_app():
    """The Typer app compiled to a Click command (cached on the function)."""
    if not hasattr(_click_app, "_cmd"):
        from typer.main import get_command
        from .main import app
        _click_app._cmd = get_command(app)
    return _click_app._cmd


def _command_tree() -> dict[str, dict]:
    """Introspect the Click app once into ``{name: {"opts": [...], "subs": {...}}}``."""
    if hasattr(_command_tree, "_tree"):
        return _command_tree._tree
    root = _click_app()
    tree: dict[str, dict] = {}

    def opts_of(cmd) -> list[str]:
        out: list[str] = []
        for p in getattr(cmd, "params", []):
            for o in getattr(p, "opts", []):
                if isinstance(o, str) and o.startswith("-"):
                    out.append(o)
        return out

    for name, cmd in getattr(root, "commands", {}).items():
        if getattr(cmd, "hidden", False):
            continue
        node = {"opts": opts_of(cmd), "subs": {}}
        for sub_name, sub in getattr(cmd, "commands", {}).items():
            if getattr(sub, "hidden", False):
                continue
            node["subs"][sub_name] = {"opts": opts_of(sub)}
        tree[name] = node
    _command_tree._tree = tree
    return tree


def _race_tokens() -> list[str]:
    try:
        from ..data_pipeline.racref import _ALIASES
        return sorted(set(_ALIASES.keys()))
    except Exception:  # noqa: BLE001
        return []


def _model_names(cfg) -> list[str]:
    try:
        from ..models import list_models
        return [m["name"] for m in list_models(cfg)]
    except Exception:  # noqa: BLE001
        return []


def _build_completer(cfg):
    from prompt_toolkit.completion import Completer, Completion

    tree = _command_tree()
    top = sorted(tree) + list(_META)
    race_tokens = _race_tokens()

    class F1Completer(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            parts = text.split()
            ends_space = text == "" or text.endswith(" ")
            cur = "" if ends_space else (parts[-1] if parts else "")
            # index of the word currently being typed
            completed = parts[:-1] if (parts and not ends_space) else parts

            def emit(cands, descs=None):
                descs = descs or {}
                for c in cands:
                    if c.startswith(cur):
                        yield Completion(c, start_position=-len(cur),
                                         display_meta=descs.get(c, ""))

            # 1) first word → top-level command or meta
            if len(completed) == 0:
                yield from emit(top, _META)
                return

            cmd = completed[0]
            node = tree.get(cmd)

            # 2) option completion (any token starting with '-')
            if cur.startswith("-"):
                opts = []
                if node:
                    opts = list(node["opts"])
                    if len(completed) >= 2 and completed[1] in node["subs"]:
                        opts = node["subs"][completed[1]]["opts"]
                yield from emit(sorted(set(opts)))
                return

            # 3) value after --preset
            if completed and completed[-1] in ("--preset", "-p"):
                yield from emit(_PRESETS)
                return

            # 4) sub-command groups
            if node and node["subs"] and len(completed) == 1:
                subs = sorted(node["subs"])
                if cmd == "model":
                    subs = subs + _model_names(cfg)
                yield from emit(subs)
                return

            # 4b) `model delete …` → model names plus the clear/all keywords
            if cmd == "model" and len(completed) >= 2 and completed[1] == "delete":
                yield from emit(_model_names(cfg) + ["clear", "all"])
                return

            # 5) race-reference commands → venue tokens
            if cmd in _RACE_COMMANDS and len(completed) == 1:
                yield from emit(race_tokens)
                return

            # 6) fallback: offer this command's options
            if node:
                yield from emit(sorted(set(node["opts"])))

    return F1Completer()


def _dispatch(argv: list[str]) -> None:
    """Run one parsed command line through the Click app without exiting the process."""
    cmd = _click_app()
    try:
        cmd.main(args=argv, prog_name="f1predict", standalone_mode=False)
    except SystemExit:
        pass  # --help / typer.Exit / argument errors already printed
    except click.ClickException as exc:
        exc.show()
    except click.Abort:
        console.print("[yellow]aborted[/yellow]")
    except KeyboardInterrupt:
        console.print("[yellow]^C[/yellow]")
    except Exception as exc:  # noqa: BLE001 — a bad command must never kill the session
        console.print(f"[red]error:[/red] {exc}")


def _print_welcome(cfg) -> None:
    from .. import __version__

    active = "—"
    try:
        from ..models import list_models
        models = list_models(cfg)
        act = next((m for m in models if m["is_production"]), None)
        active = act["name"] if act else (models[0]["name"] if models else "none trained")
    except Exception:  # noqa: BLE001
        pass

    body = Text.assemble(
        ("Interactive shell", f"bold {ACCENT}"),
        ("  ·  type commands without the ", "dim"),
        ("f1predict", "cyan"),
        (" prefix\n\n", "dim"),
        ("  predict suzuka26", "cyan"), ("        probabilistic race result\n", "dim"),
        ("  simulate vegas26 --weather rain", "cyan"), ("  Monte-Carlo\n", "dim"),
        ("  model            ", "cyan"), ("       list / switch trained models\n", "dim"),
        ("  /help  /reload  /exit", "cyan"), ("    shell meta-commands\n\n", "dim"),
        ("Tab", "bold white"), (" completes commands, races, presets & models   ", "dim"),
        ("↑/↓", "bold white"), (" history", "dim"),
    )
    console.print(Panel(body, title=f"🏎️  f1predict [bold {ACCENT}]v{__version__}[/]",
                        subtitle=f"active model · [green]{active}[/green]",
                        border_style=ACCENT, expand=False))


def _help_panel() -> None:
    t = Table(title="Shell meta-commands", header_style=f"bold {ACCENT}", expand=False)
    t.add_column("Command", style="cyan"); t.add_column("Does")
    for k, v in _META.items():
        t.add_row(k, v)
    console.print(t)
    tree = _command_tree()
    cmds = Text("  ".join(sorted(tree)), style="cyan")
    console.print(Panel(cmds, title="Available commands (run `<command> --help` for flags)",
                        border_style="dim", expand=False))


def _warm_caches(cfg) -> None:
    """Pre-load data/features/model so the first real command is instant."""
    try:
        from ..data_pipeline import load_master
        from ..whatif import get_feature_frame
        with console.status("[dim]warming up — loading data, features & model…[/dim]",
                            spinner="dots"):
            master = load_master(cfg)
            get_feature_frame(master, cfg)
            try:
                from ..models import load_production
                load_production(cfg)
            except Exception:  # noqa: BLE001 — fine if no model trained yet
                pass
    except Exception:  # noqa: BLE001 — fine if no data built yet
        pass


def _reload(cfg) -> None:
    from ..data_pipeline import clear_caches, clear_schedule_cache
    from ..whatif import clear_feature_cache
    from ..models import clear_model_cache

    clear_caches(); clear_feature_cache(); clear_model_cache(); clear_schedule_cache()
    console.print("[green]✓[/green] caches cleared — next command reloads fresh.")
    _warm_caches(cfg)


def run_shell(config: str | None = None, override: list[str] | None = None) -> None:
    """Launch the interactive f1predict shell."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.styles import Style
    except ImportError:
        console.print("[red]The interactive shell needs prompt_toolkit.[/red]\n"
                      "Install it with:  [cyan]pip install prompt_toolkit[/cyan]")
        return

    from ..config import load_config
    cfg = load_config(config, None)

    _print_welcome(cfg)
    _warm_caches(cfg)

    style = Style.from_dict({"prompt": "bold #E10600", "arrow": "#888888"})

    def prompt_fragments():
        return FormattedText([("class:prompt", "f1predict "), ("class:arrow", "❯ ")])

    session = PromptSession(
        history=FileHistory(str(_history_path())),
        auto_suggest=AutoSuggestFromHistory(),
        completer=_build_completer(cfg),
        complete_while_typing=True,
        style=style,
    )

    while True:
        try:
            line = session.prompt(prompt_fragments)
        except KeyboardInterrupt:
            console.print("[dim]( ^C — type /exit or Ctrl-D to leave )[/dim]")
            continue
        except EOFError:
            break

        line = line.strip()
        if not line:
            continue

        # meta / slash commands
        low = line.lower()
        if low in ("/exit", "/quit", "/q", "exit", "quit", ":q"):
            break
        if low in ("/help", "/?", "help", "?"):
            _help_panel(); continue
        if low in ("/clear", "/cls", "clear"):
            console.clear(); continue
        if low in ("/reload", "/refresh"):
            _reload(cfg); continue
        if low in ("/model", "/models"):
            _dispatch(["model"]); continue

        try:
            argv = shlex.split(line)
        except ValueError as exc:
            console.print(f"[red]parse error:[/red] {exc}")
            continue
        _dispatch(argv)

    console.print(f"[dim]bye — see you at the next lights-out 🏁[/dim]")
