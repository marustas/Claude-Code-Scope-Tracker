"""scope-tracker CLI.

Commands:
    scope-tracker install     # add hooks to ~/.claude/settings.json
    scope-tracker uninstall   # remove them
    scope-tracker stats       # show task history & model state
    scope-tracker reset       # wipe local data
    scope-tracker hook <evt>  # internal: dispatch a hook event (reads stdin)
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

from .. import core, hooks

DEFAULT_SETTINGS = Path.home() / ".claude" / "settings.json"
HOOK_TAG = "scope_tracker"  # used to identify our hooks in settings


def _hook_command(event: str) -> str:
    return f"{sys.executable} -m scope_tracker.cli hook {event}"


def _is_our_hook(h: dict) -> bool:
    return HOOK_TAG in (h.get("command") or "")


def cmd_install(args: argparse.Namespace) -> int:
    settings_path = Path(args.settings) if args.settings else DEFAULT_SETTINGS
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            print(f"error: {settings_path} contains invalid JSON. Refusing to overwrite.",
                  file=sys.stderr)
            return 1

    settings.setdefault("hooks", {})
    additions = {
        "UserPromptSubmit": {
            "hooks": [{"type": "command", "command": _hook_command("on-prompt"),
                       "timeout": 10}],
        },
        "PostToolUse": {
            "matcher": ".*",
            "hooks": [{"type": "command", "command": _hook_command("on-tool"),
                       "timeout": 5}],
        },
        "Stop": {
            "hooks": [{"type": "command", "command": _hook_command("on-stop"),
                       "timeout": 30}],
        },
    }

    added: list[str] = []
    for event, config in additions.items():
        existing = settings["hooks"].setdefault(event, [])
        already = any(_is_our_hook(h) for group in existing for h in group.get("hooks", []))
        if not already:
            existing.append(config)
            added.append(event)

    # Backup before write
    if settings_path.exists():
        backup = settings_path.with_suffix(".json.scope-tracker-bak")
        shutil.copy2(settings_path, backup)
        print(f"backed up existing settings to {backup}")

    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"installed hooks in {settings_path}")
    if added:
        print(f"  added events: {', '.join(added)}")
    else:
        print("  (all hooks already present — nothing to do)")
    print(f"data dir: {core.DATA_DIR}")
    print(f"predictions activate after {core.MIN_TASKS_FOR_PREDICTION} completed tasks.")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    settings_path = Path(args.settings) if args.settings else DEFAULT_SETTINGS
    if not settings_path.exists():
        print(f"no settings file at {settings_path}, nothing to uninstall.")
        return 0

    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError:
        print(f"error: {settings_path} is invalid JSON.", file=sys.stderr)
        return 1

    removed = 0
    for event, groups in list(settings.get("hooks", {}).items()):
        for group in groups:
            before = len(group.get("hooks", []))
            group["hooks"] = [h for h in group.get("hooks", []) if not _is_our_hook(h)]
            removed += before - len(group["hooks"])
        # Drop empty groups
        settings["hooks"][event] = [g for g in groups if g.get("hooks")]
        if not settings["hooks"][event]:
            del settings["hooks"][event]

    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"removed {removed} scope-tracker hook entr{'y' if removed == 1 else 'ies'} from {settings_path}")
    print("(local data in ~/.scope-tracker/ was NOT deleted — run `scope-tracker reset` to wipe it)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    conn = core.db()
    total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    stats = core.historical_stats()

    print(f"data dir:     {core.DATA_DIR}")
    print(f"model:        {'trained' if core.MODEL_PATH.exists() else 'not yet (need ' + str(core.MIN_TASKS_FOR_PREDICTION) + ' completed tasks)'}")
    print(f"tasks logged: {total}")
    print(f"  completed:  {stats['n']}")

    if stats["n"] == 0:
        conn.close()
        return 0

    print()
    print(f"token usage across completed tasks:")
    print(f"  mean: {stats['mean']:>9,}")
    print(f"  min:  {stats['min']:>9,}")
    print(f"  max:  {stats['max']:>9,}")

    print()
    print("last 10 tasks:")
    print(f"  {'when':<12} {'actual':>9} {'pred':>9} {'err':>6}  prompt")
    rows = conn.execute(
        """SELECT started_at, actual_total_tokens, predicted_tokens, prompt
           FROM tasks
           WHERE completed = 1
           ORDER BY started_at DESC LIMIT 10"""
    ).fetchall()
    for started, actual, predicted, prompt in rows:
        ts = datetime.datetime.fromtimestamp(started).strftime("%m-%d %H:%M")
        actual_s = f"{actual:,}" if actual else "—"
        pred_s = f"{predicted:,}" if predicted else "—"
        err_s = "—"
        if actual and predicted:
            err_pct = (predicted - actual) / actual * 100
            err_s = f"{err_pct:+.0f}%"
        snippet = (prompt or "").replace("\n", " ")[:60]
        print(f"  {ts:<12} {actual_s:>9} {pred_s:>9} {err_s:>6}  {snippet}")
    conn.close()
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    if not core.DATA_DIR.exists():
        print(f"no data at {core.DATA_DIR}, nothing to reset.")
        return 0
    if not args.force:
        confirm = input(f"delete everything in {core.DATA_DIR}? [y/N] ")
        if confirm.strip().lower() != "y":
            print("aborted.")
            return 0
    shutil.rmtree(core.DATA_DIR)
    print(f"wiped {core.DATA_DIR}")
    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    handlers = {
        "on-prompt": hooks.on_prompt_submit,
        "on-tool": hooks.on_tool_use,
        "on-stop": hooks.on_stop,
    }
    handler = handlers.get(args.event)
    if not handler:
        print(f"unknown hook event: {args.event}", file=sys.stderr)
        return 1
    try:
        return handler() or 0
    except Exception as e:  # noqa: BLE001 — must never crash Claude Code
        print(f"[scope-tracker] hook error ({args.event}): {e}", file=sys.stderr)
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scope-tracker",
        description="Learn token costs from your Claude Code sessions.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_install = sub.add_parser("install", help="add hooks to Claude Code settings")
    p_install.add_argument("--settings", help=f"path to settings.json (default: {DEFAULT_SETTINGS})")
    p_install.set_defaults(func=cmd_install)

    p_uninstall = sub.add_parser("uninstall", help="remove scope-tracker hooks from Claude Code settings")
    p_uninstall.add_argument("--settings", help=f"path to settings.json (default: {DEFAULT_SETTINGS})")
    p_uninstall.set_defaults(func=cmd_uninstall)

    p_stats = sub.add_parser("stats", help="show task history and model state")
    p_stats.set_defaults(func=cmd_stats)

    p_reset = sub.add_parser("reset", help="wipe all local data")
    p_reset.add_argument("--force", action="store_true")
    p_reset.set_defaults(func=cmd_reset)

    p_hook = sub.add_parser("hook", help="internal: dispatch a hook event (reads stdin)")
    p_hook.add_argument("event", choices=["on-prompt", "on-tool", "on-stop"])
    p_hook.set_defaults(func=cmd_hook)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
