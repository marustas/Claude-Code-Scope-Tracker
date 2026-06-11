"""Hook entry points called by Claude Code.

Each handler reads a JSON object from stdin and exits 0. Anything printed to
stdout from UserPromptSubmit gets added to Claude's context — that's how the
non-blocking warning surfaces to the user.

Hooks must NEVER crash Claude Code, so every handler swallows its own errors.
"""
from __future__ import annotations

import json
import sys
import time

from . import core


def _read_stdin_json() -> dict:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return {}


def _next_task_id(conn, session_id: str) -> str:
    row = conn.execute(
        """SELECT MAX(CAST(SUBSTR(task_id, INSTR(task_id, ':') + 1) AS INTEGER))
           FROM tasks WHERE session_id = ?""",
        (session_id,),
    ).fetchone()
    next_turn = (row[0] or 0) + 1
    return f"{session_id}:{next_turn}"


def _format_warning(predicted: int, predicted_p90: int, stats: dict) -> str:
    return (
        f"[scope-tracker — notice for the user]: based on {stats['n']} of your past "
        f"Claude Code tasks, this one is estimated at ~{predicted:,} tokens, and could "
        f"reach ~{predicted_p90:,} (90th percentile) "
        f"(your historical mean is {stats['mean']:,}, max {stats['max']:,}). "
        f"This is in the upper range of your usage and tasks like this sometimes hit "
        f"context limits mid-execution. You may want to scope this down or split it "
        f"into smaller chunks before proceeding. (This message is from a local hook, "
        f"not from Claude. Acknowledge it briefly to the user and offer to scope down "
        f"if appropriate, then proceed.)"
    )


def _fmt_hours(seconds: float) -> str:
    h = seconds / 3600
    if h >= 1:
        return f"~{h:.1f}h"
    return f"~{int(seconds / 60)}min"


def _format_session_warning(used: int, projected_task: int, budget: int,
                            relief_seconds: float, window_hours: float) -> str:
    projected_total = used + projected_task
    pct = projected_total / budget * 100 if budget else 0
    return (
        f"[scope-tracker — session-budget notice for the user]: in the last "
        f"{window_hours:.0f}h you've used ~{used:,} tokens of your ~{budget:,} session "
        f"limit. This task is estimated to add ~{projected_task:,}, which would put you "
        f"at ~{projected_total:,} ({pct:.0f}% of the limit). You risk hitting the "
        f"resets-every-few-hours cap mid-task; the window starts freeing up in "
        f"{_fmt_hours(relief_seconds)}. Consider a smaller task now or waiting for the "
        f"reset. (This is from a local hook, not from Claude. Mention it briefly to the "
        f"user, then proceed.)"
    )


def on_prompt_submit() -> int:
    """UserPromptSubmit: record task start, predict, optionally warn."""
    data = _read_stdin_json()
    session_id = data.get("session_id") or "unknown"
    cwd = data.get("cwd") or ""
    prompt = data.get("prompt") or ""
    transcript_path = data.get("transcript_path") or ""

    if not prompt:
        return 0

    pf = core.extract_prompt_features(prompt)
    rf = core.extract_repo_features(cwd)

    pred = core.predict(prompt, cwd)
    predicted_tokens = pred["predicted_tokens"] if pred else None
    predicted_p90 = pred["predicted_p90"] if pred else None

    conn = core.db()
    task_id = _next_task_id(conn, session_id)
    conn.execute(
        """INSERT INTO tasks (
               task_id, session_id, started_at, cwd, prompt,
               prompt_features, repo_features, predicted_tokens, predicted_p90,
               transcript_start_line, transcript_path
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            task_id,
            session_id,
            time.time(),
            cwd,
            prompt,
            json.dumps(pf),
            json.dumps(rf),
            predicted_tokens,
            predicted_p90,
            core.count_lines(transcript_path),
            transcript_path,
        ),
    )
    conn.commit()
    conn.close()

    # Only surface a warning when we actually have a trained model AND the upper
    # bound (p90) is meaningfully high. Warning on p90 rather than the point
    # estimate matches the tool's job: avoid blowing the context limit mid-task.
    if pred and pred["stats"]["n"] >= core.MIN_TASKS_FOR_PREDICTION:
        threshold = max(core.WARN_ABSOLUTE_FLOOR,
                        int(pred["stats"]["mean"] * core.WARN_MULTIPLIER))
        if predicted_p90 and predicted_p90 > threshold:
            print(_format_warning(predicted_tokens, predicted_p90, pred["stats"]))

    # Session-budget warning: would this task push rolling-window usage over the
    # quota that resets every few hours? Independent of the per-task model — fires
    # on consumed-so-far plus the best available estimate for this task.
    if core.SESSION_BUDGET > 0:
        try:
            su = core.session_window_usage()
            stats = pred["stats"] if pred else core.historical_stats()
            projected_task = predicted_p90 or predicted_tokens or stats.get("mean", 0)
            if su["used"] + projected_task > core.SESSION_BUDGET * core.SESSION_WARN_FRACTION:
                print(_format_session_warning(
                    su["used"], int(projected_task), core.SESSION_BUDGET,
                    su["relief_in_seconds"], su["window_hours"]))
        except Exception:  # noqa: BLE001 — must never crash the hook
            pass

    return 0


def _format_midtask_warning(tokens_so_far: int, projected: int, projected_p90: int,
                            tier: str, tools: int, stats: dict) -> str:
    return (
        f"[scope-tracker — mid-task notice for the user]: this is shaping up as a "
        f"{tier}-size task. It has already used ~{tokens_so_far:,} tokens across {tools} "
        f"tool calls and is now projected to finish around ~{projected:,} tokens, up to "
        f"~{projected_p90:,} (90th percentile) — above your usual range (mean "
        f"{stats['mean']:,}). Tasks like this sometimes hit context limits before "
        f"completing. Consider wrapping up the current step, committing partial progress, "
        f"or narrowing the remaining work. (This is from a local hook, not from Claude. "
        f"Mention it briefly to the user and offer to scope down the rest, then continue.)"
    )


def on_tool_use() -> int:
    """PostToolUse: count the tool call, and once signal is solid, re-project the
    final cost from what's been spent so far — warning once if it looks high."""
    data = _read_stdin_json()
    session_id = data.get("session_id") or "unknown"

    conn = core.db()
    row = conn.execute(
        """SELECT task_id, tool_calls, prompt_features, repo_features,
                  transcript_path, transcript_start_line, midtask_warned
           FROM tasks
           WHERE session_id = ? AND completed = 0
           ORDER BY started_at DESC LIMIT 1""",
        (session_id,),
    ).fetchone()
    if not row:
        conn.close()
        return 0

    task_id, tool_calls, pf_json, rf_json, tpath, start_line, warned = row
    tool_calls = (tool_calls or 0) + 1
    conn.execute("UPDATE tasks SET tool_calls = ? WHERE task_id = ?", (tool_calls, task_id))
    conn.commit()

    # Mid-task re-estimate. Guarded separately so a prediction error can never
    # undo the counter update above or crash the hook.
    try:
        if (not warned) and tool_calls >= core.MIDTASK_MIN_TOOL_CALLS:
            stats = core.historical_stats()
            if stats["n"] >= core.MIN_TASKS_FOR_PREDICTION:
                pf = json.loads(pf_json or "{}")
                rf = json.loads(rf_json or "{}")
                usage = core.parse_usage(tpath, from_line=start_line or 0)
                tokens_so_far = usage["total_tokens"]
                proj = core.predict_midtask(pf, rf, tool_calls, tokens_so_far,
                                            usage.get("tool_counts"))
                if proj:
                    threshold = max(core.WARN_ABSOLUTE_FLOOR,
                                    int(stats["mean"] * core.WARN_MULTIPLIER))
                    if proj["predicted_p90"] > threshold:
                        warn = _format_midtask_warning(
                            tokens_so_far, proj["predicted_tokens"],
                            proj["predicted_p90"], proj["tier"], tool_calls, stats)
                        print(json.dumps({"hookSpecificOutput": {
                            "hookEventName": "PostToolUse",
                            "additionalContext": warn,
                        }}))
                        conn.execute(
                            "UPDATE tasks SET midtask_warned = 1 WHERE task_id = ?",
                            (task_id,))
                        conn.commit()
    except Exception:  # noqa: BLE001 — mid-task estimate must never break counting
        pass

    conn.close()
    return 0


def on_stop() -> int:
    """Stop: read transcript, compute actual usage, finalize task, maybe retrain."""
    data = _read_stdin_json()
    session_id = data.get("session_id") or "unknown"
    transcript_path = data.get("transcript_path") or ""

    conn = core.db()
    row = conn.execute(
        """SELECT task_id, started_at, transcript_start_line
           FROM tasks
           WHERE session_id = ? AND completed = 0
           ORDER BY started_at DESC LIMIT 1""",
        (session_id,),
    ).fetchone()

    if not row:
        conn.close()
        return 0

    task_id, started_at, start_line = row
    usage = core.parse_usage(transcript_path, from_line=start_line or 0)
    now = time.time()

    conn.execute(
        """UPDATE tasks SET
               ended_at = ?,
               completed = 1,
               actual_input_tokens = ?,
               actual_output_tokens = ?,
               actual_total_tokens = ?,
               cache_read_tokens = ?,
               duration_seconds = ?,
               tool_calls = MAX(tool_calls, ?)
           WHERE task_id = ?""",
        (
            now,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
            usage["cache_read_input_tokens"],
            now - (started_at or now),
            usage["tool_calls"],
            task_id,
        ),
    )
    conn.commit()

    completed_count = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE completed = 1 AND actual_total_tokens > 0"
    ).fetchone()[0]
    conn.close()

    # Retrain on milestones, then every 25 tasks. Cheap with our small models.
    if completed_count >= core.MIN_TASKS_FOR_PREDICTION:
        if completed_count in (20, 30, 50) or completed_count % 25 == 0:
            model = core.train_model()
            if model is not None:
                core.save_model(model)
            midtask = core.train_midtask_model()
            if midtask is not None:
                core.save_midtask_model(midtask)

    return 0
