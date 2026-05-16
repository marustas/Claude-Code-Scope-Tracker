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

from .. import core


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


def _format_warning(predicted: int, stats: dict) -> str:
    return (
        f"[scope-tracker — notice for the user]: based on {stats['n']} of your past "
        f"Claude Code tasks, this one is estimated at ~{predicted:,} tokens "
        f"(your historical mean is {stats['mean']:,}, max {stats['max']:,}). "
        f"This is in the upper range of your usage and tasks like this sometimes hit "
        f"context limits mid-execution. You may want to scope this down or split it "
        f"into smaller chunks before proceeding. (This message is from a local hook, "
        f"not from Claude. Acknowledge it briefly to the user and offer to scope down "
        f"if appropriate, then proceed.)"
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

    conn = core.db()
    task_id = _next_task_id(conn, session_id)
    conn.execute(
        """INSERT INTO tasks (
               task_id, session_id, started_at, cwd, prompt,
               prompt_features, repo_features, predicted_tokens,
               transcript_start_line, transcript_path
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            task_id,
            session_id,
            time.time(),
            cwd,
            prompt,
            json.dumps(pf),
            json.dumps(rf),
            predicted_tokens,
            core.count_lines(transcript_path),
            transcript_path,
        ),
    )
    conn.commit()
    conn.close()

    # Only surface a warning when we actually have a trained model AND prediction
    # is meaningfully high. Stay silent otherwise.
    if pred and pred["stats"]["n"] >= core.MIN_TASKS_FOR_PREDICTION:
        threshold = max(core.WARN_ABSOLUTE_FLOOR,
                        int(pred["stats"]["mean"] * core.WARN_MULTIPLIER))
        if predicted_tokens and predicted_tokens > threshold:
            print(_format_warning(predicted_tokens, pred["stats"]))

    return 0


def on_tool_use() -> int:
    """PostToolUse: increment tool counter for the current open task."""
    data = _read_stdin_json()
    session_id = data.get("session_id") or "unknown"

    conn = core.db()
    conn.execute(
        """UPDATE tasks SET tool_calls = tool_calls + 1
           WHERE task_id = (
               SELECT task_id FROM tasks
               WHERE session_id = ? AND completed = 0
               ORDER BY started_at DESC LIMIT 1
           )""",
        (session_id,),
    )
    conn.commit()
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

    return 0
