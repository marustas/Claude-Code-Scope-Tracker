"""Core: storage, feature extraction, model, transcript parsing.

Everything in here is deliberately simple. The point of v0 is to find out whether
token cost is predictable from cheap features. If it is, we'll get fancier later.
If not, no amount of fancy fixes it.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

# --- Paths & config ---

DATA_DIR = Path(os.environ.get("SCOPE_TRACKER_HOME", Path.home() / ".scope-tracker"))
DB_PATH = DATA_DIR / "sessions.db"
MODEL_PATH = DATA_DIR / "model.pkl"

MIN_TASKS_FOR_PREDICTION = 20  # stay silent until we have this many labeled tasks
WARN_ABSOLUTE_FLOOR = 50_000   # never warn below this many tokens, regardless of history
WARN_MULTIPLIER = 2.0          # warn if predicted > this * historical_mean

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    cwd TEXT,
    prompt TEXT,
    prompt_features TEXT,
    repo_features TEXT,
    predicted_tokens INTEGER,
    actual_input_tokens INTEGER,
    actual_output_tokens INTEGER,
    actual_total_tokens INTEGER,
    cache_read_tokens INTEGER,
    tool_calls INTEGER DEFAULT 0,
    duration_seconds REAL,
    completed INTEGER DEFAULT 0,
    transcript_start_line INTEGER,
    transcript_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_completed ON tasks(completed);
"""


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def db() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


# --- Feature extraction ---

_VERB_TIERS = {
    2: [  # high-cost verbs
        "implement", "build", "create", "refactor", "design", "architect",
        "migrate", "rewrite", "port", "scaffold", "generate",
    ],
    1: [  # medium
        "fix", "add", "change", "update", "modify", "rename", "check",
        "test", "debug", "investigate", "review",
    ],
    0: [  # low (default — questions, lookups)
        "explain", "what", "why", "show", "list", "describe", "summarize",
        "find", "search",
    ],
}


def _has_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def extract_prompt_features(prompt: str) -> dict[str, Any]:
    """Cheap, deterministic features from the user's prompt text."""
    prompt = prompt or ""
    lower = prompt.lower()

    verb_tier = 0
    for tier in (2, 1):
        if any(_has_word(lower, w) for w in _VERB_TIERS[tier]):
            verb_tier = tier
            break

    return {
        "length": len(prompt),
        "word_count": len(prompt.split()),
        "has_code_block": "```" in prompt,
        "has_file_path": bool(re.search(r"[\w./-]+\.[a-zA-Z0-9]{1,5}\b", prompt)),
        "mentions_test": _has_word(lower, "test") or _has_word(lower, "spec"),
        "mentions_refactor": _has_word(lower, "refactor"),
        "verb_tier": verb_tier,
        "question_count": prompt.count("?"),
        "mentions_all": _has_word(lower, "all") or _has_word(lower, "entire") or _has_word(lower, "every"),
    }


def extract_repo_features(cwd: str) -> dict[str, Any]:
    """Cheap repo metadata. Time-bounded so we don't stall the hook."""
    if not cwd:
        return {"file_count": 0, "language": "unknown"}
    cwd_path = Path(cwd)
    if not cwd_path.exists() or not cwd_path.is_dir():
        return {"file_count": 0, "language": "unknown"}

    files: list[str] = []
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            files = [f for f in result.stdout.splitlines() if f][:10_000]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        files = []

    if not files:
        # Non-git directory: shallow walk, cap quickly.
        try:
            for i, p in enumerate(cwd_path.rglob("*")):
                if i >= 5_000:
                    break
                if p.is_file() and not any(part.startswith(".") for part in p.parts):
                    files.append(str(p))
        except OSError:
            pass

    ext_counts: dict[str, int] = {}
    for f in files:
        ext = Path(f).suffix.lower().lstrip(".")
        if ext:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    dominant = max(ext_counts.items(), key=lambda x: x[1])[0] if ext_counts else "unknown"

    return {
        "file_count": len(files),
        "language": dominant,
    }


# Fixed feature order — KEEP STABLE or invalidate model.
FEATURE_NAMES = (
    "length",
    "word_count",
    "has_code_block",
    "has_file_path",
    "mentions_test",
    "mentions_refactor",
    "verb_tier",
    "question_count",
    "mentions_all",
    "repo_file_count",
)


def features_to_vector(pf: dict, rf: dict) -> list[float]:
    return [
        float(pf.get("length", 0)),
        float(pf.get("word_count", 0)),
        float(int(pf.get("has_code_block", False))),
        float(int(pf.get("has_file_path", False))),
        float(int(pf.get("mentions_test", False))),
        float(int(pf.get("mentions_refactor", False))),
        float(pf.get("verb_tier", 0)),
        float(pf.get("question_count", 0)),
        float(int(pf.get("mentions_all", False))),
        float(rf.get("file_count", 0)),
    ]


# --- Model ---

def _load_training_data() -> tuple[list[list[float]], list[float]]:
    conn = db()
    rows = conn.execute(
        """SELECT prompt_features, repo_features, actual_total_tokens
           FROM tasks
           WHERE completed = 1
             AND actual_total_tokens IS NOT NULL
             AND actual_total_tokens > 0"""
    ).fetchall()
    conn.close()

    X: list[list[float]] = []
    y: list[float] = []
    for pf_json, rf_json, tokens in rows:
        try:
            pf = json.loads(pf_json or "{}")
            rf = json.loads(rf_json or "{}")
        except json.JSONDecodeError:
            continue
        X.append(features_to_vector(pf, rf))
        y.append(float(tokens))
    return X, y


def train_model() -> Any | None:
    X, y = _load_training_data()
    if len(X) < MIN_TASKS_FOR_PREDICTION:
        return None
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return None
    model = GradientBoostingRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        random_state=42,
    )
    model.fit(X, y)
    return model


def save_model(model: Any) -> None:
    if model is None:
        return
    ensure_dirs()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def load_model() -> Any | None:
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except (pickle.PickleError, OSError, EOFError):
        return None


def historical_stats() -> dict[str, Any]:
    conn = db()
    row = conn.execute(
        """SELECT COUNT(*), AVG(actual_total_tokens),
                  MIN(actual_total_tokens), MAX(actual_total_tokens)
           FROM tasks
           WHERE completed = 1 AND actual_total_tokens > 0"""
    ).fetchone()
    conn.close()
    n, mean, mn, mx = row or (0, 0, 0, 0)
    return {
        "n": int(n or 0),
        "mean": int(mean) if mean else 0,
        "min": int(mn) if mn else 0,
        "max": int(mx) if mx else 0,
    }


def predict(prompt: str, cwd: str) -> dict[str, Any] | None:
    """Predict total tokens for a task. Returns None if no model yet."""
    model = load_model()
    if model is None:
        return None
    pf = extract_prompt_features(prompt)
    rf = extract_repo_features(cwd)
    vec = features_to_vector(pf, rf)
    try:
        pred = float(model.predict([vec])[0])
    except Exception:
        return None
    return {
        "predicted_tokens": max(0, int(pred)),
        "stats": historical_stats(),
        "prompt_features": pf,
        "repo_features": rf,
    }


# --- Transcript parsing ---

def count_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def parse_usage(path: str, from_line: int = 0) -> dict[str, int]:
    """Sum usage fields from Claude Code transcript JSONL, starting at from_line.

    The transcript format records assistant messages with a `usage` block
    containing input_tokens/output_tokens/cache_*_tokens. Tool uses appear as
    content blocks of type 'tool_use'. We're permissive about exact schema
    because it can shift between Claude Code versions.
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "tool_calls": 0,
        "total_tokens": 0,
    }
    if not path or not os.path.exists(path):
        return totals

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i < from_line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = rec.get("message", rec)
                usage = msg.get("usage") or rec.get("usage") or {}
                if isinstance(usage, dict):
                    for k in ("input_tokens", "output_tokens",
                              "cache_creation_input_tokens",
                              "cache_read_input_tokens"):
                        v = usage.get(k)
                        if isinstance(v, int):
                            totals[k] += v

                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            totals["tool_calls"] += 1
    except OSError:
        return totals

    # "Total" for context-limit purposes: input + output + cache creation.
    # Cache reads count toward context window but not toward "new" tokens generated.
    totals["total_tokens"] = (
        totals["input_tokens"]
        + totals["output_tokens"]
        + totals["cache_creation_input_tokens"]
    )
    return totals
