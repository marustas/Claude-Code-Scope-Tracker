"""Core: storage, feature extraction, model, transcript parsing.

Everything in here is deliberately simple. The point of v0 is to find out whether
token cost is predictable from cheap features. If it is, we'll get fancier later.
If not, no amount of fancy fixes it.
"""
from __future__ import annotations

import json
import math
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
MIDTASK_MODEL_PATH = DATA_DIR / "model_midtask.pkl"

MIN_TASKS_FOR_PREDICTION = 20  # stay silent until we have this many labeled tasks
WARN_ABSOLUTE_FLOOR = 50_000   # never warn below this many tokens, regardless of history
WARN_MULTIPLIER = 2.0          # warn if predicted > this * historical_mean


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# Rolling session quota (the usage limit that resets every few hours). Usage is
# bursty, so a task can run the window dry mid-execution.
#
# The window is 5h on every plan, so it's fixed. The budget derives from the
# subscription tier — Pro is the base, the Max tiers scale 5x / 20x. These token
# figures are estimates (Anthropic doesn't publish the cap as a clean number);
# tune with SCOPE_TRACKER_SESSION_BUDGET, which overrides the tier default.
# Set SCOPE_TRACKER_SESSION_BUDGET=0 to turn the warning off entirely.
SESSION_WINDOW_HOURS = 5.0  # fixed across all plans

PLAN_BUDGETS = {
    "pro": 2_000_000,
    "max5x": 10_000_000,
    "max20x": 40_000_000,
}
DEFAULT_PLAN = "pro"
_PLAN = os.environ.get("SCOPE_TRACKER_PLAN", DEFAULT_PLAN).strip().lower()
SESSION_BUDGET = _env_int(
    "SCOPE_TRACKER_SESSION_BUDGET",
    PLAN_BUDGETS.get(_PLAN, PLAN_BUDGETS[DEFAULT_PLAN]),
)
SESSION_WARN_FRACTION = _env_float("SCOPE_TRACKER_SESSION_WARN_FRACTION", 0.8)

# Mid-task re-estimate (fix D): once a task is underway, observed cost-so-far is a
# far stronger signal than the prompt. We re-project the final total from it.
MIDTASK_MIN_TOOL_CALLS = 3       # don't re-estimate until there's real signal
MIN_MIDTASK_ROWS = 50            # min replayed checkpoints before the model trains

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
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column additions. CREATE TABLE IF NOT EXISTS won't add columns
    to a table that already exists, so new columns are added here."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "predicted_p90" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN predicted_p90 INTEGER")
        conn.commit()
    if "midtask_warned" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN midtask_warned INTEGER DEFAULT 0")
        conn.commit()


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

# Irregular forms our suffix rules can't derive. Keyed by base verb.
_IRREGULAR = {
    "build": ("built",),
    "rewrite": ("rewrote", "rewritten"),
    "write": ("wrote", "written"),
    "find": ("found",),
}


def _inflections(verb: str) -> set[str]:
    """Lightweight inflection generation (no NLP dependency).

    Covers the regular cases the verb lists actually need: -s/-ed/-ing,
    final-e drop (migrate→migrating), and y→ies. Irregulars come from a table.
    Not a full lemmatizer — just enough that 'implementing' matches 'implement'.
    """
    forms = {verb, verb + "s", verb + "ed", verb + "ing", verb + "d"}
    if verb.endswith("e"):
        forms |= {verb[:-1] + "ing", verb[:-1] + "ed", verb[:-1] + "es"}
    if verb.endswith("y") and len(verb) > 2 and verb[-2] not in "aeiou":
        forms |= {verb[:-1] + "ies", verb[:-1] + "ied"}
    forms |= set(_IRREGULAR.get(verb, ()))
    return forms


def _compile_words(words: set[str]) -> "re.Pattern[str]":
    longest_first = sorted(words, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in longest_first) + r")\b",
                      flags=re.IGNORECASE)


def _inflected_set(*verbs: str) -> set[str]:
    out: set[str] = set()
    for v in verbs:
        out |= _inflections(v)
    return out


# Precompiled, inflection-aware matchers (built once at import).
_VERB_TIER_RE = {
    tier: _compile_words(_inflected_set(*words)) for tier, words in _VERB_TIERS.items()
}
_TEST_RE = _compile_words(_inflected_set("test", "spec"))
_REFACTOR_RE = _compile_words(_inflections("refactor"))
_ALL_RE = _compile_words({"all", "entire", "every", "everything", "whole"})

# Context blocks Claude Code injects into the prompt. They inflate length/word
# counts with text unrelated to task scope, so we strip them before extraction.
_INJECTED_TAGS = ("ide_opened_file", "ide_selection", "system-reminder")
_INJECTED_PAIRED_RE = re.compile(
    r"<(" + "|".join(_INJECTED_TAGS) + r")>.*?</\1>", flags=re.DOTALL | re.IGNORECASE
)
_INJECTED_STRAY_RE = re.compile(
    r"</?(?:" + "|".join(_INJECTED_TAGS) + r")>", flags=re.IGNORECASE
)


def strip_injected_context(prompt: str) -> str:
    """Remove IDE/system context blocks Claude Code prepends to the user prompt."""
    if not prompt:
        return ""
    cleaned = _INJECTED_PAIRED_RE.sub(" ", prompt)
    cleaned = _INJECTED_STRAY_RE.sub(" ", cleaned)
    return cleaned.strip()


def extract_prompt_features(prompt: str) -> dict[str, Any]:
    """Cheap, deterministic features from the user's prompt text."""
    prompt = strip_injected_context(prompt or "")

    verb_tier = 0
    for tier in (2, 1):
        if _VERB_TIER_RE[tier].search(prompt):
            verb_tier = tier
            break

    return {
        "length": len(prompt),
        "word_count": len(prompt.split()),
        "has_code_block": "```" in prompt,
        "has_file_path": bool(re.search(r"[\w./-]+\.[a-zA-Z0-9]{1,5}\b", prompt)),
        "mentions_test": bool(_TEST_RE.search(prompt)),
        "mentions_refactor": bool(_REFACTOR_RE.search(prompt)),
        "verb_tier": verb_tier,
        "question_count": prompt.count("?"),
        "mentions_all": bool(_ALL_RE.search(prompt)),
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


# --- Rolling session quota ---

def session_window_usage(at_time: float | None = None,
                         window_hours: float | None = None) -> dict[str, Any]:
    """Tokens consumed in the current rolling window (default the last 5 hours).

    Models Claude's resets-every-few-hours usage limit: sum the total_tokens of
    completed tasks whose start falls inside [now - window, now]. Also returns when
    the oldest in-window task ages out, i.e. roughly when quota starts freeing up.
    """
    window = (window_hours if window_hours is not None else SESSION_WINDOW_HOURS) * 3600
    now = at_time if at_time is not None else time.time()
    conn = db()
    rows = conn.execute(
        """SELECT started_at, actual_total_tokens FROM tasks
           WHERE completed = 1 AND actual_total_tokens > 0
             AND started_at >= ? AND started_at <= ?""",
        (now - window, now),
    ).fetchall()
    conn.close()
    used = sum(int(tok) for _, tok in rows if tok)
    oldest = min((ts for ts, _ in rows), default=None)
    # When the oldest in-window task leaves the window, that much quota frees up.
    relief_in = (oldest + window - now) if oldest is not None else 0.0
    return {
        "used": used,
        "window_hours": window / 3600,
        "tasks_in_window": len(rows),
        "relief_in_seconds": max(0.0, relief_in),
    }


# --- Token tiers ("token poker") ---
#
# Exact counts aren't predictable from cheap signals, but the coarse *tier* is —
# especially mid-task (measured ~49% exact / ~92% within-one-tier, vs ~39% for a
# always-Medium baseline). Tiers also frame the estimate honestly: a relative size
# (like story points) rather than a false-precision number.
TIERS: tuple[tuple[str, int, float], ...] = (
    ("XS", 0,        5_000),
    ("S",  5_000,    20_000),
    ("M",  20_000,   75_000),
    ("L",  75_000,   200_000),
    ("XL", 200_000,  math.inf),
)
TIER_NAMES = tuple(name for name, _, _ in TIERS)


def tier_of(tokens: float) -> str:
    """Map a token count to its size tier (XS/S/M/L/XL)."""
    for name, lo, hi in TIERS:
        if lo <= tokens < hi:
            return name
    return TIERS[-1][0]


def tier_range(name: str) -> tuple[int, float]:
    for n, lo, hi in TIERS:
        if n == name:
            return lo, hi
    return 0, math.inf


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


# Quantiles we fit. p50 is the point estimate; p90 drives the warning.
QUANTILES = (0.5, 0.9)
P90_CORR_KEY = "p90_log_correction"  # conformal adjustment, in log space


def _make_gbr(alpha: float):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        loss="quantile", alpha=alpha, n_estimators=200,
        max_depth=3, learning_rate=0.05, random_state=42,
    )


CONFORMAL_CALIB_FRAC = 0.2     # share of groups held out to calibrate the p90 bound
MIN_CONFORMAL_CALIB_ROWS = 150  # below this, the 90% score quantile is too noisy


def _fit_quantile_bundle(X: list, y: list, groups: list | None = None,
                         level: float = 0.9) -> Any | None:
    """Fit p50/p90 on log1p(y), with a split-conformal p90 correction when there's
    enough calibration data to estimate it reliably.

    Split conformal: train the quantile models on most of the data, hold out a
    calibration slice, and shift the p90 bound by the finite-sample conformal
    quantile of (y - predicted_p90) on that slice. Calibrating the *same* model
    that ships gives a valid >= `level` coverage guarantee. Groups (a task id per
    row) keep a task's checkpoints from spanning the train/calibration boundary.

    With a small calibration set the 90% score quantile is essentially the max and
    the correction overshoots — so below MIN_CONFORMAL_CALIB_ROWS we skip it, fit
    on all data, and keep the nominal (uncorrected) p90. In practice conformal
    engages for the data-rich mid-task model and the submit-time prior stays
    nominal rather than burning its scarce data on a noisy correction.
    """
    try:
        import sklearn.ensemble  # noqa: F401
    except ImportError:
        return None
    y_log = [math.log1p(v) for v in y]
    n = len(X)
    if groups is None:
        groups = list(range(n))
    uniq = list(dict.fromkeys(groups))
    import random as _random
    _random.Random(42).shuffle(uniq)
    n_cal = int(round(len(uniq) * CONFORMAL_CALIB_FRAC))

    # The p50 point estimate always trains on ALL data — it drives within-2x and
    # needs no calibration. Only the p90 bound uses the split.
    bundle: dict[Any, Any] = {0.5: _make_gbr(0.5).fit(X, y_log)}

    calib = set(uniq[:n_cal]) if n_cal >= 1 else set()
    ca = [i for i in range(n) if groups[i] in calib]
    if len(ca) < MIN_CONFORMAL_CALIB_ROWS or len(uniq) - n_cal < 2:
        # Not enough to calibrate reliably — fit p90 on all data, no correction.
        bundle[0.9] = _make_gbr(0.9).fit(X, y_log)
        bundle[P90_CORR_KEY] = 0.0
        return bundle

    tr = [i for i in range(n) if groups[i] not in calib]
    bundle[0.9] = _make_gbr(0.9).fit([X[i] for i in tr], [y_log[i] for i in tr])
    preds = bundle[0.9].predict([X[i] for i in ca])
    scores = sorted(float(y_log[i] - p) for i, p in zip(ca, preds))
    # Smallest c with >= level of (y - p90) <= c. Finite-sample rank, clamped.
    k = min(len(scores), math.ceil((len(scores) + 1) * level))
    bundle[P90_CORR_KEY] = scores[k - 1]
    return bundle


def train_model() -> Any | None:
    """Fit log-space quantile models (p50/p90 + conformal correction) on token cost.

    Token cost spans ~3 orders of magnitude and is heavily right-skewed. Training
    on raw counts with squared error chases the few huge outliers and collapses to
    the mean. Log-space + quantile (pinball) loss respects the multiplicative scale
    and yields an interval; the conformal correction makes the p90 a real coverage
    guarantee rather than a nominal one.

    Returns a dict {0.5: model, 0.9: model, P90_CORR_KEY: float} or None.
    """
    X, y = _load_training_data()
    if len(X) < MIN_TASKS_FOR_PREDICTION:
        return None
    return _fit_quantile_bundle(X, y)


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
    """Predict a token range for a task. Returns None if no usable model yet.

    `predicted_tokens` is the p50 point estimate; `predicted_p90` is the upper
    bound used for warnings. Legacy single-estimator models are treated as no
    model (a retrain on the next milestone produces the new quantile models).
    """
    models = load_model()
    if not isinstance(models, dict):
        return None
    pf = extract_prompt_features(prompt)
    rf = extract_repo_features(cwd)
    vec = features_to_vector(pf, rf)
    try:
        corr = float(models.get(P90_CORR_KEY, 0.0))
        p50 = math.expm1(float(models[0.5].predict([vec])[0]))
        p90 = math.expm1(float(models[0.9].predict([vec])[0]) + corr)
    except Exception:
        return None
    p50 = max(0, int(p50))
    p90 = max(p50, int(p90))  # quantile crossing guard
    return {
        "predicted_tokens": p50,
        "predicted_p90": p90,
        "tier": tier_of(p50),
        "tier_p90": tier_of(p90),
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
        "tool_counts": {},  # name -> count, for tool-type features
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
                            name = block.get("name", "?")
                            totals["tool_counts"][name] = totals["tool_counts"].get(name, 0) + 1
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


# --- Mid-task model (fix D) ---
#
# Once a task is underway, the cost accumulated so far predicts the final total
# far better than the prompt ever could. We re-project from (submit features +
# tool_calls_so_far + tokens_so_far). Trained offline by replaying transcripts;
# at runtime the PostToolUse hook calls predict_midtask once signal is solid.

# Tool-type buckets. *Which* tools have fired is signal the raw count misses:
# code mutation (Edit/Write) and subagent spawns predict larger final cost than
# the same number of Reads. Validated to lift early (3-call) accuracy by ~7 pts.
_TOOL_MUTATE = ("Edit", "Write", "NotebookEdit")
_TOOL_EXEC = ("Bash",)
_TOOL_EXPLORE = ("Read", "Grep", "Glob")
_TOOL_SUBAGENT = ("Task", "TaskCreate")


def tool_type_features(tool_counts: dict | None, tools_so_far: int) -> list[float]:
    """Six features describing the tool-call mix so far. Robust to a missing or
    empty count map (returns all-zero so the vector length stays fixed)."""
    counts = tool_counts or {}
    denom = float(max(1, tools_so_far))
    mutate = sum(counts.get(n, 0) for n in _TOOL_MUTATE)
    return [
        sum(counts.get(n, 0) for n in _TOOL_EXEC) / denom,     # exec fraction
        sum(counts.get(n, 0) for n in _TOOL_EXPLORE) / denom,  # explore fraction
        mutate / denom,                                        # mutate fraction
        float(mutate),                                         # mutate count
        float(len(counts)),                                    # distinct tools used
        1.0 if any(counts.get(n, 0) for n in _TOOL_SUBAGENT) else 0.0,  # subagent spawned
    ]


MIDTASK_FEATURE_NAMES = FEATURE_NAMES + (
    "tool_calls_so_far", "log_tokens_so_far",
    "exec_frac", "explore_frac", "mutate_frac", "mutate_count",
    "distinct_tools", "has_subagent",
)


def midtask_vector(pf: dict, rf: dict, tools_so_far: int, tokens_so_far: float,
                   tool_counts: dict | None = None) -> list[float]:
    return features_to_vector(pf, rf) + [
        float(tools_so_far),
        math.log1p(max(0.0, float(tokens_so_far))),
    ] + tool_type_features(tool_counts, tools_so_far)


def _region_checkpoints(path: str, start: int, end: int | None) -> list[tuple[int, int, dict]]:
    """Replay one task's transcript region, returning a (tool_calls, tokens,
    tool_counts) checkpoint after each assistant message. `end` bounds the region
    (the next task's start line in the same transcript), or None for end-of-file.
    `tool_counts` is a snapshot copy of the cumulative name->count map."""
    cum_tok = 0
    cum_tools = 0
    counts: dict[str, int] = {}
    ckpts: list[tuple[int, int, dict]] = []
    if not path or not os.path.exists(path):
        return ckpts
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i < (start or 0):
                    continue
                if end is not None and i >= end:
                    break
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
                              "cache_creation_input_tokens"):
                        v = usage.get(k)
                        if isinstance(v, int):
                            cum_tok += v
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            cum_tools += 1
                            name = block.get("name", "?")
                            counts[name] = counts.get(name, 0) + 1
                ckpts.append((cum_tools, cum_tok, dict(counts)))
    except OSError:
        return ckpts
    return ckpts


def _midtask_training_rows() -> tuple[list[list[float]], list[float], list[int]]:
    """Build mid-task training data by replaying stored transcripts.

    Emits one row per checkpoint where tool_calls_so_far >= MIDTASK_MIN_TOOL_CALLS,
    pairing the state at that moment with the task's final total. Many checkpoints
    per task makes the model robust to *when* it's asked to predict. Also returns a
    `groups` list (one task id per row) so calibration folds stay task-disjoint.
    """
    conn = db()
    rows = conn.execute(
        """SELECT prompt_features, repo_features, transcript_path,
                  transcript_start_line, actual_total_tokens
           FROM tasks
           WHERE completed = 1 AND actual_total_tokens > 0
             AND transcript_path IS NOT NULL
           ORDER BY transcript_path, transcript_start_line"""
    ).fetchall()
    conn.close()

    # Map each transcript region to the next task's start line in that file.
    starts_by_path: dict[str, list[int]] = {}
    for _, _, path, start, _ in rows:
        if path is not None and start is not None:
            starts_by_path.setdefault(path, []).append(start)
    for path in starts_by_path:
        starts_by_path[path].sort()

    X: list[list[float]] = []
    y: list[float] = []
    groups: list[int] = []
    for task_idx, (pf_json, rf_json, path, start, final) in enumerate(rows):
        try:
            pf = json.loads(pf_json or "{}")
            rf = json.loads(rf_json or "{}")
        except json.JSONDecodeError:
            continue
        starts = starts_by_path.get(path, [])
        end = next((s for s in starts if start is not None and s > start), None)
        ckpts = _region_checkpoints(path, start or 0, end)
        for tools, tokens, counts in ckpts:
            if tools >= MIDTASK_MIN_TOOL_CALLS:
                X.append(midtask_vector(pf, rf, tools, tokens, counts))
                y.append(float(final))
                groups.append(task_idx)
    return X, y, groups


def train_midtask_model() -> Any | None:
    """Fit log-space quantile models (p50/p90 + conformal p90 correction) projecting
    final tokens from mid-task state. None if there isn't enough replayed signal."""
    X, y, groups = _midtask_training_rows()
    if len(X) < MIN_MIDTASK_ROWS:
        return None
    return _fit_quantile_bundle(X, y, groups=groups)


def load_midtask_model() -> Any | None:
    if not MIDTASK_MODEL_PATH.exists():
        return None
    try:
        with open(MIDTASK_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except (pickle.PickleError, OSError, EOFError):
        return None


def save_midtask_model(model: Any) -> None:
    if model is None:
        return
    ensure_dirs()
    with open(MIDTASK_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def predict_midtask(pf: dict, rf: dict, tools_so_far: int, tokens_so_far: float,
                    tool_counts: dict | None = None) -> dict[str, Any] | None:
    """Project final total tokens from observed mid-task state. None if no model."""
    models = load_midtask_model()
    if not isinstance(models, dict):
        return None
    vec = midtask_vector(pf, rf, tools_so_far, tokens_so_far, tool_counts)
    try:
        corr = float(models.get(P90_CORR_KEY, 0.0))
        p50 = math.expm1(float(models[0.5].predict([vec])[0]))
        p90 = math.expm1(float(models[0.9].predict([vec])[0]) + corr)
    except Exception:
        return None
    p50 = max(0, int(p50))
    # Final can't be less than what's already burned, nor p90 below p50.
    p50 = max(p50, int(tokens_so_far))
    p90 = max(p50, int(p90))
    return {
        "predicted_tokens": p50,
        "predicted_p90": p90,
        "tier": tier_of(p50),
        "tier_p90": tier_of(p90),
        "stats": historical_stats(),
    }
