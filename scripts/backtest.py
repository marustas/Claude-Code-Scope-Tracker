"""Expanding-window backtest: old raw model vs new log-space quantile model.

Replays your real logged tasks in time order. For each task i (i >= MIN), trains
on tasks 0..i-1 and predicts task i — the honest, no-leakage setup. Reports
median absolute % error (MdAPE, robust to the small-actual blowups that make mean
MAPE meaningless here), "within 2x" hit rate, and p90 coverage for the new model.

Re-extracts features from the stored raw prompt so the new IDE-stripping +
inflection logic is exercised (the DB's cached prompt_features predate it).

Run: python -m scripts.backtest
"""
from __future__ import annotations

import math
import statistics

from scope_tracker import core


def _load_ordered():
    conn = core.db()
    rows = conn.execute(
        """SELECT prompt, repo_features, actual_total_tokens
           FROM tasks
           WHERE completed = 1 AND actual_total_tokens > 0
           ORDER BY started_at ASC"""
    ).fetchall()
    conn.close()
    import json
    out = []
    for prompt, rf_json, tokens in rows:
        try:
            rf = json.loads(rf_json or "{}")
        except json.JSONDecodeError:
            rf = {}
        pf = core.extract_prompt_features(prompt or "")
        out.append((core.features_to_vector(pf, rf), float(tokens)))
    return out


def _fit_old(X, y):
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                  learning_rate=0.08, random_state=42)
    m.fit(X, y)
    return m


def _fit_new(X, y):
    from sklearn.ensemble import GradientBoostingRegressor
    ylog = [math.log1p(v) for v in y]
    models = {}
    for q in core.QUANTILES:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                      max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(X, ylog)
        models[q] = m
    return models


def main():
    data = _load_ordered()
    n = len(data)
    start = core.MIN_TASKS_FOR_PREDICTION
    if n <= start:
        print(f"not enough data: {n} tasks (need > {start})")
        return

    old_ape, new_ape = [], []
    old_within2x = new_within2x = 0
    p90_covered = 0
    evaluated = 0

    for i in range(start, n):
        Xtr = [d[0] for d in data[:i]]
        ytr = [d[1] for d in data[:i]]
        xi, yi = data[i]

        old_model = _fit_old(Xtr, ytr)
        new_models = _fit_new(Xtr, ytr)

        old_pred = max(0.0, float(old_model.predict([xi])[0]))
        p50 = math.expm1(float(new_models[0.5].predict([xi])[0]))
        p90 = math.expm1(float(new_models[0.9].predict([xi])[0]))
        p50 = max(0.0, p50)
        p90 = max(p50, p90)

        old_ape.append(abs(old_pred - yi) / yi * 100)
        new_ape.append(abs(p50 - yi) / yi * 100)
        if 0.5 <= (old_pred / yi if yi else 0) <= 2.0:
            old_within2x += 1
        if 0.5 <= (p50 / yi if yi else 0) <= 2.0:
            new_within2x += 1
        if yi <= p90:
            p90_covered += 1
        evaluated += 1

    def fmt(xs):
        return (f"median={statistics.median(xs):6.0f}%  "
                f"mean={statistics.mean(xs):7.0f}%")

    print(f"backtested {evaluated} predictions (expanding window, no leakage)\n")
    print(f"  OLD (raw counts, sq.err):   {fmt(old_ape)}")
    print(f"  NEW (log-space, p50):       {fmt(new_ape)}")
    print()
    print(f"  within 2x of actual:  OLD {old_within2x/evaluated*100:4.0f}%   "
          f"NEW {new_within2x/evaluated*100:4.0f}%")
    print(f"  p90 coverage (NEW):   {p90_covered/evaluated*100:4.0f}%  "
          f"(target ~90% if well calibrated)")


if __name__ == "__main__":
    main()
