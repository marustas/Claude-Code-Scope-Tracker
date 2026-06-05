"""Train the mid-task model (fix D) from replayed transcripts, with an honest
task-level holdout eval, then save it to model_midtask.pkl.

Eval splits by TASK, not by checkpoint — checkpoints from one task never land in
both train and test, so the reported within-2x isn't inflated by leakage. We
evaluate at the first checkpoint past MIDTASK_MIN_TOOL_CALLS (when the hook would
actually fire) and report against submit-time accuracy for comparison.

Run: python -m scripts.train_midtask
"""
from __future__ import annotations

import json
import math
import random
import statistics

from scope_tracker import core


def _tasks_with_checkpoints():
    """Yield (pf, rf, final, checkpoints) per completed task by replaying transcripts."""
    conn = core.db()
    rows = conn.execute(
        """SELECT prompt_features, repo_features, transcript_path,
                  transcript_start_line, actual_total_tokens
           FROM tasks
           WHERE completed = 1 AND actual_total_tokens > 0
             AND transcript_path IS NOT NULL
           ORDER BY transcript_path, transcript_start_line"""
    ).fetchall()
    conn.close()

    starts_by_path: dict[str, list[int]] = {}
    for _, _, path, start, _ in rows:
        if path is not None and start is not None:
            starts_by_path.setdefault(path, []).append(start)
    for p in starts_by_path:
        starts_by_path[p].sort()

    out = []
    for pf_json, rf_json, path, start, final in rows:
        try:
            pf = json.loads(pf_json or "{}")
            rf = json.loads(rf_json or "{}")
        except json.JSONDecodeError:
            continue
        starts = starts_by_path.get(path, [])
        end = next((s for s in starts if start is not None and s > start), None)
        ck = core._region_checkpoints(path, start or 0, end)
        out.append((pf, rf, float(final), ck))
    return out


def _fit(X, y):
    from sklearn.ensemble import GradientBoostingRegressor
    ylog = [math.log1p(v) for v in y]
    models = {}
    for q in core.QUANTILES:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                      max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(X, ylog)
        models[q] = m
    return models


def _first_checkpoint(ck):
    for tools, tokens, counts in ck:
        if tools >= core.MIDTASK_MIN_TOOL_CALLS:
            return tools, tokens, counts
    return None


def evaluate(tasks, folds=5, seeds=(0, 1, 2, 3, 4)):
    """Task-level k-fold within-2x of the mid-task p50 at the first-fire checkpoint,
    averaged over several shuffles. Single-seed CV on ~70 tasks swings ±8 pts, so
    we average to report a number that isn't an artifact of one lucky split."""
    eligible = [t for t in tasks if _first_checkpoint(t[3])]
    if len(eligible) < folds * 2:
        print(f"  (only {len(eligible)} tasks reach {core.MIDTASK_MIN_TOOL_CALLS} "
              f"tool calls — eval skipped)")
        return

    w2_runs, cov_runs = [], []
    for seed in seeds:
        pool = eligible[:]
        random.Random(seed).shuffle(pool)
        buckets = [pool[i::folds] for i in range(folds)]
        within2x = p90_cov = evaluated = 0
        for f in range(folds):
            test = buckets[f]
            train = [t for i, b in enumerate(buckets) if i != f for t in b]
            X, y = [], []
            for pf, rf, final, ck in train:
                for tools, tokens, counts in ck:
                    if tools >= core.MIDTASK_MIN_TOOL_CALLS:
                        X.append(core.midtask_vector(pf, rf, tools, tokens, counts))
                        y.append(final)
            if len(X) < core.MIN_MIDTASK_ROWS:
                continue
            models = _fit(X, y)
            for pf, rf, final, ck in test:
                cp = _first_checkpoint(ck)
                if not cp:
                    continue
                tools, tokens, counts = cp
                vec = core.midtask_vector(pf, rf, tools, tokens, counts)
                p50 = max(tokens, math.expm1(float(models[0.5].predict([vec])[0])))
                p90 = max(p50, math.expm1(float(models[0.9].predict([vec])[0])))
                if final and 0.5 <= p50 / final <= 2.0:
                    within2x += 1
                if final <= p90:
                    p90_cov += 1
                evaluated += 1
        if evaluated:
            w2_runs.append(within2x / evaluated * 100)
            cov_runs.append(p90_cov / evaluated * 100)
    if w2_runs:
        print(f"  mid-task (at first ≥{core.MIDTASK_MIN_TOOL_CALLS} tool calls, "
              f"avg of {len(w2_runs)} seeds): "
              f"within-2x = {statistics.mean(w2_runs):.0f}%   "
              f"p90 coverage = {statistics.mean(cov_runs):.0f}%   (n={len(eligible)})")
        print(f"  submit-time baseline was ~41% within-2x")


def main():
    tasks = _tasks_with_checkpoints()
    print(f"replayed {len(tasks)} tasks")
    X, y = core._midtask_training_rows()
    print(f"training checkpoints (tool_calls >= {core.MIDTASK_MIN_TOOL_CALLS}): {len(X)}")
    if len(X) < core.MIN_MIDTASK_ROWS:
        print(f"not enough rows (need {core.MIN_MIDTASK_ROWS}) — model not trained")
        return
    print("\nholdout eval:")
    evaluate(tasks)

    model = core.train_midtask_model()
    if model is not None:
        core.save_midtask_model(model)
        print(f"\nsaved mid-task model -> {core.MIDTASK_MODEL_PATH}")


if __name__ == "__main__":
    main()
