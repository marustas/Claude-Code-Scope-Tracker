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

    w2_runs, nom_runs, cal_runs = [], [], []
    for seed in seeds:
        pool = eligible[:]
        random.Random(seed).shuffle(pool)
        buckets = [pool[i::folds] for i in range(folds)]
        within2x = nom_cov = cal_cov = evaluated = 0
        for f in range(folds):
            test = buckets[f]
            train = [t for i, b in enumerate(buckets) if i != f for t in b]
            X, y, groups = [], [], []
            for gi, (pf, rf, final, ck) in enumerate(train):
                for tools, tokens, counts in ck:
                    if tools >= core.MIDTASK_MIN_TOOL_CALLS:
                        X.append(core.midtask_vector(pf, rf, tools, tokens, counts))
                        y.append(final)
                        groups.append(gi)
            if len(X) < core.MIN_MIDTASK_ROWS:
                continue
            models = core._fit_quantile_bundle(X, y, groups=groups)
            corr = float(models.get(core.P90_CORR_KEY, 0.0))
            for pf, rf, final, ck in test:
                cp = _first_checkpoint(ck)
                if not cp:
                    continue
                tools, tokens, counts = cp
                vec = core.midtask_vector(pf, rf, tools, tokens, counts)
                p50 = max(tokens, math.expm1(float(models[0.5].predict([vec])[0])))
                p90_log = float(models[0.9].predict([vec])[0])
                nom = max(p50, math.expm1(p90_log))             # nominal p90
                cal = max(p50, math.expm1(p90_log + corr))      # conformal-calibrated
                if final and 0.5 <= p50 / final <= 2.0:
                    within2x += 1
                if final <= nom:
                    nom_cov += 1
                if final <= cal:
                    cal_cov += 1
                evaluated += 1
        if evaluated:
            w2_runs.append(within2x / evaluated * 100)
            nom_runs.append(nom_cov / evaluated * 100)
            cal_runs.append(cal_cov / evaluated * 100)
    if w2_runs:
        print(f"  mid-task (at first ≥{core.MIDTASK_MIN_TOOL_CALLS} tool calls, "
              f"avg of {len(w2_runs)} seeds, n={len(eligible)}):")
        print(f"    within-2x (p50)          = {statistics.mean(w2_runs):.0f}%  "
              f"(submit-time baseline ~41%)")
        print(f"    p90 coverage, nominal    = {statistics.mean(nom_runs):.0f}%")
        print(f"    p90 coverage, conformal  = {statistics.mean(cal_runs):.0f}%  "
              f"(target ≥90%)")


def main():
    tasks = _tasks_with_checkpoints()
    print(f"replayed {len(tasks)} tasks")
    X, y, _ = core._midtask_training_rows()
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
