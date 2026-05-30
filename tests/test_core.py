"""Tests for feature extraction and the log-space quantile model.

Stdlib unittest only — no test dependency added. Run: python -m unittest discover -s tests
"""
from __future__ import annotations

import math
import unittest

from scope_tracker import core


class _ConstModel:
    """Stand-in regressor returning a constant log-space prediction."""

    def __init__(self, val: float):
        self.val = val

    def predict(self, X):
        return [self.val]


class TestInflectionMatching(unittest.TestCase):
    def test_inflected_high_cost_verbs_hit_tier_2(self):
        for prompt in ("implementing the auth layer",
                       "refactored the payments module",
                       "migrating the logic over",
                       "rewrote the parser",
                       "we are building a new service",
                       "generates the client"):
            with self.subTest(prompt=prompt):
                self.assertEqual(core.extract_prompt_features(prompt)["verb_tier"], 2)

    def test_inflected_medium_verbs_hit_tier_1(self):
        for prompt in ("fixing the failing test", "updated the config", "renaming things"):
            with self.subTest(prompt=prompt):
                self.assertEqual(core.extract_prompt_features(prompt)["verb_tier"], 1)

    def test_questions_stay_tier_0(self):
        self.assertEqual(core.extract_prompt_features("what does this function do?")["verb_tier"], 0)

    def test_refactor_inflections(self):
        self.assertTrue(core.extract_prompt_features("refactoring this")["mentions_refactor"])
        self.assertTrue(core.extract_prompt_features("a big refactor")["mentions_refactor"])
        self.assertFalse(core.extract_prompt_features("just read the file")["mentions_refactor"])

    def test_test_and_spec_inflections(self):
        self.assertTrue(core.extract_prompt_features("add tests")["mentions_test"])
        self.assertTrue(core.extract_prompt_features("write a spec")["mentions_test"])
        self.assertFalse(core.extract_prompt_features("the latest greatest thing")["mentions_test"])

    def test_no_false_positive_substrings(self):
        # 'testing' should match, but 'attest'/'contest' substrings must not via word boundary
        self.assertFalse(core.extract_prompt_features("the contestant arrived")["mentions_test"])


class TestInjectedContextStripping(unittest.TestCase):
    def test_paired_tag_removed(self):
        raw = "<ide_opened_file>The user opened /a/b.py with lots of text</ide_opened_file>\nfix the bug"
        self.assertEqual(core.strip_injected_context(raw), "fix the bug")

    def test_length_reflects_real_prompt_only(self):
        raw = "<ide_selection>" + ("x " * 200) + "</ide_selection> add a flag"
        feats = core.extract_prompt_features(raw)
        self.assertLess(feats["length"], 20)
        self.assertEqual(feats["verb_tier"], 1)  # 'add'

    def test_empty_prompt(self):
        self.assertEqual(core.strip_injected_context(""), "")
        self.assertEqual(core.extract_prompt_features("")["length"], 0)


class TestFeatureVectorStability(unittest.TestCase):
    def test_vector_length_matches_feature_names(self):
        pf = core.extract_prompt_features("implement a thing")
        rf = {"file_count": 42, "language": "py"}
        self.assertEqual(len(core.features_to_vector(pf, rf)), len(core.FEATURE_NAMES))


class TestLogSpaceQuantileModel(unittest.TestCase):
    def test_train_and_predict_roundtrip(self):
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")

        # Synthetic: token cost grows with verb_tier and repo size, spanning 3 orders.
        import random
        rng = random.Random(0)
        X, y = [], []
        for _ in range(120):
            tier = rng.choice([0, 1, 2])
            files = rng.randint(10, 5000)
            base = {0: 3_000, 1: 30_000, 2: 200_000}[tier]
            tokens = base * (0.5 + files / 5000) * rng.uniform(0.6, 1.6)
            pf = {"length": 50, "word_count": 8, "has_code_block": False,
                  "has_file_path": False, "mentions_test": False,
                  "mentions_refactor": tier == 2, "verb_tier": tier,
                  "question_count": 0, "mentions_all": False}
            X.append(core.features_to_vector(pf, {"file_count": files}))
            y.append(tokens)

        ylog = [math.log1p(v) for v in y]
        models = {}
        for q in core.QUANTILES:
            m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                          max_depth=3, learning_rate=0.05, random_state=42)
            m.fit(X, ylog)
            models[q] = m

        # A tier-2 / big-repo prompt should predict far above a tier-0 / small-repo one.
        big = core.features_to_vector({"verb_tier": 2, "mentions_refactor": True}, {"file_count": 5000})
        small = core.features_to_vector({"verb_tier": 0}, {"file_count": 10})
        p50_big = math.expm1(models[0.5].predict([big])[0])
        p50_small = math.expm1(models[0.5].predict([small])[0])
        self.assertGreater(p50_big, p50_small * 3)

    def test_predict_clamps_quantile_crossing(self):
        # GBM quantile models can cross (p90 < p50) at the edges of the feature
        # space. predict() must never expose that: p90 is clamped to >= p50.
        from unittest import mock

        crossing = {0.5: _ConstModel(12.0), 0.9: _ConstModel(9.0)}  # p50 > p90 raw
        with mock.patch.object(core, "load_model", return_value=crossing):
            out = core.predict("implement a thing", "")
        self.assertIsNotNone(out)
        self.assertGreaterEqual(out["predicted_p90"], out["predicted_tokens"])


class TestMidTaskModel(unittest.TestCase):
    def test_midtask_vector_length(self):
        pf = core.extract_prompt_features("implement a thing")
        rf = {"file_count": 100}
        vec = core.midtask_vector(pf, rf, tools_so_far=4, tokens_so_far=12000)
        self.assertEqual(len(vec), len(core.MIDTASK_FEATURE_NAMES))
        # appended features: tool count raw, tokens in log space
        self.assertEqual(vec[-2], 4.0)
        self.assertAlmostEqual(vec[-1], math.log1p(12000))

    def test_predict_midtask_floors_at_spent_and_clamps_p90(self):
        from unittest import mock
        # Raw models project LOW (log ~9 ≈ 8k) and crossed (p90<p50), but the task
        # has already burned 40k — the projection must never dip below that.
        crossing = {0.5: _ConstModel(9.0), 0.9: _ConstModel(8.0)}
        with mock.patch.object(core, "load_midtask_model", return_value=crossing):
            out = core.predict_midtask({"verb_tier": 2}, {"file_count": 100},
                                       tools_so_far=5, tokens_so_far=40000)
        self.assertIsNotNone(out)
        self.assertGreaterEqual(out["predicted_tokens"], 40000)  # floored at spent
        self.assertGreaterEqual(out["predicted_p90"], out["predicted_tokens"])

    def test_predict_midtask_none_without_model(self):
        from unittest import mock
        with mock.patch.object(core, "load_midtask_model", return_value=None):
            self.assertIsNone(core.predict_midtask({}, {}, 5, 1000))


if __name__ == "__main__":
    unittest.main()
