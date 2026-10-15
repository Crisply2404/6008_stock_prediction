"""Shared data, metric and model checks / 统一数据、指标和模型检查。"""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import Booster

from src.data import FEATURES, build_dataset, make_features
from src.evaluate import evaluate
from src.models.lightgbm import fit_predict
from src.models.patchtst import PatchTSTRegressor
from src.models.lstm import LSTMRegressor
from src.models.gru import GRURegressor

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    """Small independent price series / 可手算的小型价格序列。"""
    n = 24
    close = 100 + np.arange(n) ** 1.2
    frame = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n)})
    for prefix in ("", "nasdaq_"):
        frame[prefix + "open"] = close - 0.5
        frame[prefix + "high"] = close + 1
        frame[prefix + "low"] = close - 1
        frame[prefix + "close"] = close
        frame[prefix + "volume"] = 1000 + np.arange(n) * 10
    config = {"window": 3, "start": "2020-01-01",
              "train_end": str(frame.date.iloc[13].date()),
              "validation_end": str(frame.date.iloc[20].date())}
    return frame, config


class DataTests(unittest.TestCase):
    def test_nasdaq_ablation_preserves_samples_and_scaling(self):
        raw, config = fixture()
        full = build_dataset(raw, config)
        reduced = build_dataset(raw, {**config, "nasdaq_features": "return_volume"})
        self.assertEqual(full.features, FEATURES)
        self.assertEqual(reduced.features, FEATURES[:6])
        self.assertEqual(list(reduced.feature_frame), ["date", *FEATURES[:6]])
        for split in ("train", "validation"):
            original, subset = getattr(full, split), getattr(reduced, split)
            np.testing.assert_array_equal(subset.X, original.X[:, :, :6])
            np.testing.assert_array_equal(subset.y, original.y)
            np.testing.assert_array_equal(subset.last_return, original.last_return)
            pd.testing.assert_frame_equal(subset.dates, original.dates)
        np.testing.assert_allclose(reduced.feature_mean, full.feature_mean[:6])
        np.testing.assert_allclose(reduced.feature_scale, full.feature_scale[:6])
        self.assertEqual(reduced.target_mean, full.target_mean)
        self.assertEqual(reduced.target_scale, full.target_scale)

    def test_removed_nasdaq_feature_does_not_add_samples(self):
        raw, config = fixture()
        raw.loc[[8, 16], "nasdaq_open"] = np.nan
        full = build_dataset(raw, config)
        reduced = build_dataset(raw, {**config, "nasdaq_features": "return_volume"})
        for split in ("train", "validation"):
            pd.testing.assert_frame_equal(getattr(full, split).dates, getattr(reduced, split).dates)
        self.assertEqual(full.audit["skipped_invalid_windows"],
                         reduced.audit["skipped_invalid_windows"])

    def test_unknown_nasdaq_mode_rejected(self):
        raw, config = fixture()
        with self.assertRaisesRegex(ValueError, "nasdaq_features"):
            build_dataset(raw, {**config, "nasdaq_features": "typo"})

    def test_label_and_context_alignment(self):
        raw, config = fixture()
        d = build_dataset(raw, config)
        self.assertEqual(len(d.train.y), 10)
        self.assertEqual(len(d.validation.y), 7)
        self.assertAlmostEqual(d.train.y[0], raw.close.iloc[4] / raw.close.iloc[3] - 1)
        self.assertEqual(d.validation.dates.iloc[0].signal_date, raw.date.iloc[13])
        self.assertEqual(d.validation.dates.iloc[0].target_date, raw.date.iloc[14])
        self.assertEqual(d.validation.dates.iloc[0].window_start, raw.date.iloc[11])
        self.assertLessEqual(d.train.dates.target_date.max(), pd.Timestamp(config["train_end"]))

    def test_scaler_uses_unique_train_input_dates(self):
        raw, config = fixture()
        d = build_dataset(raw, config)
        expected = make_features(raw).iloc[1:13].to_numpy()
        np.testing.assert_allclose(d.feature_mean, expected.mean(0))
        np.testing.assert_allclose(d.feature_scale, expected.std(0))
        self.assertEqual(d.audit["scaler_unique_rows"], 12)

    def test_validation_changes_cannot_affect_train(self):
        raw, config = fixture()
        original = build_dataset(raw, config)
        changed = raw.copy()
        mask = changed.date > pd.Timestamp(config["train_end"])
        changed.loc[mask, changed.columns != "date"] *= 5
        d = build_dataset(changed, config)
        np.testing.assert_array_equal(d.train.X, original.train.X)
        np.testing.assert_array_equal(d.train.y, original.train.y)
        np.testing.assert_array_equal(d.feature_mean, original.feature_mean)
        self.assertEqual(d.target_mean, original.target_mean)

    def test_sealed_future_is_excluded_before_features(self):
        raw, config = fixture()
        original = build_dataset(raw, config)
        raw.loc[raw.date > pd.Timestamp(config["validation_end"]), "close"] = -999
        d = build_dataset(raw, config)
        np.testing.assert_array_equal(d.validation.X, original.validation.X)
        np.testing.assert_array_equal(d.validation.y, original.validation.y)

    def test_missing_prices_do_not_bridge_dates(self):
        raw, config = fixture()
        raw.loc[16, "close"] = np.nan
        d = build_dataset(raw, config)
        self.assertNotIn(raw.date.iloc[16], d.validation.dates.target_date.tolist())
        self.assertNotIn(raw.date.iloc[17], d.validation.dates.target_date.tolist())
        self.assertTrue(all((d.validation.dates.target_date - d.validation.dates.signal_date).dt.days <= 3))
        self.assertGreater(d.audit["skipped_invalid_windows"]["validation"], 0)

    def test_zero_volume_is_not_infinite_or_filled(self):
        raw, config = fixture()
        raw.loc[8, "volume"] = 0
        f = make_features(raw)
        self.assertTrue(np.isnan(f.vol_change.iloc[9]))
        d = build_dataset(raw, config)
        self.assertTrue(np.isfinite(d.train.X).all())
        self.assertNotIn(raw.date.iloc[10], d.train.dates.target_date.tolist())

    def test_duplicate_dates_rejected(self):
        raw, config = fixture()
        raw.loc[3, "date"] = raw.date.iloc[2]
        with self.assertRaises(ValueError):
            build_dataset(raw, config)

    def test_no_silent_sort(self):
        raw, config = fixture()
        with self.assertRaises(ValueError):
            build_dataset(raw.iloc[::-1], config)

    def test_invalid_ohlc_rejected(self):
        raw, config = fixture()
        raw.loc[5, "high"] = 1
        with self.assertRaises(ValueError):
            build_dataset(raw, config)

    def test_invalid_split_rejected(self):
        raw, config = fixture()
        config["train_end"] = config["validation_end"]
        with self.assertRaises(ValueError):
            build_dataset(raw, config)


class MetricTests(unittest.TestCase):
    def test_zero_baseline_and_flat_direction(self):
        metrics = evaluate(np.array([-0.1, 0.0, 0.1]), np.zeros(3))
        self.assertEqual(metrics["r2_vs_zero"], 0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 0]])
        self.assertEqual(metrics["flat_count"], 1)
        self.assertAlmostEqual(metrics["direction_accuracy"], 2 / 3)

    def test_undefined_r2_is_none(self):
        metrics = evaluate(np.zeros(3), np.zeros(3))
        self.assertIsNone(metrics["r2"])
        self.assertIsNone(metrics["r2_vs_zero"])

    def test_missing_prediction_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(np.ones(3), np.array([0, 1, np.nan]))


class PatchTests(unittest.TestCase):
    def test_shape_gradient_and_restore(self):
        kwargs = dict(window=12, channels=2, patch_len=4, stride=4,
                      d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0)
        model = PatchTSTRegressor(**kwargs)
        x = torch.randn(5, 12, 2)
        y = model(x)
        self.assertEqual(y.shape, (5,))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(model.projection.weight.grad).all())
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = PatchTSTRegressor(**kwargs)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        model.eval()
        restored.eval()
        torch.testing.assert_close(model(x), restored(x))

    def test_patch_tail_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            PatchTSTRegressor(window=13, channels=2, patch_len=4, stride=4)

    def test_input_channel_count_checked(self):
        model = PatchTSTRegressor(window=12, channels=2, patch_len=4, stride=4)
        with self.assertRaises(ValueError):
            model(torch.randn(3, 12, 3))


class LightGBMTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 150
        raw = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n)})
        for prefix in ("", "nasdaq_"):
            close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
            raw[prefix + "open"] = close * 0.995
            raw[prefix + "high"] = close * 1.01
            raw[prefix + "low"] = close * 0.99
            raw[prefix + "close"] = close
            raw[prefix + "volume"] = rng.integers(1000, 5000, n)
        config = {"window": 12, "start": "2020-01-01",
                  "train_end": str(raw.date.iloc[99].date()),
                  "validation_end": str(raw.date.iloc[-1].date())}
        self.dataset = build_dataset(raw, config)
        self.reduced_dataset = build_dataset(raw, {**config, "nasdaq_features": "return_volume"})
        self.settings = {"n_estimators": 40, "learning_rate": 0.05,
                         "num_leaves": 7, "min_child_samples": 5,
                         "early_stopping_rounds": 5, "n_jobs": 1, "verbosity": -1}

    def test_six_feature_interface_and_names(self):
        model, prediction, details = fit_predict(self.reduced_dataset, self.settings, seed=42)
        self.assertEqual(prediction.shape, self.reduced_dataset.validation.y.shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(details["channels"], 6)
        self.assertEqual(model.n_features_, 12 * 6)
        self.assertEqual(details["feature_names"][:6], [f"lag_11_{f}" for f in FEATURES[:6]])
        self.assertEqual(details["feature_names"][-6:], [f"lag_0_{f}" for f in FEATURES[:6]])

    def test_interface_and_lag_order(self):
        settings = copy.deepcopy(self.settings)
        model, prediction, details = fit_predict(self.dataset, settings, seed=42)
        self.assertEqual(settings, self.settings)
        self.assertEqual(prediction.shape, self.dataset.validation.y.shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(details["n_features"], 12 * 8)
        self.assertEqual(details["feature_names"][:8], [f"lag_11_{f}" for f in FEATURES])
        self.assertEqual(details["feature_names"][-8:], [f"lag_0_{f}" for f in FEATURES])
        best = details["best_iteration"]
        self.assertGreaterEqual(best, 1)
        history = details["history"]["validation"]["rmse"]
        self.assertAlmostEqual(history[best - 1], min(history))
        self.assertEqual(model.n_features_, 96)
        json.dumps(details, allow_nan=False)

    def test_prediction_scale_and_saved_booster(self):
        model, prediction, details = fit_predict(self.dataset, self.settings, seed=42)
        validation = self.dataset.transform(self.dataset.validation.X).reshape(
            len(self.dataset.validation.y), -1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lightgbm.txt"
            model.booster_.save_model(str(path), num_iteration=details["best_iteration"])
            restored = Booster(model_file=str(path))
            actual = (restored.predict(validation) * self.dataset.target_scale
                      + self.dataset.target_mean)
            np.testing.assert_allclose(actual, prediction, rtol=1e-9, atol=1e-12)
        rmse = float(np.sqrt(np.mean((prediction - self.dataset.validation.y) ** 2)))
        scaled = details["history"]["validation"]["rmse"][details["best_iteration"] - 1]
        self.assertAlmostEqual(rmse, scaled * self.dataset.target_scale, places=7)

    def test_reproducible_seed(self):
        _, first, _ = fit_predict(self.dataset, self.settings, seed=42)
        _, second, _ = fit_predict(self.dataset, self.settings, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_without_early_stopping_records_actual_tree_count(self):
        settings = dict(self.settings, early_stopping_rounds=0, n_estimators=4)
        model, _, details = fit_predict(self.dataset, settings, seed=42)
        self.assertEqual(details["best_iteration"], model.booster_.current_iteration())
        self.assertEqual(details["best_iteration"], 4)

    def test_shared_config_contains_all_models(self):
        common = json.loads((ROOT / "configs/aapl.json").read_text(encoding="utf-8"))
        required = ("ticker", "data_path", "start", "train_end", "validation_end",
                    "test_end", "holdout_end", "window", "seed")
        self.assertTrue(all(key in common for key in required))
        for name in ("xgboost", "lightgbm", "patchtst", "lstm", "gru"):
            with self.subTest(model=name):
                self.assertIsInstance(common[name], dict)
                self.assertGreater(common[name]["learning_rate"], 0)

    def test_help_does_not_import_other_model_runtimes(self):
        code = (
            "import sys; from scripts.run_experiment import main; "
            "sys.argv=['main.py','--help']; "
            "\ntry: main()\nexcept SystemExit as e: assert e.code == 0\n"
            "assert 'torch' not in sys.modules; assert 'xgboost' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("xgboost", "lightgbm", "patchtst", "lstm", "gru"):
            self.assertIn(name, result.stdout)


class LSTMTests(unittest.TestCase):
    def test_forward_shape(self):
        model = LSTMRegressor(input_size=8, hidden_size=16, num_layers=1, dropout=0)
        x = torch.randn(5, 12, 8)
        output = model(x)
        self.assertEqual(output.shape, (5,))

    def test_gradient_is_finite(self):
        model = LSTMRegressor(input_size=8, hidden_size=16, num_layers=1, dropout=0)
        output = model(torch.randn(5, 12, 8))
        output.square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_save_load_restore(self):
        kwargs = dict(input_size=8, hidden_size=16, num_layers=1, dropout=0)
        model = LSTMRegressor(**kwargs)
        x = torch.randn(5, 12, 8)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = LSTMRegressor(**kwargs)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        model.eval()
        restored.eval()
        torch.testing.assert_close(model(x), restored(x))

    def test_wrong_input_shape_checked(self):
        model = LSTMRegressor(input_size=8, hidden_size=16, num_layers=1, dropout=0)
        with self.assertRaises(ValueError):
            model(torch.randn(5, 8))
        with self.assertRaises(ValueError):
            model(torch.randn(5, 12, 10))


class GRUTests(unittest.TestCase):
    def test_forward_and_gradients(self):
        model = GRURegressor(window=12, channels=8, hidden_size=16)
        output = model(torch.randn(5, 12, 8))
        self.assertEqual(output.shape, (5,))
        output.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_checkpoint_restores_predictions(self):
        kwargs = dict(window=12, channels=8, hidden_size=16)
        model = GRURegressor(**kwargs).eval()
        x = torch.randn(5, 12, 8)
        buffer = io.BytesIO()
        torch.save({"model_kwargs": kwargs, "state_dict": model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = GRURegressor(**checkpoint["model_kwargs"]).eval()
        restored.load_state_dict(checkpoint["state_dict"])
        torch.testing.assert_close(model(x), restored(x))

    def test_rejects_wrong_window_and_features(self):
        model = GRURegressor(window=12, channels=8, hidden_size=16)
        for shape in ((5, 8), (5, 11, 8), (5, 12, 9)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))

    def test_single_layer_requires_zero_dropout(self):
        with self.assertRaises(ValueError):
            GRURegressor(window=12, channels=8, num_layers=1, dropout=0.2)
        model = GRURegressor(window=12, channels=8, num_layers=1, dropout=0)
        self.assertEqual(model(torch.randn(1, 12, 8)).shape, (1,))


if __name__ == "__main__":
    unittest.main()
