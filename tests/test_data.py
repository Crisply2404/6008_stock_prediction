"""Behavioral checks for temporal integrity / 时间关系与数值行为检查。"""
import unittest
import numpy as np
import pandas as pd

from src.data import build_dataset, make_features
from src.evaluate import evaluate


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


if __name__ == "__main__":
    unittest.main()
