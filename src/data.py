"""Causal features and target-date splits / 因果特征与按目标日切分。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    prefix + name
    for prefix in ("", "nasdaq_")
    for name in ("return", "vol_change", "intraday_return", "range")
]


@dataclass
class Samples:
    """Aligned inputs and future labels / 对齐的历史输入与未来标签。"""
    X: np.ndarray
    y: np.ndarray
    dates: pd.DataFrame
    last_return: np.ndarray


@dataclass
class Dataset:
    """Shared data for all models / 全部模型共用的数据与训练统计量。"""
    train: Samples
    validation: Samples
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    feature_frame: pd.DataFrame
    audit: dict

    def transform(self, X):
        """Use training statistics only / 只应用训练区间统计量。"""
        return ((X - self.feature_mean) / self.feature_scale).astype(np.float32)


def make_features(raw):
    """Compute features without filling gaps / 计算历史特征，不填补数据缺口。"""
    result = pd.DataFrame(index=raw.index)
    for prefix in ("", "nasdaq_"):
        close, volume = raw[prefix + "close"], raw[prefix + "volume"]
        result[prefix + "return"] = close / close.shift(1) - 1
        result[prefix + "vol_change"] = volume / volume.shift(1) - 1
        result[prefix + "intraday_return"] = close / raw[prefix + "open"] - 1
        result[prefix + "range"] = (raw[prefix + "high"] - raw[prefix + "low"]) / close
    return result.replace([np.inf, -np.inf], np.nan)


def build_dataset(raw, config):
    """Build train/validation only; future rows excluded first.

    先截断到验证期，再构造特征；缺失日不删除，避免把间隔两日误当成次日。
    """
    window = config["window"]
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    start, train_end, val_end = map(pd.Timestamp, (
        config["start"], config["train_end"], config["validation_end"]))
    if not start < train_end < val_end:
        raise ValueError("Require start < train_end < validation_end")
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    raw = raw.loc[raw.date.between(start, val_end)].reset_index(drop=True)
    if raw.empty or raw.date.duplicated().any() or not raw.date.is_monotonic_increasing:
        raise ValueError("Selected dates must be nonempty, unique and increasing")
    required = [p + c for p in ("", "nasdaq_")
                for c in ("open", "high", "low", "close", "volume")]
    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
        if np.isinf(raw[col]).any() or (raw[col] < 0).any():
            raise ValueError(f"Invalid values in {col}")
        if not col.endswith("volume") and (raw[col] == 0).any():
            raise ValueError(f"Nonpositive price in {col}")
    for p in ("", "nasdaq_"):
        high, low = raw[p + "high"], raw[p + "low"]
        if ((high < low) | (raw[p + "open"] > high + 1e-8)
                | (raw[p + "open"] < low - 1e-8)
                | (raw[p + "close"] > high + 1e-8)
                | (raw[p + "close"] < low - 1e-8)).any():
            raise ValueError(f"Inconsistent OHLC for {p or 'stock'}")
    frame = make_features(raw)
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    # 目标与原始下一行绑定，不能在 dropna 后重新 shift / Shift before any filtering.
    labels = (raw.close.shift(-1) / raw.close - 1).to_numpy()
    buckets = {name: [] for name in ("train", "validation")}
    used_train_rows = np.zeros(len(raw), dtype=bool)
    skipped = {name: 0 for name in buckets}
    for i in range(window - 1, len(raw) - 1):
        target_date = raw.date.iloc[i + 1]
        partition = "train" if target_date <= train_end else "validation"
        x = values[i - window + 1:i + 1]
        if not np.isfinite(x).all() or not np.isfinite(labels[i]):
            skipped[partition] += 1
            continue
        buckets[partition].append((i, x.copy(), labels[i]))
        if partition == "train":
            used_train_rows[i - window + 1:i + 1] = True
    if any(not rows for rows in buckets.values()):
        raise ValueError("No usable train or validation samples; inspect dates/window/missing data")

    def assemble(rows):
        indices = np.array([row[0] for row in rows])
        return Samples(
            X=np.stack([row[1] for row in rows]),
            y=np.array([row[2] for row in rows]),
            dates=pd.DataFrame({
                "window_start": raw.date.iloc[indices - window + 1].to_numpy(),
                "signal_date": raw.date.iloc[indices].to_numpy(),
                "target_date": raw.date.iloc[indices + 1].to_numpy(),
            }),
            last_return=values[indices, 0].copy(),
        )

    train, val = assemble(buckets["train"]), assemble(buckets["validation"])
    unique_inputs = values[used_train_rows]
    scale = unique_inputs.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    y_scale = float(train.y.std(ddof=0))
    if y_scale == 0:
        y_scale = 1.0
    audit = {
        "read_cutoff": str(val_end.date()),
        "split_by": "target_date",
        "selected_raw_rows": len(raw),
        "train_samples": len(train.y), "validation_samples": len(val.y),
        "skipped_invalid_windows": skipped,
        "scaler_unique_rows": int(used_train_rows.sum()),
        "scaler_first_date": str(raw.loc[used_train_rows, "date"].iloc[0].date()),
        "scaler_last_date": str(raw.loc[used_train_rows, "date"].iloc[-1].date()),
        "train_target_first": str(train.dates.target_date.min().date()),
        "train_target_last": str(train.dates.target_date.max().date()),
        "validation_target_first": str(val.dates.target_date.min().date()),
        "validation_target_last": str(val.dates.target_date.max().date()),
        "test_evaluated": False, "holdout_evaluated": False,
    }
    frame.insert(0, "date", raw.date)
    return Dataset(train, val, unique_inputs.mean(axis=0), scale,
                   float(train.y.mean()), y_scale, frame, audit)


def load_dataset(config, root):
    """Read project-relative CSV / 相对于项目根目录读取数据。"""
    return build_dataset(pd.read_csv(Path(root) / config["data_path"]), config)
