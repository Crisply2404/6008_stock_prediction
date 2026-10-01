"""Simple reference forecasts / 简单预测参照。"""
import numpy as np


def predict_baselines(train, evaluation):
    """Estimate constants using train only / 常数基线只从训练集估计。"""
    return {
        "zero_return": np.zeros(len(evaluation.y)),
        "historical_mean": np.full(len(evaluation.y), train.y.mean()),
        "last_return": evaluation.last_return.copy(),
    }
