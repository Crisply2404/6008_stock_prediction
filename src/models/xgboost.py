"""XGBoost adapter with validation early stopping / 验证集早停适配。"""
import time
import numpy as np
from xgboost import XGBRegressor


def fit_predict(dataset, settings, seed):
    """Flatten the same historical windows / 展开与序列模型相同的窗口。"""
    started = time.perf_counter()
    train = dataset.transform(dataset.train.X).reshape(len(dataset.train.y), -1)
    val = dataset.transform(dataset.validation.X).reshape(len(dataset.validation.y), -1)
    yt = (dataset.train.y - dataset.target_mean) / dataset.target_scale
    yv = (dataset.validation.y - dataset.target_mean) / dataset.target_scale
    model = XGBRegressor(objective="reg:squarederror", eval_metric="rmse",
                         tree_method="hist", random_state=seed, **settings)
    model.fit(train, yt, eval_set=[(val, yv)], verbose=False)
    # sklearn API automatically uses best_iteration / sklearn 接口自动使用最佳迭代。
    prediction = model.predict(val).astype(np.float64) * dataset.target_scale + dataset.target_mean
    details = {"seconds": time.perf_counter() - started,
               "best_iteration": int(model.best_iteration),
               "device": settings.get("device", "cpu"),
               "history": model.evals_result()}
    return model, prediction, details
