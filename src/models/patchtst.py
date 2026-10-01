"""PatchTST-style single-target adapter / PatchTST 单目标适配实现。

Independent implementation of patching and shared channel encoding, inspired by
https://github.com/yuqinie98/PatchTST (Nie et al., ICLR 2023).
Uses LayerNorm, no residual attention/RevIN, and a joint scalar head.
使用 LayerNorm，不启用残差注意力或 RevIN；多通道表征汇总为一个收益预测。
This is a documented adaptation, not a reproduction of the official benchmark.
"""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PatchTSTRegressor(nn.Module):
    """One shared patch encoder across channels / 各变量共用时间块编码器。"""

    def __init__(self, window, channels, patch_len=12, stride=6, d_model=32,
                 n_heads=4, n_layers=2, d_ff=64, dropout=0.2):
        super().__init__()
        if not 0 < stride <= patch_len <= window or (window - patch_len) % stride:
            raise ValueError("Patches must cover the whole window without gaps or an unused tail")
        if d_model % n_heads or n_layers < 1:
            raise ValueError("Invalid heads/layers")
        self.window, self.channels = window, channels
        self.patch_len, self.stride = patch_len, stride
        patches = 1 + (window - patch_len) // stride
        self.projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(1, patches, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        # Separate initialization per layer / 每层独立初始化，避免克隆同一初始权重。
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                       activation="gelu", batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Dropout(dropout),
                                  nn.Linear(channels * patches * d_model, 1))

    def forward(self, x):
        """[batch, time, channel] to one return / 历史序列映射为单个收益。"""
        if x.ndim != 3 or x.shape[1:] != (self.window, self.channels):
            raise ValueError("Unexpected input shape")
        batch = len(x)
        patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
        tokens = self.projection(patches).flatten(0, 1)
        tokens = self.input_dropout(tokens + self.position)
        # All tokens are historical: no future mask needed / 所有块均在信号日前。
        for layer in self.layers:
            tokens = layer(tokens)
        return self.head(tokens.reshape(batch, self.channels, -1)).squeeze(-1)


def fit_predict(dataset, settings, seed):
    """Fit on train, select epoch on validation / 训练拟合，验证集选择轮次。"""
    started = time.perf_counter()
    device_name = settings["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    kwargs = {key: settings[key] for key in (
        "patch_len", "stride", "d_model", "n_heads", "n_layers", "d_ff", "dropout")}
    kwargs.update(window=dataset.train.X.shape[1], channels=dataset.train.X.shape[2])
    model = PatchTSTRegressor(**kwargs).to(device)
    train_x = torch.from_numpy(dataset.transform(dataset.train.X))
    train_y = torch.tensor((dataset.train.y - dataset.target_mean) / dataset.target_scale,
                           dtype=torch.float32)
    val_x = torch.from_numpy(dataset.transform(dataset.validation.X)).to(device)
    val_y = torch.tensor((dataset.validation.y - dataset.target_mean) / dataset.target_scale,
                         dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=settings["batch_size"],
                        shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    loss_fn = nn.MSELoss()
    best_loss, stale, best_epoch, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(val_x), val_y).item()
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Non-finite validation loss")
        history.append({"epoch": epoch, "train_mse_scaled": total / len(train_y),
                        "validation_mse_scaled": validation_loss})
        print(f"PatchTST epoch {epoch:02d}: train={total / len(train_y):.5f} "
              f"val={validation_loss:.5f}", flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise ValueError("epochs must be positive")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(val_x).cpu().numpy().astype(np.float64)
    prediction = prediction * dataset.target_scale + dataset.target_mean
    details = {"seconds": time.perf_counter() - started, "device": str(device),
               "best_epoch": best_epoch, "epochs_run": len(history),
               "parameters": sum(p.numel() for p in model.parameters()),
               "model_kwargs": kwargs, "history": history,
               "variant": "shared-channel patch encoder; LayerNorm; no RevIN/residual attention; joint scalar head"}
    return model, prediction, details
