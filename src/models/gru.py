"""GRU regression adapter / GRU 次日收益回归适配实现。

The model consumes the shared [batch, window, channels] sequence and predicts
one next-day return. Training and preprocessing follow the PatchTST adapter.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GRURegressor(nn.Module):
    """Unidirectional GRU with a scalar regression head.

    Input shape: [batch, window, channels]
    Output shape: [batch]
    """

    def __init__(self, window, channels, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        if window < 1 or channels < 1:
            raise ValueError("window and channels must be positive")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if num_layers == 1 and dropout != 0:
            raise ValueError("dropout must be 0 when num_layers=1")
        self.window, self.channels = window, channels
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Map [batch, time, channel] to one return / 序列映射为单个收益。"""
        if x.ndim != 3 or x.shape[1:] != (self.window, self.channels):
            raise ValueError("Unexpected input shape")
        _, h_n = self.gru(x)
        # h_n[-1] is the final hidden state of the last GRU layer.
        return self.head(h_n[-1]).squeeze(-1)


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
        "hidden_size", "num_layers", "dropout")}
    kwargs.update(window=dataset.train.X.shape[1], channels=dataset.train.X.shape[2])
    model = GRURegressor(**kwargs).to(device)

    train_x = torch.from_numpy(dataset.transform(dataset.train.X))
    train_y = torch.tensor(
        (dataset.train.y - dataset.target_mean) / dataset.target_scale,
        dtype=torch.float32,
    )
    val_x = torch.from_numpy(dataset.transform(dataset.validation.X)).to(device)
    val_y = torch.tensor(
        (dataset.validation.y - dataset.target_mean) / dataset.target_scale,
        dtype=torch.float32, device=device,
    )

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
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

        train_loss = total / len(train_y)
        history.append({
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "validation_mse_scaled": validation_loss,
        })
        print(
            f"GRU epoch {epoch:02d}: train={train_loss:.5f} val={validation_loss:.5f}",
            flush=True,
        )

        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
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

    details = {
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_kwargs": kwargs,
        "history": history,
        "variant": "unidirectional GRU; final hidden state; scalar regression head",
    }
    return model, prediction, details
