"""Standard LSTM regression adapter / 标准 LSTM 回归适配。"""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    """Many-to-one unidirectional LSTM for next-return regression."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_size, hidden_size and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=lstm_dropout,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """[batch, time, features] to one scaled return."""
        if x.ndim != 3:
            raise ValueError("Expected input shape [batch, time, features]")
        if x.shape[2] != self.input_size:
            raise ValueError("Unexpected feature count")
        if x.shape[1] < 1:
            raise ValueError("Sequence length must be positive")
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def fit_predict(dataset, settings, seed):
    """Fit on train, select epoch on validation, return raw-return predictions."""
    started = time.perf_counter()
    device_name = settings["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_kwargs = {
        "input_size": dataset.train.X.shape[-1],
        "hidden_size": settings["hidden_size"],
        "num_layers": settings["num_layers"],
        "dropout": settings["dropout"],
    }
    model = LSTMRegressor(**model_kwargs).to(device)
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
        train_loss = total / len(train_y)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(val_x), val_y).item()
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Non-finite validation loss")
        history.append({"epoch": epoch, "train_mse_scaled": train_loss,
                        "validation_mse_scaled": validation_loss})
        print(f"LSTM epoch {epoch:02d}: train={train_loss:.5f} "
              f"val={validation_loss:.5f}", flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if stale >= settings["patience"]:
            print(f"LSTM early stopping at epoch {epoch:02d}", flush=True)
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
               "model_kwargs": model_kwargs, "history": history,
               "sequence_length": int(dataset.train.X.shape[1])}
    return model, prediction, details
