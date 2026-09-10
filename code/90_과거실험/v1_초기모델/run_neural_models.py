"""Preliminary RNN/LSTM comparison on a regular hourly PV sequence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python_packages"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CAPACITY = 240.0
SEQ_LEN = 48
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)


def solar_elevation_approx(index: pd.DatetimeIndex) -> np.ndarray:
    """Approximate solar elevation at Gwangju ASOS coordinates, local KST."""
    lat = math.radians(35.17294)
    n = index.dayofyear.to_numpy()
    local_hour = index.hour.to_numpy() + 0.5
    gamma = 2 * math.pi / 365 * (n - 1 + (local_hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                       - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    decl = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))
    time_offset = eqtime + 4 * 126.89156 - 60 * 9
    true_solar_min = local_hour * 60 + time_offset
    ha = np.radians(true_solar_min / 4 - 180)
    elev = np.arcsin(np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(ha))
    return np.degrees(elev)


def regular_series():
    df = pd.read_csv(ROOT / "outputs" / "gwangju_hourly_plant.csv", parse_dates=["hour"])
    idx = pd.date_range(df["hour"].min().floor("h"), df["hour"].max().floor("h"), freq="h")
    grid = pd.DataFrame(index=idx)
    complete = df[df["complete_5_inverters"]].set_index("hour")["plant_output_power_mean_sum"]
    grid["power"] = complete.reindex(idx)
    grid["observed"] = grid["power"].notna().astype(float)
    elevation = solar_elevation_approx(idx)
    structural_night = (elevation <= -3) & grid["power"].isna().to_numpy()
    grid.loc[structural_night, "power"] = 0.0
    grid.loc[structural_night, "observed"] = 1.0
    grid["power_input"] = grid["power"].fillna(0.0) / CAPACITY
    hour = idx.hour + 0.5
    doy = idx.dayofyear
    grid["hour_sin"] = np.sin(2*np.pi*hour/24)
    grid["hour_cos"] = np.cos(2*np.pi*hour/24)
    grid["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    grid["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    return grid


def sequences(grid):
    cols = ["power_input", "observed", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    xall = grid[cols].to_numpy(np.float32)
    yall = grid["power"].to_numpy(float)
    xs, ys, times = [], [], []
    for target_pos in range(SEQ_LEN, len(grid)):
        if not np.isfinite(yall[target_pos]):
            continue
        # Only evaluate/model target hours for which all five inverters reported.
        # Structural night zeros remain context, not targets.
        if grid["observed"].iloc[target_pos] != 1 or yall[target_pos] == 0:
            continue
        xs.append(xall[target_pos-SEQ_LEN:target_pos])
        ys.append(yall[target_pos] / CAPACITY)
        times.append(grid.index[target_pos])
    return np.stack(xs), np.asarray(ys, np.float32), pd.DatetimeIndex(times)


class SeqRegressor(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        cls = nn.RNN if kind == "RNN" else nn.LSTM
        self.seq = cls(input_size=6, hidden_size=32, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.seq(x)
        return self.head(out[:, -1]).squeeze(-1)


def score(y, pred):
    pred = np.clip(pred, 0, CAPACITY)
    err = pred - y
    denom = np.abs(y) + np.abs(pred)
    return {
        "n": int(len(y)), "ME_kW": float(err.mean()),
        "MAE_kW": float(mean_absolute_error(y, pred)),
        "MSE_kW2": float(mean_squared_error(y, pred)),
        "RMSE_kW": float(math.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)), "Pearson_r": float(np.corrcoef(y, pred)[0, 1]),
        "NMAE_capacity_pct": float(mean_absolute_error(y, pred)/CAPACITY*100),
        "NRMSE_capacity_pct": float(math.sqrt(mean_squared_error(y, pred))/CAPACITY*100),
        "WAPE_pct": float(np.abs(err).sum()/np.abs(y).sum()*100),
        "sMAPE_pct": float(np.mean(np.divide(2*np.abs(err), denom, out=np.zeros_like(err), where=denom>1e-9))*100),
    }


def train_model(kind, x_train, y_train, x_valid, y_valid, x_test):
    model = SeqRegressor(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=128, shuffle=False)
    xv, yv = torch.from_numpy(x_valid), torch.from_numpy(y_valid)
    best, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(40):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(model(xv), yv).item()
        history.append({"epoch": epoch+1, "train_mse": float(np.mean(losses)), "valid_mse": valid_loss})
        if valid_loss < best - 1e-6:
            best, stale = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test)).numpy() * CAPACITY
    return pred, history


def main():
    grid = regular_series()
    x, y, times = sequences(grid)
    n = len(y)
    a, b = int(n*0.70), int(n*0.85)
    splits = (x[:a], y[:a], x[a:b], y[a:b], x[b:], y[b:], times[b:])
    xtr, ytr, xv, yv, xt, yt, tt = splits
    all_metrics, pred_table, histories = {}, pd.DataFrame({"timestamp": tt, "actual_kw": yt*CAPACITY}), {}
    for kind in ("RNN", "LSTM"):
        pred, history = train_model(kind, xtr, ytr, xv, yv, xt)
        all_metrics[kind] = score(yt*CAPACITY, pred)
        pred_table[kind] = pred
        histories[kind] = history
    out = ROOT / "outputs"
    pred_table.to_csv(out / "preliminary_neural_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "preliminary_no_weather", "sequence_hours": SEQ_LEN,
        "architecture": "2-layer hidden-size-32 sequence model; early stopping on chronological validation",
        "target": "next observed complete five-inverter hourly plant output",
        "limitations": ["No KMA weather inputs", "Night zeros used only as sequence context",
                        "Missing daytime plant values encoded as zero plus an observation mask"],
        "metrics": all_metrics, "training_history": histories,
    }
    (out / "preliminary_neural_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(all_metrics).T[["MAE_kW", "RMSE_kW", "R2", "Pearson_r"]].to_string())


if __name__ == "__main__":
    main()
