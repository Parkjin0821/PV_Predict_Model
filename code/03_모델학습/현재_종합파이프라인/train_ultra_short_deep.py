"""초단기 13개 시점을 동시에 출력하는 LSTM·GRU 시간순 교차검증."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model_common import regression_metrics, write_json
from pv_pipeline import OUTPUT as DATA_OUTPUT
from pv_pipeline import ROOT, load_config, solar_position


MODEL_OUTPUT = ROOT / "outputs" / "초단기_딥러닝"
torch.set_num_threads(4)


@dataclass
class Prepared:
    history_values: np.ndarray
    positions: np.ndarray
    issue_times: pd.DatetimeIndex
    future_values: np.ndarray
    targets: np.ndarray
    daylight: np.ndarray
    target_times: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, prepared: Prepared, indices: np.ndarray, history_length: int):
        self.prepared = prepared
        self.indices = np.asarray(indices, dtype=int)
        self.history_length = history_length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample = self.indices[item]
        position = int(self.prepared.positions[sample])
        history = self.prepared.history_values[position - self.history_length : position]
        return (
            torch.from_numpy(history),
            torch.from_numpy(self.prepared.future_values[sample]),
            torch.from_numpy(self.prepared.targets[sample]),
        )


class MultiHorizonSequenceModel(nn.Module):
    def __init__(self, kind: str, history_features: int, future_features: int, hidden_size: int = 40):
        super().__init__()
        recurrent = nn.LSTM if kind == "장단기기억모델" else nn.GRU
        self.encoder = recurrent(history_features, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + future_features, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 1),
        )

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(history)
        context = encoded[:, -1].unsqueeze(1).expand(-1, future.shape[1], -1)
        return self.head(torch.cat([context, future], dim=-1)).squeeze(-1)


def prepare(config: dict) -> Prepared:
    five = pd.read_parquet(DATA_OUTPUT / "정제_5분_기준자료.parquet")
    forecast = pd.read_parquet(DATA_OUTPUT / "기상청_650일_5분확장예보.parquet")
    capacity = float(config["site"]["capacity_kw"])
    index = five.index
    power = five["발전출력_kW"]
    target_mean = power.rolling(3, min_periods=3).mean().shift(-2)

    minute = index.hour * 60 + index.minute
    history = np.column_stack(
        [
            power.fillna(0).to_numpy(float) / capacity,
            power.notna().to_numpy(float),
            five["입력전력_kW"].fillna(0).to_numpy(float) / capacity,
            five["인버터평균온도_C"].fillna(0).to_numpy(float) / 80,
            five["인버터평균온도_C"].notna().to_numpy(float),
            five["통신정상비율"].fillna(0).to_numpy(float),
            np.clip(five["태양고도_deg"].to_numpy(float) / 90, -1, 1),
            np.sin(2 * np.pi * minute / 1440),
            np.cos(2 * np.pi * minute / 1440),
        ]
    ).astype("float32")

    horizons = np.asarray(config["ultra_short"]["horizon_minutes"], dtype=int)
    steps = horizons // 5
    history_length = int(config["ultra_short"]["sequence_history_minutes"] // 5)
    start_position = history_length
    end_position = len(index) - int(steps.max()) - 2
    latitude = float(config["site"]["latitude"])
    longitude = float(config["site"]["longitude"])

    positions = np.arange(start_position, end_position, dtype=int)
    issues = pd.DatetimeIndex(index[positions])
    target_time_rows = issues.to_numpy()[:, None] + horizons.astype("timedelta64[m]")[None, :]
    flat_target_times = pd.DatetimeIndex(target_time_rows.reshape(-1))
    weather = forecast.reindex(flat_target_times)
    weather_columns = ["기온예보_C", "하늘상태예보", "습도예보_pct", "예보운량_10분율"]
    weather_values = weather[weather_columns].to_numpy(float).reshape(len(positions), len(horizons), -1)
    weather_issue = pd.to_datetime(weather["예보발표시각"]).to_numpy().reshape(len(positions), len(horizons))
    target_values = target_mean.to_numpy(float)[positions[:, None] + steps[None, :]]
    elevation, azimuth = solar_position(flat_target_times, latitude, longitude)
    elevation = elevation.reshape(len(positions), len(horizons))
    azimuth = azimuth.reshape(len(positions), len(horizons))
    daylight_rows = elevation > 0
    target_minute = (flat_target_times.hour * 60 + flat_target_times.minute).to_numpy().reshape(
        len(positions), len(horizons)
    )
    target_dayofyear = flat_target_times.dayofyear.to_numpy().reshape(len(positions), len(horizons))
    valid = (
        np.isfinite(weather_values).all(axis=(1, 2))
        & (weather_issue <= issues.to_numpy()[:, None]).all(axis=1)
        & np.isfinite(target_values).all(axis=1)
        & daylight_rows.any(axis=1)
    )
    positions = positions[valid]
    issues = issues[valid]
    target_time_rows = target_time_rows[valid]
    weather_values = weather_values[valid]
    target_values = target_values[valid]
    elevation = elevation[valid]
    azimuth = azimuth[valid]
    daylight_rows = daylight_rows[valid]
    target_minute = target_minute[valid]
    target_dayofyear = target_dayofyear[valid]
    future_rows = np.stack(
        [
            (weather_values[:, :, 0] + 20) / 60,
            weather_values[:, :, 2] / 100,
            weather_values[:, :, 1] / 4,
            weather_values[:, :, 3] / 10,
            np.clip(elevation / 90, -1, 1),
            np.sin(np.deg2rad(azimuth)),
            np.cos(np.deg2rad(azimuth)),
            np.sin(2 * np.pi * target_minute / 1440),
            np.cos(2 * np.pi * target_minute / 1440),
            np.sin(2 * np.pi * target_dayofyear / 365.25),
            np.cos(2 * np.pi * target_dayofyear / 365.25),
        ],
        axis=2,
    ).astype("float32")

    return Prepared(
        history_values=history,
        positions=positions,
        issue_times=issues,
        future_values=future_rows,
        targets=(target_values / capacity).astype("float32"),
        daylight=daylight_rows,
        target_times=target_time_rows,
    )


def train_model(
    kind: str,
    prepared: Prepared,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    history_length: int,
    seed: int,
) -> tuple[MultiHorizonSequenceModel, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiHorizonSequenceModel(
        kind,
        history_features=prepared.history_values.shape[1],
        future_features=prepared.future_values.shape[2],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    # 15분 간격으로 학습표본을 얇게 하되 검증·운영예측은 모든 5분 발행시각을 사용한다.
    train_indices = train_indices[::3]
    train_loader = DataLoader(
        SequenceDataset(prepared, train_indices, history_length),
        batch_size=512,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        SequenceDataset(prepared, valid_indices, history_length),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )
    best_loss, best_state, stale, history_log = float("inf"), None, 0, []
    for epoch in range(12):
        model.train()
        train_losses = []
        for history, future, target in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(history, future), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for history, future, target in valid_loader:
                valid_losses.append(float(loss_function(model(history, future), target).item()))
        valid_loss = float(np.mean(valid_losses))
        history_log.append(
            {"시대": epoch + 1, "학습손실": float(np.mean(train_losses)), "검증손실": valid_loss}
        )
        if valid_loss < best_loss - 1e-6:
            best_loss, stale = valid_loss, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError("딥러닝 최적 상태가 저장되지 않았습니다.")
    model.load_state_dict(best_state)
    return model, history_log


def predict_model(
    model: MultiHorizonSequenceModel,
    prepared: Prepared,
    indices: np.ndarray,
    history_length: int,
    capacity: float,
) -> np.ndarray:
    loader = DataLoader(
        SequenceDataset(prepared, indices, history_length),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )
    values = []
    model.eval()
    with torch.no_grad():
        for history, future, _ in loader:
            values.append(model(history, future).numpy())
    prediction = np.clip(np.concatenate(values, axis=0) * capacity, 0, capacity)
    prediction[~prepared.daylight[indices]] = 0.0
    return prediction


def long_predictions(
    prepared: Prepared,
    indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    horizons: list[int],
    capacity: float,
    fold_name: str | None = None,
) -> pd.DataFrame:
    rows = []
    actual = prepared.targets[indices] * capacity
    for row_number, sample in enumerate(indices):
        for horizon_number, horizon in enumerate(horizons):
            if not prepared.daylight[sample, horizon_number]:
                continue
            row = {
                "예측발행시각": prepared.issue_times[sample],
                "예측대상시각": pd.Timestamp(prepared.target_times[sample, horizon_number]),
                "예측수평_분": int(horizon),
                "실제_15분평균출력_kW": float(actual[row_number, horizon_number]),
            }
            for name, matrix in predictions.items():
                row[f"{name}예측_kW"] = float(matrix[row_number, horizon_number])
            if fold_name is not None:
                row["검증구간"] = fold_name
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    capacity = float(config["site"]["capacity_kw"])
    history_length = int(config["ultra_short"]["sequence_history_minutes"] // 5)
    horizons = [int(value) for value in config["ultra_short"]["horizon_minutes"]]
    prepared = prepare(config)
    seed = int(config["random_seed"])
    test_start = pd.Timestamp(config["final_test"]["start"])
    test_end = pd.Timestamp(config["final_test"]["end"])

    oof_parts, histories = [], {}
    max_targets = pd.to_datetime(prepared.target_times[:, -1])
    min_targets = pd.to_datetime(prepared.target_times[:, 0])
    for fold_number, window in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        train_indices = np.flatnonzero(max_targets < start)
        valid_indices = np.flatnonzero((min_targets >= start) & (max_targets <= end))
        if len(train_indices) == 0 or len(valid_indices) == 0:
            continue
        fold_predictions, fold_history = {}, {}
        for kind in ["장단기기억모델", "게이트순환모델"]:
            print(f"{kind} 검증{fold_number}: 학습 {len(train_indices):,}, 검증 {len(valid_indices):,}")
            model, history = train_model(kind, prepared, train_indices, valid_indices, history_length, seed)
            fold_predictions[kind] = predict_model(model, prepared, valid_indices, history_length, capacity)
            fold_history[kind] = history
        oof_parts.append(
            long_predictions(
                prepared,
                valid_indices,
                fold_predictions,
                horizons,
                capacity,
                fold_name=f"검증{fold_number}",
            )
        )
        histories[f"검증{fold_number}"] = fold_history

    train_indices = np.flatnonzero(max_targets < test_start)
    test_indices = np.flatnonzero((min_targets >= test_start) & (max_targets <= test_end))
    test_predictions, final_history = {}, {}
    for kind in ["장단기기억모델", "게이트순환모델"]:
        print(f"{kind} 최종: 학습 {len(train_indices):,}, 시험 {len(test_indices):,}")
        model, history = train_model(kind, prepared, train_indices, test_indices, history_length, seed)
        test_predictions[kind] = predict_model(model, prepared, test_indices, history_length, capacity)
        final_history[kind] = history
        torch.save(
            {
                "model_state": model.state_dict(),
                "kind": kind,
                "history_features": prepared.history_values.shape[1],
                "future_features": prepared.future_values.shape[2],
                "history_length": history_length,
                "horizons_minutes": horizons,
                "capacity_kw": capacity,
            },
            MODEL_OUTPUT / f"{kind}.pt",
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    test = long_predictions(prepared, test_indices, test_predictions, horizons, capacity)
    oof.to_parquet(MODEL_OUTPUT / "교차검증_비표본예측.parquet")
    test.to_parquet(MODEL_OUTPUT / "최종시험_예측.parquet")
    score_rows = []
    for horizon in horizons:
        part = test[test["예측수평_분"].eq(horizon)]
        for kind in ["장단기기억모델", "게이트순환모델"]:
            score_rows.append(
                {
                    "예측수평_분": horizon,
                    "모델": kind,
                    **regression_metrics(part["실제_15분평균출력_kW"], part[f"{kind}예측_kW"]),
                }
            )
    pd.DataFrame(score_rows).to_csv(MODEL_OUTPUT / "최종시험_성능표.csv", index=False, encoding="utf-8-sig")
    histories["최종"] = final_history
    write_json(MODEL_OUTPUT / "학습이력.json", histories)
    write_json(
        MODEL_OUTPUT / "자료정보.json",
        {
            "전체사용가능표본": int(len(prepared.positions)),
            "최종학습표본": int(len(train_indices)),
            "최종시험표본": int(len(test_indices)),
            "학습표본간격": "15분(메모리·시간 절감), 검증과 운영출력은 5분",
            "출력수": len(horizons),
        },
    )
    print(pd.DataFrame(score_rows).to_string(index=False))


if __name__ == "__main__":
    main()
