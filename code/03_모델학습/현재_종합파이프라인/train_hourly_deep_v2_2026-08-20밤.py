# -*- coding: utf-8 -*-
"""1시간 +1~48시간 LSTM·GRU — 08-20 밤 전면 재작업(공식 KMA 자료로 교체).

## 왜 다시 만드는가
기존 `train_hourly_deep.py`는 `open_meteo_forecast_*`(비공식, 08-18 확정
기준상 공식 학습·검증에서 배제) 컬럼에 의존했다. 이 스크립트는 그 자리를
전부 08-20 밤 확정 **공식 KMA 파이프라인**(정렬 v3, 최종 특성 14개)으로
교체한다.

## 데이터·정렬 규칙 (LightGBM 6번과 동일 원칙 재사용)
- 인코더(history) 채널: **직전까지 완결된 값**만 쓴다 — 슬라이스 자체가
  `[t-history_length, t)`로 현재 행(t)을 제외하므로, 각 시각 τ(<t)의
  원본 raw 값을 그대로 써도 안전하다(그 시각에 이미 완결된 정보).
- 디코더(future, 수평별) 채널: FINAL_14 중 예보(FORECAST) 성격 8개를
  `shift(-h)`로 목표시각에 맞춰 넣는다(v3와 동일 회전 로직).
- **타깃 인덱싱 수정(중요)**: 기존 스크립트는 수평 h의 타깃을 `position+h`
  행으로 잡았는데, 이건 08-20 밤에 확정한 "h=1의 타깃은 shift 없는 그
  행 자체"라는 결론과 어긋난다(1시간 어긋나 있었음). 이번엔
  `position+(h-1)` 행으로 수정했다 — LightGBM 6번 스크립트와 정확히
  같은 관례.
- 특성은 **08-20 밤 최종 확정 14개만** 쓴다(LightGBM과 동일 거버넌스 —
  트리모델엔 특성선택을 적용하고 딥러닝엔 임의로 더 넣는 불일치를
  피함).
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v2_2026-08-20밤.csv"
)
OUT = ROOT / "outputs" / "6번_LSTM_GRU_v2_2026-08-20밤"
torch.set_num_threads(4)

# FINAL_14를 성격별로 분류(v3 스크립트와 동일 근거)
HISTORY_STARTLABEL = ["plant_input_power_kw", "mean_power_factor", "mean_input_voltage_v"]
HISTORY_OBSERVED = ["기상청관측_일조시간_hr", "기상청관측_상대습도_pct", "기상청관측_전운량_pct"]
FUTURE_FORECAST = ["DSWRF", "REH", "추정_출력온도", "DSWRFLX_bsrn정제", "LCDC", "TCDC", "POP", "SKY"]
# 참고: FINAL_14 = HISTORY_STARTLABEL(3) + HISTORY_OBSERVED(3) + FUTURE_FORECAST(8) = 14개 전부 포함

NORM = {  # 대략적 정규화 상수(문헌·데이터 범위 기반, 표준화 목적일 뿐 정밀 스케일러 아님)
    "plant_input_power_kw": 240.0, "mean_power_factor": 100.0, "mean_input_voltage_v": 500.0,
    "기상청관측_일조시간_hr": 1.0, "기상청관측_상대습도_pct": 100.0, "기상청관측_전운량_pct": 100.0,
    "DSWRF": 1000.0, "REH": 100.0, "추정_출력온도": 60.0, "DSWRFLX_bsrn정제": 1000.0,
    "LCDC": 1.0, "TCDC": 1.0, "POP": 100.0, "SKY": 4.0,
}


@dataclass
class Prepared:
    history: np.ndarray
    positions: np.ndarray
    issue_times: pd.DatetimeIndex
    future: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    daylight: np.ndarray
    target_times: np.ndarray


class HourlyDataset(Dataset):
    def __init__(self, data: Prepared, indices: np.ndarray, history_length: int):
        self.data, self.indices, self.history_length = data, np.asarray(indices, int), history_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        sample = self.indices[item]
        position = self.data.positions[sample]
        return (
            torch.from_numpy(self.data.history[position - self.history_length: position]),
            torch.from_numpy(self.data.future[sample]),
            torch.from_numpy(self.data.targets[sample]),
            torch.from_numpy(self.data.target_mask[sample].astype("float32")),
        )


class HourlyModel(nn.Module):
    def __init__(self, kind: str, history_features: int, future_features: int):
        super().__init__()
        recurrent = nn.LSTM if kind == "장단기기억모델" else nn.GRU
        self.encoder = recurrent(history_features, 48, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48 + future_features, 56), nn.ReLU(), nn.Dropout(0.1), nn.Linear(56, 1))

    def forward(self, history, future):
        encoded, _ = self.encoder(history)
        context = encoded[:, -1].unsqueeze(1).expand(-1, future.shape[1], -1)
        return self.head(torch.cat([context, future], dim=-1)).squeeze(-1)


def load_data(config: dict) -> Prepared:
    frame = pd.read_csv(DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    capacity = float(config["site"]["capacity_kw"])
    index = frame.index
    power = frame["plant_output_kw"]
    elevation_now = frame["solar_elevation_deg"].to_numpy(float)
    minute = index.hour * 60 + index.minute

    history_cols = [
        power.fillna(0).to_numpy(float) / capacity,
        power.notna().to_numpy(float),
        np.clip(elevation_now / 90, -1, 1),
        np.sin(2 * np.pi * minute / 1440),
        np.cos(2 * np.pi * minute / 1440),
    ]
    for c in HISTORY_STARTLABEL + HISTORY_OBSERVED:
        history_cols.append(frame[c].to_numpy(float) / NORM[c])
        history_cols.append(frame[c].notna().to_numpy(float))
    history = np.column_stack(history_cols).astype("float32")
    history[~np.isfinite(history)] = 0

    horizons = np.arange(1, 49, dtype=int)
    history_length = int(config["hourly"]["sequence_history_hours"])
    positions = np.arange(history_length, len(frame) - 48, dtype=int)
    issue_times = pd.DatetimeIndex(index[positions])

    # 08-20 밤 정렬 수정: 수평 h의 타깃/미래공변량은 (h-1)만큼 미래 행
    offsets = horizons - 1
    target_positions = positions[:, None] + offsets[None, :]
    target_times = issue_times.to_numpy()[:, None] + offsets.astype("timedelta64[h]")[None, :]

    targets_raw = power.to_numpy(float)[target_positions]
    target_mask = np.isfinite(targets_raw)
    targets = np.nan_to_num(targets_raw, nan=0.0) / capacity

    target_elev = elevation_now[target_positions]
    daylight = target_elev > 0
    flat_target = pd.DatetimeIndex(target_times.reshape(-1))
    target_azimuth = frame["solar_azimuth_deg"].to_numpy(float)[target_positions]
    target_minute = (flat_target.hour * 60 + flat_target.minute).to_numpy().reshape(len(positions), 48)
    target_doy = flat_target.dayofyear.to_numpy().reshape(len(positions), 48)

    future_arrays = [
        np.clip(target_elev / 90, -1, 1),
        np.sin(np.deg2rad(target_azimuth)),
        np.cos(np.deg2rad(target_azimuth)),
        np.sin(2 * np.pi * target_minute / 1440),
        np.cos(2 * np.pi * target_minute / 1440),
        np.sin(2 * np.pi * target_doy / 365.25),
        np.cos(2 * np.pi * target_doy / 365.25),
    ]
    for c in FUTURE_FORECAST:
        col = frame[c].to_numpy(float)
        # shift(-h): 행 position+h (예보 종료라벨이 목표버킷[target=position+h-1]과 일치)
        fetch_positions = positions[:, None] + horizons[None, :]
        values = col[fetch_positions] / NORM[c]
        future_arrays.append(np.nan_to_num(values, nan=0.0))
        future_arrays.append(np.isfinite(col[fetch_positions]).astype(float))
    future = np.stack(future_arrays, axis=2).astype("float32")

    valid = (
        np.isfinite(future).all(axis=(1, 2))
        & (target_mask.mean(axis=1) >= 0.5)
        & (daylight & target_mask).any(axis=1)
    )
    return Prepared(
        history=history, positions=positions[valid], issue_times=issue_times[valid],
        future=future[valid], targets=targets[valid].astype("float32"),
        target_mask=target_mask[valid], daylight=daylight[valid], target_times=target_times[valid],
    )


def masked_loss(prediction, target, mask):
    error = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none") * mask
    return error.sum() / mask.sum().clamp_min(1)


def train_model(kind, data, train_indices, valid_indices, history_length, seed, epochs=14):
    torch.manual_seed(seed)
    model = HourlyModel(kind, data.history.shape[1], data.future.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(HourlyDataset(data, train_indices, history_length), batch_size=256, shuffle=True)
    valid_loader = DataLoader(HourlyDataset(data, valid_indices, history_length), batch_size=512, shuffle=False)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for history, future, target, mask in train_loader:
            optimizer.zero_grad()
            loss = masked_loss(model(history, future), target, mask)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval(); valid_losses = []
        with torch.no_grad():
            for history, future, target, mask in valid_loader:
                valid_losses.append(float(masked_loss(model(history, future), target, mask).item()))
        value = float(np.mean(valid_losses))
        if value < best - 1e-6:
            best, stale = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, data, indices, history_length, capacity):
    loader = DataLoader(HourlyDataset(data, indices, history_length), batch_size=512, shuffle=False)
    output = []
    model.eval()
    with torch.no_grad():
        for history, future, _, _ in loader:
            output.append(model(history, future).numpy())
    values = np.clip(np.concatenate(output) * capacity, 0, capacity)
    values[~data.daylight[indices]] = 0
    return values


def metrics(y, p):
    e = y - p
    return {"n": int(len(y)), "MAE_kW": round(float(np.abs(e).mean()), 3),
            "RMSE_kW": round(float(np.sqrt((e ** 2).mean())), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    history_length = int(config["hourly"]["sequence_history_hours"])

    print("데이터 준비 중...")
    data = load_data(config)
    print(f"유효 표본수: {len(data.positions)}")

    min_targets = pd.to_datetime(data.target_times[:, 0])
    max_targets = pd.to_datetime(data.target_times[:, -1])

    fold_scores = []
    for number, window in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        train_idx = np.flatnonzero(max_targets < start)
        valid_idx = np.flatnonzero((min_targets >= start) & (max_targets <= end))
        if len(train_idx) < 200 or len(valid_idx) < 30:
            print(f"[검증{number}] 표본 부족으로 건너뜀")
            continue
        print(f"\n=== 검증{number}({window.get('_계절','')}): 학습 {len(train_idx):,}, 검증 {len(valid_idx):,} ===")
        for kind in ["장단기기억모델", "게이트순환모델"]:
            model = train_model(kind, data, train_idx, valid_idx, history_length, seed)
            pred = predict(model, data, valid_idx, history_length, capacity)
            actual = data.targets[valid_idx] * capacity
            mask = data.target_mask[valid_idx] & data.daylight[valid_idx]
            for h in [1, 24, 48]:
                hi = h - 1
                y, p = actual[:, hi][mask[:, hi]], pred[:, hi][mask[:, hi]]
                if len(y) < 10:
                    continue
                m = metrics(y, p)
                m.update({"검증구간": f"검증{number}", "모델": kind, "수평_h": h})
                fold_scores.append(m)
                print(f"  {kind} +{h}h: MAE={m['MAE_kW']:.2f} RMSE={m['RMSE_kW']:.2f} (n={m['n']})")

    scores_df = pd.DataFrame(fold_scores)
    scores_df.to_csv(OUT / "폴드별_성능.csv", index=False, encoding="utf-8-sig")
    if len(scores_df):
        summary = scores_df.groupby(["모델", "수평_h"])[["MAE_kW", "RMSE_kW"]].mean().round(3)
        print("\n=== 모델·수평별 평균(전체 폴드) ===")
        print(summary.to_string())
        summary.to_csv(OUT / "모델수평별_평균.csv", encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
