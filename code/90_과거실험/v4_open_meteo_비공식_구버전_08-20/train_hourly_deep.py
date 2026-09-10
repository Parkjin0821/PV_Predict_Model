"""1시간 +1~48시간 다중출력 LSTM·GRU 시간순 교차검증."""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model_common import regression_metrics, write_json
from pv_pipeline import ROOT, load_config, solar_position


OUT = ROOT / "outputs" / "1시간_딥러닝"
KMA_MODEL_DATA = Path(r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\gwangju_1hour_model_dataset_kma_observed.csv")  # 08-20 경로 정비: 원래 존재하지 않던 pv_environment_data/processed 경로를 실제 위치로 교정. 주의: 이 파일은 여전히 open_meteo_forecast_* 컬럼(비공식)에 의존하므로 공식 결과로 쓰려면 이 스크립트의 특성 로직을 build_official_hourly_dataset_v1.py 산출물(같은 폴더의 processed/gwangju_1hour_model_dataset_official_v1_2026-08-20.csv, Open-Meteo 없음) 기준으로 다시 짜야 한다 — 5/6번 단계 작업.
torch.set_num_threads(4)
WEATHER = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]


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
            torch.from_numpy(self.data.history[position - self.history_length : position]),
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
    frame = pd.read_csv(KMA_MODEL_DATA, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    capacity = float(config["site"]["capacity_kw"])
    index = frame.index
    power = frame["plant_output_kw"]
    minute = index.hour * 60 + index.minute
    history = np.column_stack(
        [
            power.fillna(0).to_numpy(float) / capacity,
            power.notna().to_numpy(float),
            frame["plant_input_power_kw"].fillna(0).to_numpy(float) / capacity,
            frame["mean_inverter_temperature_c"].fillna(0).to_numpy(float) / 80,
            frame["mean_inverter_temperature_c"].notna().to_numpy(float),
            (frame["기상청관측_기온_C"].to_numpy(float) + 20) / 60,
            frame["기상청관측_상대습도_pct"].to_numpy(float) / 100,
            frame["기상청관측_강수량_mm"].to_numpy(float) / 20,
            frame["기상청관측_전운량_pct"].to_numpy(float) / 100,
            frame["기상청관측_일사량_W_m2"].to_numpy(float) / 1000,
            frame["기상청관측_일조시간_hr"].to_numpy(float),
            frame["기상청관측_풍속_m_s"].to_numpy(float) / 20,
            (frame["기상청관측_현지기압_hPa"].to_numpy(float) - 900) / 200,
            frame["기상청관측_적설_cm"].to_numpy(float) / 10,
            (frame["기상청관측_지면온도_C"].to_numpy(float) + 20) / 80,
            np.clip(frame["solar_elevation_deg"].to_numpy(float) / 90, -1, 1),
            np.sin(2 * np.pi * minute / 1440),
            np.cos(2 * np.pi * minute / 1440),
        ]
    ).astype("float32")
    history[~np.isfinite(history)] = 0

    horizons = np.arange(1, 49, dtype=int)
    history_length = int(config["hourly"]["sequence_history_hours"])
    positions = np.arange(history_length, len(frame) - 48, dtype=int)
    issue_times = pd.DatetimeIndex(index[positions])
    target_times = issue_times.to_numpy()[:, None] + horizons.astype("timedelta64[h]")[None, :]
    target_positions = positions[:, None] + horizons[None, :]
    targets = power.to_numpy(float)[target_positions]
    target_mask = np.isfinite(targets)
    targets = np.nan_to_num(targets, nan=0.0) / capacity
    flat_target = pd.DatetimeIndex(target_times.reshape(-1))
    elevation, azimuth = solar_position(
        flat_target,
        float(config["site"]["latitude"]),
        float(config["site"]["longitude"]),
    )
    elevation = elevation.reshape(len(positions), 48)
    azimuth = azimuth.reshape(len(positions), 48)
    daylight = elevation > 0
    target_minute = (flat_target.hour * 60 + flat_target.minute).to_numpy().reshape(len(positions), 48)
    target_doy = flat_target.dayofyear.to_numpy().reshape(len(positions), 48)

    weather_arrays = []
    for name in WEATHER:
        values = []
        for horizon in horizons:
            day = 1 if horizon <= 24 else 2
            source = frame[f"open_meteo_forecast_{name}_previous_day{day}"].to_numpy(float)
            values.append(source[positions + horizon])
        weather_arrays.append(np.column_stack(values))
    temperature, humidity, precipitation, cloud, shortwave, direct, diffuse, wind, direction, pressure = weather_arrays
    future = np.stack(
        [
            (temperature + 20) / 60,
            humidity / 100,
            precipitation / 20,
            cloud / 100,
            shortwave / 1000,
            direct / 1000,
            diffuse / 1000,
            wind / 20,
            np.sin(np.deg2rad(direction)),
            np.cos(np.deg2rad(direction)),
            (pressure - 900) / 200,
            np.clip(elevation / 90, -1, 1),
            np.sin(np.deg2rad(azimuth)),
            np.cos(np.deg2rad(azimuth)),
            np.sin(2 * np.pi * target_minute / 1440),
            np.cos(2 * np.pi * target_minute / 1440),
            np.sin(2 * np.pi * target_doy / 365.25),
            np.cos(2 * np.pi * target_doy / 365.25),
        ],
        axis=2,
    ).astype("float32")
    valid = (
        np.isfinite(future).all(axis=(1, 2))
        & (target_mask.mean(axis=1) >= 0.8)
        & (daylight & target_mask).any(axis=1)
    )
    return Prepared(
        history=history,
        positions=positions[valid],
        issue_times=issue_times[valid],
        future=future[valid],
        targets=targets[valid].astype("float32"),
        target_mask=target_mask[valid],
        daylight=daylight[valid],
        target_times=target_times[valid],
    )


def masked_loss(prediction, target, mask):
    error = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none") * mask
    return error.sum() / mask.sum().clamp_min(1)


def train(kind, data, train_indices, valid_indices, history_length, seed):
    torch.manual_seed(seed)
    model = HourlyModel(kind, data.history.shape[1], data.future.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(HourlyDataset(data, train_indices, history_length), batch_size=256, shuffle=True)
    valid_loader = DataLoader(HourlyDataset(data, valid_indices, history_length), batch_size=512, shuffle=False)
    best, best_state, stale, log = float("inf"), None, 0, []
    for epoch in range(18):
        model.train(); train_losses = []
        for history, future, target, mask in train_loader:
            optimizer.zero_grad(); loss = masked_loss(model(history, future), target, mask)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval(); valid_losses = []
        with torch.no_grad():
            for history, future, target, mask in valid_loader:
                valid_losses.append(float(masked_loss(model(history, future), target, mask).item()))
        value = float(np.mean(valid_losses)); log.append({"시대":epoch+1,"학습손실":float(np.mean(train_losses)),"검증손실":value})
        if value < best - 1e-6:
            best, stale = value, 0; best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4: break
    model.load_state_dict(best_state)
    return model, log


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


def to_long(data, indices, predictions, capacity, fold=None):
    rows=[]; actual=data.targets[indices]*capacity
    for row_num,sample in enumerate(indices):
        for horizon in range(1,49):
            h=horizon-1
            if not data.target_mask[sample,h] or not data.daylight[sample,h]: continue
            row={"예측발행시각":data.issue_times[sample],"예측대상시각":pd.Timestamp(data.target_times[sample,h]),"예측수평_시간":horizon,"실제_1시간평균출력_kW":float(actual[row_num,h])}
            for name,matrix in predictions.items(): row[f"{name}예측_kW"]=float(matrix[row_num,h])
            if fold: row["검증구간"]=fold
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True); config=load_config(); data=load_data(config)
    capacity=float(config["site"]["capacity_kw"]); seed=int(config["random_seed"]); history_length=int(config["hourly"]["sequence_history_hours"])
    min_targets=pd.to_datetime(data.target_times[:,0]); max_targets=pd.to_datetime(data.target_times[:,-1])
    oof_parts=[]; histories={}
    for number,window in enumerate(config["cross_validation_windows"],start=1):
        start,end=pd.Timestamp(window["start"]),pd.Timestamp(window["end"])
        train_idx=np.flatnonzero(max_targets<start); valid_idx=np.flatnonzero((min_targets>=start)&(max_targets<=end))
        if not len(train_idx) or not len(valid_idx): continue
        preds={}; fold_log={}
        for kind in ["장단기기억모델","게이트순환모델"]:
            print(f"1시간 {kind} 검증{number}: 학습 {len(train_idx):,}, 검증 {len(valid_idx):,}")
            model,log=train(kind,data,train_idx,valid_idx,history_length,seed); preds[kind]=predict(model,data,valid_idx,history_length,capacity); fold_log[kind]=log
        oof_parts.append(to_long(data,valid_idx,preds,capacity,f"검증{number}")); histories[f"검증{number}"]=fold_log
    test_start,test_end=pd.Timestamp(config["final_test"]["start"]),pd.Timestamp(config["final_test"]["end"])
    train_idx=np.flatnonzero(max_targets<test_start); test_idx=np.flatnonzero((min_targets>=test_start)&(max_targets<=test_end))
    preds={}; final_log={}
    for kind in ["장단기기억모델","게이트순환모델"]:
        print(f"1시간 {kind} 최종: 학습 {len(train_idx):,}, 시험 {len(test_idx):,}")
        model,log=train(kind,data,train_idx,test_idx,history_length,seed); preds[kind]=predict(model,data,test_idx,history_length,capacity); final_log[kind]=log
        torch.save({"model_state":model.state_dict(),"kind":kind,"history_features":data.history.shape[1],"future_features":data.future.shape[2],"history_length":history_length},OUT/f"{kind}.pt")
    oof=pd.concat(oof_parts,ignore_index=True); test=to_long(data,test_idx,preds,capacity)
    oof.to_parquet(OUT/"교차검증_비표본예측.parquet"); test.to_parquet(OUT/"최종시험_예측.parquet")
    scores=[]
    for horizon in range(24,49):
        part=test[test["예측수평_시간"].eq(horizon)]
        for kind in ["장단기기억모델","게이트순환모델"]: scores.append({"예측수평_시간":horizon,"모델":kind,**regression_metrics(part["실제_1시간평균출력_kW"],part[f"{kind}예측_kW"])})
    pd.DataFrame(scores).to_csv(OUT/"공식24_48시간_성능표.csv",index=False,encoding="utf-8-sig")
    histories["최종"]=final_log; write_json(OUT/"학습이력.json",histories); write_json(OUT/"자료정보.json",{"전체표본":len(data.positions),"최종학습":len(train_idx),"최종시험":len(test_idx)})
    print(pd.DataFrame(scores).to_string(index=False))


if __name__=="__main__": main()
