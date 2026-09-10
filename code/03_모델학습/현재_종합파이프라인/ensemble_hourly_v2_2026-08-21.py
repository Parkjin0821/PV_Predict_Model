# -*- coding: utf-8 -*-
"""+24h·+48h 3자 앙상블 v2 — v1의 "LSTM 불공정 핸디캡" 수정판.

## v1에서 발견한 결함 (08-22 재검사, 사용자 요청)
v1(`ensemble_hourly_v1_2026-08-21.py`)은 조기종료 누출은 제대로 고쳤지만,
**시험 예측을 만드는 학습량이 모델마다 달랐다**:
- 트리모델(LightGBM/XGBoost): 학습구간 **100%**로 학습한 모델로 시험 예측
- LSTM: 학습구간 **80%(inner_tr)**로만 학습한 모델로 시험 예측

즉 LSTM만 20% 적은 데이터로 싸웠다. v1의 "LSTM 21.4 vs XGBoost 20.1"은
**정직한 조기종료 효과 + 학습량 핸디캡이 뒤섞인 숫자**라 순수 비교가
아니었다.

## v2의 수정: 2단계 학습(표준 관행)
1. **에폭 수 결정**: inner_tr(80%)로 학습하고 inner_va(20%)로 조기종료해
   **최적 에폭 수**를 찾는다(시험셋은 절대 안 봄 — 누출 방지 유지).
2. **최종 재학습**: 그 에폭 수만큼 **학습구간 100%**로 다시 학습해
   시험 예측을 만든다 → 트리모델과 학습량이 동일해진다.

앙상블 가중치 산출용 inner_va 예측은 1단계 모델(80% 학습)에서 그대로
쓴다 — 트리모델의 inner 예측도 inner_tr(80%) 학습분이라 여기선 조건이
이미 같다.

## 추가로 확인하는 것
- 모델 간 **오차 상관계수**를 폴드마다 출력한다. 앙상블이 도움이 되려면
  두 모델이 "서로 다른 실수"를 해야 하는데(오차 상관이 낮아야 함), v1에서
  앙상블이 단독보다 나빴던 이유를 이 숫자로 설명할 수 있는지 본다.
- MAE도 RMSE와 함께 산출한다(08-22 검수 지적사항 반영 — 한 지표만 보고
  판정하지 않는다).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent

_spec_h = importlib.util.spec_from_file_location("harness", ROOT / "backtest_harness_v1_2026-08-20밤.py")
harness = importlib.util.module_from_spec(_spec_h)
sys.modules["harness"] = harness
_spec_h.loader.exec_module(harness)

_spec_d = importlib.util.spec_from_file_location("deep", ROOT / "train_hourly_deep_v3_2026-08-21.py")
deep = importlib.util.module_from_spec(_spec_d)
sys.modules["deep"] = deep
_spec_d.loader.exec_module(deep)

sys.path.insert(0, str(ROOT))
from model_common import optimize_nonnegative_weights, write_json  # noqa: E402

OUT = ROOT / "outputs" / "앙상블_1시간_3자_v2_2026-08-21"
HORIZONS = [24, 48]
MAX_EPOCHS = 14


def train_lstm_track_epochs(data, tr_idx, va_idx, history_length, seed):
    """1단계: inner_tr로 학습 + inner_va로 조기종료 → (모델, 최적에폭수) 반환.
    deep.train_model과 동일 로직에 '몇 번째 에폭이 최선이었나'만 추가로 기록."""
    torch.manual_seed(seed)
    model = deep.HourlyModel("장단기기억모델", data.history.shape[1], data.future.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(deep.HourlyDataset(data, tr_idx, history_length), batch_size=256, shuffle=True)
    valid_loader = DataLoader(deep.HourlyDataset(data, va_idx, history_length), batch_size=512, shuffle=False)
    best, best_state, best_epoch, stale = float("inf"), None, 1, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for history, future, target, mask in train_loader:
            optimizer.zero_grad()
            loss = deep.masked_loss(model(history, future), target, mask)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval(); losses = []
        with torch.no_grad():
            for history, future, target, mask in valid_loader:
                losses.append(float(deep.masked_loss(model(history, future), target, mask).item()))
        value = float(np.mean(losses))
        if value < best - 1e-6:
            best, stale, best_epoch = value, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def train_lstm_fixed_epochs(data, tr_idx, history_length, seed, n_epochs):
    """2단계: 학습구간 100%로 정해진 에폭 수만큼 재학습(조기종료 없음).
    → 트리모델(학습구간 100%)과 학습량을 맞춘다."""
    torch.manual_seed(seed)
    model = deep.HourlyModel("장단기기억모델", data.history.shape[1], data.future.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(deep.HourlyDataset(data, tr_idx, history_length), batch_size=256, shuffle=True)
    for _ in range(n_epochs):
        model.train()
        for history, future, target, mask in loader:
            optimizer.zero_grad()
            loss = deep.masked_loss(model(history, future), target, mask)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model


def tree_fold_predictions(df, horizon, start, end, seed, capacity_kw):
    """하네스와 동일 절차. 시험 예측은 학습구간 100% 모델, inner 예측은 80% 모델."""
    candidate_cols = harness.sel.CANDIDATE_COLUMNS
    frame = harness.build_frame(df, horizon, candidate_cols)
    daylight = frame[frame["목표_낮시간"] > 0]
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]

    train_all = daylight[daylight.index < start]
    test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]

    tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
    chosen = harness.select_features_in_fold(
        tr_for_sel, candidate_cols, threshold=0.3, apply_multicollinearity=True, apply_deploy_filter=True,
    )
    feature_cols = base_cols + chosen
    required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
    train = train_all.dropna(subset=required)
    test = test_all.dropna(subset=required)
    if len(train) < 200 or len(test) < 30:
        return None

    cut = int(len(train) * 0.8)
    inner_tr, inner_va = train.iloc[:cut], train.iloc[cut:]

    result = {"inner": {}, "test": {}, "actual_inner": inner_va["목표_발전출력_kW"],
              "actual_test": test["목표_발전출력_kW"]}
    for name in ["LightGBM", "XGBoost"]:
        m_full = harness.make_model(name, seed)
        m_full.fit(train[feature_cols], train["목표_발전출력_kW"])
        result["test"][name] = pd.Series(
            np.clip(m_full.predict(test[feature_cols]), 0, capacity_kw), index=test.index)
        m_inner = harness.make_model(name, seed)
        m_inner.fit(inner_tr[feature_cols], inner_tr["목표_발전출력_kW"])
        result["inner"][name] = pd.Series(
            np.clip(m_inner.predict(inner_va[feature_cols]), 0, capacity_kw), index=inner_va.index)
    return result


def lstm_fold_predictions(data, start, end, history_length, seed, capacity_kw):
    """★v2 수정★ 2단계 학습으로 트리모델과 학습량을 맞춘다."""
    issue_times = pd.DatetimeIndex(data.issue_times)
    train_idx = np.flatnonzero(issue_times < start)
    test_idx = np.flatnonzero((issue_times >= start) & (issue_times <= end))
    if len(train_idx) < 200 or len(test_idx) < 30:
        return None
    cut = int(len(train_idx) * 0.8)
    inner_tr_idx, inner_va_idx = train_idx[:cut], train_idx[cut:]
    if len(inner_va_idx) < 30:
        return None

    # 1단계: 에폭 수 결정(시험셋 안 봄) + 가중치용 inner 예측
    model_inner, best_epoch = train_lstm_track_epochs(data, inner_tr_idx, inner_va_idx, history_length, seed)
    inner_pred = deep.predict(model_inner, data, inner_va_idx, history_length, capacity_kw)

    # 2단계: 학습구간 100%로 그 에폭 수만큼 재학습 → 시험 예측
    model_full = train_lstm_fixed_epochs(data, train_idx, history_length, seed, best_epoch)
    test_pred = deep.predict(model_full, data, test_idx, history_length, capacity_kw)

    return {"inner_issue": issue_times[inner_va_idx], "inner_pred": inner_pred,
            "test_issue": issue_times[test_idx], "test_pred": test_pred, "best_epoch": best_epoch}


def run_all(df, data, config, capacity_kw, seed):
    history_length = int(config["hourly"]["sequence_history_hours"])
    fold_rows = []
    for i, w in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold_name = f"{i}_{w.get('_계절','')}"
        print(f"\n[{fold_name}] LSTM 2단계 학습(에폭탐색 → 전체재학습)...")
        lstm = lstm_fold_predictions(data, start, end, history_length, seed, capacity_kw)
        if lstm is None:
            print(f"  건너뜀(표본 부족)")
            continue
        print(f"  최적 에폭={lstm['best_epoch']} (이 에폭 수로 학습구간 100% 재학습 완료)")

        for horizon in HORIZONS:
            hi = horizon - 1
            tree = tree_fold_predictions(df, horizon, start, end, seed, capacity_kw)
            if tree is None:
                continue
            lstm_inner = pd.Series(lstm["inner_pred"][:, hi], index=lstm["inner_issue"])
            lstm_test = pd.Series(lstm["test_pred"][:, hi], index=lstm["test_issue"])

            inner_idx = tree["actual_inner"].index.intersection(lstm_inner.index)
            test_idx_c = tree["actual_test"].index.intersection(lstm_test.index)
            if len(inner_idx) < 30 or len(test_idx_c) < 30:
                continue

            names = ["LightGBM", "XGBoost", "LSTM"]
            inner_matrix = np.column_stack([
                tree["inner"]["LightGBM"].reindex(inner_idx).to_numpy(),
                tree["inner"]["XGBoost"].reindex(inner_idx).to_numpy(),
                lstm_inner.reindex(inner_idx).to_numpy()])
            w_res = optimize_nonnegative_weights(tree["actual_inner"].reindex(inner_idx).to_numpy(), inner_matrix)
            weights = np.asarray(w_res["가중치"], float)

            test_matrix = np.column_stack([
                tree["test"]["LightGBM"].reindex(test_idx_c).to_numpy(),
                tree["test"]["XGBoost"].reindex(test_idx_c).to_numpy(),
                lstm_test.reindex(test_idx_c).to_numpy()])
            test_actual = tree["actual_test"].reindex(test_idx_c).to_numpy()
            ens_pred = np.clip(test_matrix @ weights, 0, capacity_kw)

            row = {"폴드": fold_name, "수평_h": horizon, "표본수": len(test_idx_c),
                   "최적에폭": lstm["best_epoch"],
                   "가중치_LightGBM": round(float(weights[0]), 4),
                   "가중치_XGBoost": round(float(weights[1]), 4),
                   "가중치_LSTM": round(float(weights[2]), 4)}
            for j, name in enumerate(names):
                m = harness.metrics(test_actual, test_matrix[:, j], capacity_kw)
                row[f"{name}_RMSE"] = m["RMSE_kW"]
                row[f"{name}_MAE"] = m["MAE_kW"]
            m_ens = harness.metrics(test_actual, ens_pred, capacity_kw)
            row["앙상블_RMSE"], row["앙상블_MAE"] = m_ens["RMSE_kW"], m_ens["MAE_kW"]

            # 오차 상관계수(앙상블이 왜 도움이 되는지/안 되는지 진단)
            errs = test_matrix - test_actual[:, None]
            row["오차상관_XGB_LSTM"] = round(float(np.corrcoef(errs[:, 1], errs[:, 2])[0, 1]), 4)
            row["오차상관_LGBM_XGB"] = round(float(np.corrcoef(errs[:, 0], errs[:, 1])[0, 1]), 4)

            fold_rows.append(row)
            print(f"  +{horizon}h n={len(test_idx_c)} 가중치={weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f} "
                  f"RMSE: LGBM={row['LightGBM_RMSE']:.2f} XGB={row['XGBoost_RMSE']:.2f} "
                  f"LSTM={row['LSTM_RMSE']:.2f} 앙상블={row['앙상블_RMSE']:.2f} "
                  f"| 오차상관(XGB,LSTM)={row['오차상관_XGB_LSTM']:.3f}")
    return fold_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    print("LSTM용 시퀀스 데이터 준비 중...")
    data = deep.load_data(config)
    print(f"  유효 표본수: {len(data.positions)}")

    rows = run_all(df, data, config, capacity_kw, seed)
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "폴드별_결과.csv", index=False, encoding="utf-8-sig")

    cols = ["LightGBM_RMSE", "XGBoost_RMSE", "LSTM_RMSE", "앙상블_RMSE",
            "LightGBM_MAE", "XGBoost_MAE", "LSTM_MAE", "앙상블_MAE", "오차상관_XGB_LSTM"]
    summary = df_rows.groupby("수평_h")[cols].mean().round(3)
    print("\n=== 수평별 평균(폴드 평균) ===")
    print(summary.to_string())
    summary.to_csv(OUT / "수평별_평균.csv", encoding="utf-8-sig")
    write_json(OUT / "요약.json", {"폴드별": rows, "수평별_평균": summary.reset_index().to_dict("records")})
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
