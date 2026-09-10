# -*- coding: utf-8 -*-
"""XGBoost(공식 단기 모델) + LSTM 잔차보정 하이브리드 파일럿 - 단기+1h 단일 수평.

## 배경 (사용자 09-03 대화)
사용자가 "보통 LightGBM/XGBoost에 LSTM을 끼얹는 방식을 쓴다는데 어떤가"
질문. AGENTS.md 08-26 절에 이미 기록된 사실: +1h는 LSTM이 XGBoost 대비
57% 열세(19.844 vs 12.610), XGBoost+LSTM(같은 타깃을 각자 예측 후 평균)
앙상블도 오차상관 0.89~0.90이 너무 높아 RMSE +0.7%뿐이고 MAE는 오히려
악화. 재검토 조건(Q5)에 "오차상관 0.8 이하로 낮아지는 구성 발견"이
명시돼 있음 - 이번 파일럿은 "같은 타깃을 각자 예측 후 평균"이 아니라
**"XGBoost 잔차(진짜값-XGBoost예측)를 LSTM이 따로 예측해서 더하는"**
잔차보정(residual-stacking) 구성으로, 구조적으로 오차상관 조건이 달라질
가능성이 있는지 실측한다. 문헌에서도 "같은 타깃 병렬앙상블"보다
"잔차보정형 하이브리드"가 실제 성과가 보고되는 쪽이었다(WebSearch
09-03, ARIMA-CNN-LSTM residual-corrected 등).

## 범위(파일럿)
단기 +1h 단일 수평만 - 08-26 절에서 LSTM이 가장 크게(57%) 졌던 조건을
그대로 재사용해 "잔차보정이 그 실패를 뒤집을 수 있는가"를 가장 엄격한
케이스로 먼저 확인한다. 여기서 안 되면 다른 수평·인버터단위로 확장할
이유가 약하다.

## 방법
- XGBoost 베이스라인: `defect_policy_comparison`의 `load_short_frame(1)`
  (v5 결함구간보정 반영 공식 프레임 로더 재사용) + `backtest_harness`의
  `select_features_in_fold`(폴드 학습구간 안에서만 상관계수 계산, 누출
  방지)·`make_model("XGBoost", seed)`(공식과 동일 하이퍼파라미터) 재사용.
  새로 재구현하지 않음.
- 폴드: `inverter_disaggregation_cv_v1_2026-08-27.py`와 동일한 공식
  5계절 rolling-origin(OFFICIAL_WINDOWS) - 다른 인버터/하이브리드
  실험과 비교 가능하도록 통일.
- LSTM 잔차모델: 과거 W=12시간의 실측 발전출력(발행시각 기준 과거만,
  미래 없음) 시퀀스 → LSTM 인코더 → XGBoost가 고른 정적특성과 결합해
  "XGBoost 학습잔차(실측-XGBoost예측, 학습구간 in-sample)"를 회귀.
  시험구간에서 그 잔차예측을 XGBoost 예측에 더해 최종값 산출.
- 판정: XGBoost 단독 대비 하이브리드의 MAE/RMSE, 그리고 AGENTS.md Q5
  재검토조건(a)에 맞춰 "XGBoost 시험오차"와 "LSTM 잔차예측"의 상관계수를
  폴드별로 보고(잔차보정이 실제로 의미있는 신호를 잡았는지 판단 근거 -
  이 값이 낮으면 LSTM이 노이즈를 학습했다는 뜻).

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\03_모델학습\\현재_종합파이프라인"
python residual_stacking_xgb_lstm_v1_2026-09-03.py
```
로컬 재학습만(API 없음). CPU torch, 폴드당 수십 초 내외 예상.

## 산출물 (`outputs/잔차보정_XGB_LSTM_파일럿_v1_2026-09-03/`)
- `폴드별_성능.csv` - 폴드×방법(XGBoost단독/하이브리드): MAE·RMSE·n
- `요약.txt`
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "잔차보정_XGB_LSTM_파일럿_v1_2026-09-03"
DPC_SCRIPT = ROOT / "defect_policy_comparison_v1_2026-08-21.py"
CAPACITY_KW = 240.58
SEED = 42
WINDOW_H = 12  # LSTM 입력: 과거 12시간 실측 발전출력

OFFICIAL_WINDOWS = [
    ("1_여름", "2025-06-01", "2025-08-14"),
    ("2_가을_결함종료후", "2025-11-17", "2025-12-14"),
    ("3_겨울", "2025-12-15", "2026-02-14"),
    ("4_봄", "2026-02-15", "2026-04-14"),
    ("5_초여름", "2026-04-15", "2026-08-04"),
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ResidualLSTM(nn.Module):
    def __init__(self, static_dim: int, hidden: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(seq)
        x = torch.cat([h[-1], static], dim=1)
        return self.head(x).squeeze(-1)


def _build_sequences(power_series: pd.Series, issue_times: pd.DatetimeIndex,
                      window_h: int) -> tuple[np.ndarray, np.ndarray]:
    """각 발행시각 직전 window_h시간의 실측 발전출력 시퀀스. 미래값 없음."""
    seqs = np.full((len(issue_times), window_h), np.nan, dtype=np.float32)
    valid = np.ones(len(issue_times), dtype=bool)
    idx = power_series.index
    for i, t in enumerate(issue_times):
        # 발행시각(t) 시점에 이미 알려진 값 = t 포함 과거. build_frame의
        # plant_output_kw.shift(1) 관례와 맞춰 t 미포함(직전 시간까지).
        window_start = t - pd.Timedelta(hours=window_h)
        window = power_series.loc[(idx >= window_start) & (idx < t)]
        if len(window) != window_h or window.isna().any():
            valid[i] = False
            continue
        seqs[i] = window.to_numpy()
    return seqs, valid


def _train_residual_lstm(seq_train: np.ndarray, static_train: np.ndarray,
                          resid_train: np.ndarray, static_dim: int,
                          epochs: int = 80, seed: int = SEED) -> tuple[ResidualLSTM, float, float]:
    torch.manual_seed(seed)
    model = ResidualLSTM(static_dim=static_dim)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    resid_mu, resid_sigma = float(resid_train.mean()), float(resid_train.std())
    resid_sigma = resid_sigma if resid_sigma > 1e-6 else 1.0
    resid_norm = (resid_train - resid_mu) / resid_sigma

    seq_t = torch.tensor(np.nan_to_num(seq_train[:, :, None]), dtype=torch.float32)
    static_t = torch.tensor(np.nan_to_num(static_train), dtype=torch.float32)
    y_t = torch.tensor(resid_norm, dtype=torch.float32)
    n = len(y_t)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, 64):
            batch = perm[start:start + 64]
            opt.zero_grad()
            pred = model(seq_t[batch], static_t[batch])
            loss = nn.functional.mse_loss(pred, y_t[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    model.eval()
    return model, resid_mu, resid_sigma


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dpc = _load_module("dpc_residual_20260903", DPC_SCRIPT)
    harness = dpc.harness

    frame = dpc.load_short_frame(1)
    daylight = frame[frame["목표_낮시간"] > 0].copy()
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    base_cols = [c for c in daylight.columns if c not in candidate_cols
                 and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]

    # LSTM 시퀀스용 실측 발전출력 원계열(다시 로드 - build_frame과 동일 소스)
    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                          parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    v5_dir = ROOT / "outputs" / "v5_복구_2026-08-21"
    v5_1h = pd.read_parquet(v5_dir / "집계_1시간_자료_v5.parquet")
    common = hourly.index.intersection(v5_1h.index)
    power_series = v5_1h.loc[common, "발전출력_kW"].sort_index()

    fold_rows = []
    detail_rows = []
    for fold, s, e in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train_all = daylight[daylight.index < start]
        test_all = daylight[(daylight.index >= start) & (daylight.index < end)]
        if len(train_all) < 300 or len(test_all) < 30:
            print(f"[{fold}] 표본 부족 - 생략 (train={len(train_all)}, test={len(test_all)})")
            continue

        chosen = harness.select_features_in_fold(train_all, candidate_cols, 0.3,
                                                   apply_multicollinearity=True,
                                                   apply_deploy_filter=True)
        features = base_cols + chosen
        native_ok = harness.NATIVE_MISSING_OK
        required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW"]
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=[c for c in features if c not in native_ok]
                                + ["목표_발전출력_kW"])
        if len(train) < 300 or len(test) < 30:
            print(f"[{fold}] 결측 제거 후 표본 부족 - 생략")
            continue

        model_xgb = harness.make_model("XGBoost", SEED)
        model_xgb.fit(train[features], train["목표_발전출력_kW"])
        pred_train_xgb = np.clip(model_xgb.predict(train[features]), 0, CAPACITY_KW)
        pred_test_xgb = np.clip(model_xgb.predict(test[features]), 0, CAPACITY_KW)
        resid_train = (train["목표_발전출력_kW"].to_numpy() - pred_train_xgb).astype(np.float32)

        seq_train, ok_train = _build_sequences(power_series, train.index, WINDOW_H)
        seq_test, ok_test = _build_sequences(power_series, test.index, WINDOW_H)
        if ok_train.sum() < 300 or ok_test.sum() < 30:
            print(f"[{fold}] LSTM 시퀀스 유효표본 부족 - 생략 "
                  f"(train_ok={ok_train.sum()}, test_ok={ok_test.sum()})")
            continue

        static_cols = features  # XGBoost가 고른 특성을 그대로 정적입력으로 재사용
        static_train_raw = train.loc[ok_train, static_cols].to_numpy(dtype=np.float32)
        static_test_raw = test.loc[ok_test, static_cols].to_numpy(dtype=np.float32)
        mu, sigma = static_train_raw.mean(axis=0), static_train_raw.std(axis=0)
        sigma[sigma < 1e-6] = 1.0
        static_train = (static_train_raw - mu) / sigma
        static_test = (static_test_raw - mu) / sigma

        seq_mu, seq_sigma = seq_train[ok_train].mean(), seq_train[ok_train].std()
        seq_sigma = seq_sigma if seq_sigma > 1e-6 else 1.0
        seq_train_n = (seq_train[ok_train] - seq_mu) / seq_sigma
        seq_test_n = (seq_test[ok_test] - seq_mu) / seq_sigma

        lstm, resid_mu, resid_sigma = _train_residual_lstm(
            seq_train_n, static_train, resid_train[ok_train], static_dim=len(static_cols))
        with torch.no_grad():
            resid_pred_norm = lstm(
                torch.tensor(np.nan_to_num(seq_test_n[:, :, None]), dtype=torch.float32),
                torch.tensor(np.nan_to_num(static_test), dtype=torch.float32),
            ).numpy()
        resid_pred_test = resid_pred_norm * resid_sigma + resid_mu

        test_ok = test.loc[ok_test]
        pred_test_xgb_ok = pred_test_xgb[np.isin(test.index, test_ok.index)]
        y_test = test_ok["목표_발전출력_kW"].to_numpy()
        pred_hybrid = np.clip(pred_test_xgb_ok + resid_pred_test, 0, CAPACITY_KW)

        def _mae_rmse(y, p):
            mae = float(np.mean(np.abs(y - p)))
            rmse = float(np.sqrt(np.mean((y - p) ** 2)))
            return mae, rmse

        mae_x, rmse_x = _mae_rmse(y_test, pred_test_xgb_ok)
        mae_h, rmse_h = _mae_rmse(y_test, pred_hybrid)
        true_resid_test = y_test - pred_test_xgb_ok
        corr = float(np.corrcoef(true_resid_test, resid_pred_test)[0, 1]) \
            if np.std(resid_pred_test) > 1e-9 else float("nan")

        fold_rows.append({
            "폴드": fold, "n": len(y_test),
            "XGBoost단독_MAE": round(mae_x, 3), "XGBoost단독_RMSE": round(rmse_x, 3),
            "하이브리드_MAE": round(mae_h, 3), "하이브리드_RMSE": round(rmse_h, 3),
            "MAE개선율_pct": round((1 - mae_h / mae_x) * 100, 2) if mae_x else float("nan"),
            "RMSE개선율_pct": round((1 - rmse_h / rmse_x) * 100, 2) if rmse_x else float("nan"),
            "잔차예측_상관계수": round(corr, 3),
            "특성수": len(features),
        })
        print(f"[{fold}] XGBoost MAE={mae_x:.3f} RMSE={rmse_x:.3f} | "
              f"하이브리드 MAE={mae_h:.3f} RMSE={rmse_h:.3f} | 잔차상관={corr:.3f}")

    result = pd.DataFrame(fold_rows)
    result.to_csv(OUT / "폴드별_성능.csv", index=False, encoding="utf-8-sig")

    overall_lines = []
    if len(result):
        w_mae_x = np.average(result["XGBoost단독_MAE"], weights=result["n"])
        w_mae_h = np.average(result["하이브리드_MAE"], weights=result["n"])
        w_rmse_x = np.average(result["XGBoost단독_RMSE"], weights=result["n"])
        w_rmse_h = np.average(result["하이브리드_RMSE"], weights=result["n"])
        overall_lines = [
            f"가중평균 XGBoost단독: MAE={w_mae_x:.3f} RMSE={w_rmse_x:.3f}",
            f"가중평균 하이브리드: MAE={w_mae_h:.3f} RMSE={w_rmse_h:.3f}",
            f"전체 MAE 개선율: {(1 - w_mae_h / w_mae_x) * 100:.2f}%",
            f"전체 RMSE 개선율: {(1 - w_rmse_h / w_rmse_x) * 100:.2f}%",
            f"폴드별 잔차예측 상관계수: {result['잔차예측_상관계수'].tolist()}",
            "재검토조건(a) 판정(AGENTS.md Q5, 상관 0.8 이하 아님에 유의 - "
            "여기서는 '잔차예측이 실제 잔차를 얼마나 잘 맞췄는가'로 해석):"
            + (" 신호 있음(상관 0 초과)" if (result['잔차예측_상관계수'] > 0).any() else " 신호 거의 없음"),
        ]
    else:
        overall_lines = ["모든 폴드가 표본 부족으로 생략됨 - 결과 없음"]
    (OUT / "요약.txt").write_text("\n".join(overall_lines), encoding="utf-8")
    print("\n=== 요약 ===")
    print("\n".join(overall_lines))


if __name__ == "__main__":
    main()
