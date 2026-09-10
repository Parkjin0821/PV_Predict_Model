# -*- coding: utf-8 -*-
"""+24h·+48h 3자 앙상블(LightGBM+XGBoost+LSTM) — 하네스·LSTM 통합 첫 시도.

## 배경 (08-22, 사용자 요청 "3개 다 앙상블 돌려보자")
`train_hourly_deep_v3_2026-08-21.py`에서 LSTM이 +24h·+48h에서 XGBoost
단독보다 나은 걸 확인했다. 그런데 그 스크립트를 자세히 보니 **조기종료
(early stopping) 검증셋으로 시험폴드(test) 자체를 썼다** — 이건
`train_model(kind, data, train_idx, valid_idx, ...)`에서 `valid_idx`가
`main()`에서 그 폴드의 시험셋과 동일하게 넘어갔기 때문이다(v1 때부터
있던 관행을 그대로 물려받음, 이번에 처음 발견). 시험셋을 보고 "가장
좋은 체크포인트"를 고르는 건 경미하지만 실질적인 누출이다 — 지금까지
보고한 LSTM 성능이 실제보다 낙관적일 수 있다는 뜻.

## 이 스크립트가 고친 것
1. **조기종료 누출 제거**: 각 폴드 학습구간(train)을 다시 80/20으로
   쪼개(`inner_tr`/`inner_va`, 트리모델 앙상블 가중치 최적화와 정확히
   같은 관례) LSTM을 `inner_tr`로만 학습하고 `inner_va`로만 조기종료를
   판단한다. 시험셋은 최종 평가에만 쓴다.
2. **폴드 정의를 트리모델과 통일**: 하네스는 "발행시각(issue time)"이
   폴드 경계 안에 있는지로 학습/시험을 나눈다(목표시각 아님). 기존
   LSTM 스크립트는 목표시각 기준이었다 — 이번엔 하네스와 동일하게
   발행시각 기준으로 통일해 두 계열이 정확히 같은 발행시각 표본을
   보게 했다.
3. **3자 가중결합**: LightGBM·XGBoost는 하네스와 동일한 절차(폴드 내
   상관계수→다중공선성→배포필터 특성재선택)로 학습하고, 여기에 LSTM
   예측을 더해 `model_common.optimize_nonnegative_weights`로 비음수
   가중결합한다(내부 홀드아웃에서 가중치 산출 → 시험폴드에 적용, 트리
   앙상블과 동일 원칙). 세 모델 예측이 전부 존재하는 발행시각만 채택
   (교집합) — 표본이 살짝 줄어들 수 있음, 정직하게 표시.

## 대상 수평
+24h·+48h만(+1h는 LSTM이 압도적으로 져서 애초에 앙상블 후보가 아님,
`train_hourly_deep_v3` 결과 참고).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

OUT = ROOT / "outputs" / "앙상블_1시간_3자_v1_2026-08-21"
HORIZONS = [24, 48]


def tree_fold_predictions(df: pd.DataFrame, horizon: int, start: pd.Timestamp, end: pd.Timestamp,
                           seed: int, capacity_kw: float):
    """하네스와 동일 절차로 LightGBM·XGBoost를 학습해 발행시각→예측 매핑을 만든다.
    (train_all/test_all은 발행시각으로 자름 — 하네스 run_backtest와 동일)"""
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


def lstm_fold_predictions(data, start: pd.Timestamp, end: pd.Timestamp,
                           history_length: int, seed: int, capacity_kw: float):
    """하네스와 동일하게 발행시각으로 폴드를 나누고, LSTM은 inner_tr로만 학습
    (test를 조기종료에 쓰지 않음 — 이번에 고친 부분). 폴드 정의가 수평(H)과
    무관하므로 **폴드당 한 번만 학습**해서 두 수평(+24h·+48h) 예측을 동시에
    뽑는다(모델이 원래 48개 수평을 한 번에 출력하는 구조라 재사용 가능)."""
    issue_times = pd.DatetimeIndex(data.issue_times)
    train_idx = np.flatnonzero(issue_times < start)
    test_idx = np.flatnonzero((issue_times >= start) & (issue_times <= end))
    if len(train_idx) < 200 or len(test_idx) < 30:
        return None

    cut = int(len(train_idx) * 0.8)
    inner_tr_idx, inner_va_idx = train_idx[:cut], train_idx[cut:]
    if len(inner_va_idx) < 30:
        return None

    torch.manual_seed(seed)
    model = deep.train_model("장단기기억모델", data, inner_tr_idx, inner_va_idx, history_length, seed)
    inner_pred_all = deep.predict(model, data, inner_va_idx, history_length, capacity_kw)  # (n,48)
    test_pred_all = deep.predict(model, data, test_idx, history_length, capacity_kw)

    return {
        "inner_issue": issue_times[inner_va_idx], "inner_pred": inner_pred_all,
        "test_issue": issue_times[test_idx], "test_pred": test_pred_all,
    }


def run_all(df: pd.DataFrame, data, config, capacity_kw: float, seed: int) -> list[dict]:
    history_length = int(config["hourly"]["sequence_history_hours"])
    fold_rows = []
    for i, w in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold_name = f"{i}_{w.get('_계절','')}"
        print(f"\n[{fold_name}] LSTM 폴드당 1회 학습(수평 공용)...")
        lstm = lstm_fold_predictions(data, start, end, history_length, seed, capacity_kw)
        if lstm is None:
            print(f"  [{fold_name}] LSTM 표본 부족으로 이 폴드 전체 건너뜀")
            continue

        for horizon in HORIZONS:
            hi = horizon - 1
            tree = tree_fold_predictions(df, horizon, start, end, seed, capacity_kw)
            if tree is None:
                print(f"  [{fold_name}] +{horizon}h 트리모델 표본 부족으로 건너뜀")
                continue

            lstm_inner = pd.Series(lstm["inner_pred"][:, hi], index=lstm["inner_issue"])
            lstm_test = pd.Series(lstm["test_pred"][:, hi], index=lstm["test_issue"])

            # 세 모델 예측이 전부 있는 발행시각만 채택(교집합)
            inner_idx = tree["actual_inner"].index.intersection(lstm_inner.index)
            test_idx_common = tree["actual_test"].index.intersection(lstm_test.index)
            if len(inner_idx) < 30 or len(test_idx_common) < 30:
                print(f"  [{fold_name}] +{horizon}h 교집합 표본 부족으로 건너뜀 "
                      f"(inner={len(inner_idx)}, test={len(test_idx_common)})")
                continue

            names = ["LightGBM", "XGBoost", "LSTM"]
            inner_matrix = np.column_stack([
                tree["inner"]["LightGBM"].reindex(inner_idx).to_numpy(),
                tree["inner"]["XGBoost"].reindex(inner_idx).to_numpy(),
                lstm_inner.reindex(inner_idx).to_numpy(),
            ])
            inner_actual = tree["actual_inner"].reindex(inner_idx).to_numpy()
            w_res = optimize_nonnegative_weights(inner_actual, inner_matrix)
            weights = np.asarray(w_res["가중치"], float)

            test_matrix = np.column_stack([
                tree["test"]["LightGBM"].reindex(test_idx_common).to_numpy(),
                tree["test"]["XGBoost"].reindex(test_idx_common).to_numpy(),
                lstm_test.reindex(test_idx_common).to_numpy(),
            ])
            test_actual = tree["actual_test"].reindex(test_idx_common).to_numpy()
            ens_pred = np.clip(test_matrix @ weights, 0, capacity_kw)

            row = {"폴드": fold_name, "수평_h": horizon, "표본수": len(test_idx_common),
                   "가중치_LightGBM": round(float(weights[0]), 4),
                   "가중치_XGBoost": round(float(weights[1]), 4),
                   "가중치_LSTM": round(float(weights[2]), 4)}
            for j, name in enumerate(names):
                m = harness.metrics(test_actual, test_matrix[:, j], capacity_kw)
                row[f"{name}_RMSE"] = m["RMSE_kW"]
            m_ens = harness.metrics(test_actual, ens_pred, capacity_kw)
            row["앙상블_RMSE"] = m_ens["RMSE_kW"]
            fold_rows.append(row)
            print(f"  [{fold_name}] +{horizon}h n={len(test_idx_common)} 가중치(LGBM/XGB/LSTM)="
                  f"{weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f} "
                  f"RMSE: LightGBM={row['LightGBM_RMSE']:.2f} XGBoost={row['XGBoost_RMSE']:.2f} "
                  f"LSTM={row['LSTM_RMSE']:.2f} 앙상블={row['앙상블_RMSE']:.2f}")
    return fold_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    print("LSTM용 시퀀스 데이터 준비 중(수평 공용, 한 번만)...")
    data = deep.load_data(config)
    print(f"  유효 표본수: {len(data.positions)}")

    all_rows = run_all(df, data, config, capacity_kw, seed)

    df_rows = pd.DataFrame(all_rows)
    df_rows.to_csv(OUT / "폴드별_결과.csv", index=False, encoding="utf-8-sig")

    print("\n=== 수평별 평균(폴드 평균) ===")
    summary = df_rows.groupby("수평_h")[["LightGBM_RMSE", "XGBoost_RMSE", "LSTM_RMSE", "앙상블_RMSE"]].mean().round(3)
    print(summary.to_string())
    summary.to_csv(OUT / "수평별_평균.csv", encoding="utf-8-sig")
    write_json(OUT / "요약.json", {"폴드별": all_rows, "수평별_평균": summary.reset_index().to_dict("records")})
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
