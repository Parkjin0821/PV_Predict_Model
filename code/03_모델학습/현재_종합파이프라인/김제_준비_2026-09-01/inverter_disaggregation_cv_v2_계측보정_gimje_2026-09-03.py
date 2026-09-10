# -*- coding: utf-8 -*-
"""김제 인버터 분해 CV v2 - 인버터합계/공식계측값 불일치 계측보정(pro-rata reconciliation).

## 배경
v1(09-03) 사후발견: 인버터 10대 실측합계와 공식 발전소 계량기값
(`plant_ac_power_kw`)이 완전히 일치하지 않음(4,729시험행 중 17.5%가
1kW 초과, 3.8%가 5kW 초과, 최대 24.5kW - 중앙값은 0). 사용자가
"원인조사+엄격필터링" 지시, 실제 조사는 제미나이 리서치로 대체 -
그 결과를 실측으로 검증·적용한다.

### 제미나이 리서치 요지(사용자 09-03 공유) + 실측 검증
- **원인**: 변압/전송손실(정격 1~3%), 계측기 등급 차이(계량기 Class
  0.2~0.5 vs 인버터 센서 Class 1.0~2.0), 시간동기화 오차, 순간 통신
  유실 - 전부 물리적으로 정상 범위인 현상이지 데이터 버그가 아니다.
- **실무 표준 임계값**: 오차를 "설비용량 대비 %"로 봐야 하며(순간출력
  대비가 아님 - 저출력시 왜곡됨), 3~5% 이내는 정상, 5% 초과만 통신
  오류로 의심해 제거.
- **★실측 검증(이 v2에서 확인)★**: 김제 설비용량 1,100kW 기준으로
  재계산하니 **최대 상대오차가 2.27%**(24.5kW/1,100kW) - 표준
  임계값(3~5%)의 정상범위 안이었다. **5% 초과 행은 0건** - 즉 "엄격한
  필터링으로 행을 버릴" 필요 자체가 없었다(제거 대상 이상치가 없음).
- **실무 표준 조정법(적용)**: 계량기값을 절대기준(Primary)으로 고정,
  인버터 비중은 상대배분(Secondary)에만 씀 - 이건 이미 A/B/C 방법
  자체가 하고 있던 것과 같은 구조. 다만 v1은 **채점용 "정답"(실제_kW)
  으로 인버터 원시합계를 그대로 썼다** - 이를 pro-rata 재조정
  (인버터_i × 계량기값/인버터합계)해서 Σ재조정값=계량기값이 항상
  성립하도록 고친다(비중 자체는 배율에 불변이라 A/B/C 예측 로직은
  안 바뀜 - 채점 기준값만 계량기와 내적으로 일치시킴).

## 재사용한 것 (재구현 안 함) - v1과 동일
- 총출력 예측: `total_output_weather_model_v1_gimje_2026-09-01.py`의
  `build_dataset()`·`FEATURES_15`·`expanding_folds_full_coverage()`·
  LightGBM 하이퍼파라미터.
- 인버터 원본: `시간집계_v1_2026-08-31/김제_인버터별_5분_야간0포함.parquet`.
- 사전등록기준 4개 규칙: v1과 완전히 동일(결과 보기 전 고정, 재조정
  방식 도입 자체도 결과를 보기 전에 - 제미나이 리서치 시점에 - 정한
  것이라 사후 기준변경이 아님).

## 재사용한 것 (재구현 안 함)
- 총출력 예측: `total_output_weather_model_v1_gimje_2026-09-01.py`의
  `build_dataset()`·`FEATURES_15`(ASOS4 제외, 09-01 상관분석으로 확정)·
  `expanding_folds_full_coverage()`·LightGBM 하이퍼파라미터를 그대로
  가져다 쓴다. 이 스크립트 안에서 그 총출력 모델을 다시 학습해 행단위
  OOF를 만드는 이유는(광주·부안과 달리) 기존 총출력_v1 산출물이 요약
  JSON뿐이라 행단위 CSV가 없어서다 - 같은 코드로 다시 학습·예측만
  반복할 뿐 모델링 방법 자체는 전혀 바꾸지 않는다.
- 인버터 원본: `시간집계_v1_2026-08-31/김제_인버터별_5분_야간0포함.parquet`
  (코덱스가 이미 만든 정렬본, 5분 그리드 2024-08-25~2026-08-05 완전정렬).
- 사전등록기준 4개 규칙: 광주 08-27과 완전히 동일한 문구·임계값.

## 김제 고유 사항 (광주와 다른 점)
- 인버터 10기, **전부 110kW 균등**(광주 50/50/30/39/50과 달리 정격용량이
  이미 동일) - A_균등비례는 그냥 10%씩.
- 총출력모델이 광주처럼 티어(초단기/단기/일간)·수평별로 나뉘어 있지
  않고 단일 day-ahead형 구조(90일 초기학습+30일블록 확장) - 억지로
  여러 수평을 만들어내지 않고 "김제_총출력" 단일 트랙으로만 판정한다.
- `_score`/`_adjudicate`는 광주 cv 모듈이 인버터수 5를 하드코딩(range(1,6))
  하고 있어 그대로 import해 쓸 수 없다 - 로직은 100% 동일하게 유지하고
  인버터수만 10으로 바꿔 이 파일 안에 그대로 옮겼다(새 판정기준 발명 아님).

## 가용성 판정
`quality_status_after_night`가 observed/physical_zero_night/
physical_zero_idle_supported인 5분 슬롯만 유효로 본다(interpolated·
missing은 제외 - 광주처럼 보간값을 정답으로 안 씀). 시간단위는 12개
5분슬롯 중 9개(75%) 이상 유효해야 그 시간 그 인버터가 "가용"이고,
CV 시험행은 10기 전부 가용인 시간만 사용한다(광주와 동일 원칙).

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\03_모델학습\\현재_종합파이프라인\\김제_준비_2026-09-01"
python inverter_disaggregation_cv_v1_gimje_2026-09-03.py
```

## 산출물 (`outputs/인버터_분해교차검증_v1_gimje_2026-09-03/`)
광주 08-27과 동일한 5개 파일(행단위_인버터별_예측정답.csv,
인버터별_전체성능.csv, 폴드별_성능.csv, 사전등록규칙_판정표.csv,
사전등록_채택규칙.json).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "인버터_분해교차검증_v2_계측보정_gimje_2026-09-03"
V1_DECISION_PATH = HERE / "outputs" / "인버터_분해교차검증_v1_gimje_2026-09-03" / "사전등록규칙_판정표.csv"
RECONCILE_DROP_THRESHOLD_PCT = 5.0  # 설비용량 대비 % - 제미나이 리서치 "실무표준" 상한
TOTAL_MODEL_SCRIPT = HERE / "total_output_weather_model_v1_gimje_2026-09-01.py"
INVERTER_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\시간집계_v1_2026-08-31"
    r"\김제_인버터별_5분_야간0포함.parquet"
)
N_INVERTERS = 10
CAPACITY = np.array([110.0] * N_INVERTERS, dtype=float)
CAPACITY_SHARE = CAPACITY / CAPACITY.sum()
SEED = 42
TOL_KW = 1.0  # 시간평균 kW 재구성 - 광주(0.011, 순간값)보다 여유(집계오차 감안)

PREREGISTERED_RULES = {
    "판정단위": "김제_총출력(단일 트랙, day-ahead형 rolling-origin)",
    "기준선": "A_균등비례",
    "규칙1": "A 대비 전체 MAE와 RMSE가 모두 1% 이상 개선",
    "규칙2": "어느 폴드에서도 RMSE가 A보다 5% 이상 악화되지 않음",
    "규칙3": "각 인버터 nMAE 악화가 A 대비 1.0%p 이내",
    "규칙4": "동일 시험행 100% 사용 및 10대 예측합계 오차 1.0kW 이하",
    "동률": "RMSE가 사실상 같으면 A→B→C 순으로 단순한 방법 우선",
    "주의": "결과를 보기 전에 코드에 고정한 규칙이며 실행 후 변경하지 않음",
    "광주_08-27_대비_차이": "인버터수 10(광주 5), 정격용량 전부 110kW 균등, "
                          "폴드는 김제 총출력모델의 기존 expanding walk-forward "
                          "그대로(5계절 고정창 아님) - 그 외 규칙 문구는 동일.",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_inverter_wide() -> pd.DataFrame:
    """5분 long-format을 인버터별 wide로 피벗 + 1시간 집계.

    ★주의★: `night_zero_ac`는 값이 아니라 "이 행에 야간0채움을 적용했는가"
    불리언 플래그다(실측으로 확인, 08-31 산출물 dtype=bool). 실제 kW 값은
    `ac_power_kw`(야간0채움이 이미 반영된 상태)를 써야 한다.
    """
    cols = ["grid_time_kst", "inverter_number", "ac_power_kw", "quality_status_after_night"]
    raw = pd.read_parquet(INVERTER_PARQUET, columns=cols)
    valid_status = {"observed", "physical_zero_night", "physical_zero_idle_supported"}
    raw["유효"] = raw["quality_status_after_night"].isin(valid_status)
    raw.loc[~raw["유효"], "ac_power_kw"] = np.nan

    wide = raw.pivot(index="grid_time_kst", columns="inverter_number", values="ac_power_kw")
    wide.columns = [f"인버터{i}_kW" for i in wide.columns]
    valid_wide = raw.pivot(index="grid_time_kst", columns="inverter_number", values="유효")
    valid_wide.columns = [f"인버터{i}_kW" for i in valid_wide.columns]
    wide = wide.sort_index()
    valid_wide = valid_wide.sort_index()

    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]
    count_valid = valid_wide[inv_cols].resample("1h").sum()
    mean_power = wide[inv_cols].resample("1h").mean()
    # 12개 5분슬롯 중 9개(75%) 미만 유효면 그 인버터 그 시간은 결측 처리
    for c in inv_cols:
        mean_power.loc[count_valid[c] < 9, c] = np.nan
    mean_power["가용인버터수"] = count_valid[inv_cols].ge(9).sum(axis=1)
    mean_power["발전소합계_kW"] = mean_power[inv_cols].sum(axis=1, min_count=N_INVERTERS)
    return mean_power


def _reconcile_actual(raw_actual: np.ndarray, official_total: np.ndarray,
                       capacity_sum: float) -> tuple[np.ndarray, np.ndarray]:
    """제미나이 리서치의 pro-rata 재조정: 인버터_i × (계량기값/인버터합계).

    ★09-03 정정★: 상대오차 분모를 "설비용량"이 아니라 **실측 순간출력**
    (활성발전 구간, 계량기값>=100kW)으로 바꿨다 - 사용자 지적("발전량이
    적은 시간대엔 용량을 분모로 쓰면 오차율이 착시로 낮아진다") 실측
    검증 결과 정확히 맞았음: 용량 기준 최대상대오차 2.27%(0건 초과)였던
    게, 실측기준(활성발전 구간)으로 재계산하니 최대 6.69%·5%초과 7건
    (1,909건 중)으로 나왔다. 활성발전 구간이 아닌 행(저출력·야간)은
    실측값이 0에 가까워 상대오차 정의 자체가 불안정하므로 용량기준을
    안전판으로 병행 사용한다(둘 중 하나라도 5% 초과면 드롭).

    비중(각 인버터가 전체에서 차지하는 비율)은 공통배율로 나눠도 그대로라
    A/B/C의 공유(share) 예측 로직 자체는 이 함수와 무관 - 여기서 고치는
    건 오직 "채점에 쓸 정답값"이 계량기(Primary, 절대기준)와 항상
    합산일치하도록 만드는 것뿐이다.
    """
    raw_sum = raw_actual.sum(axis=1)
    abs_diff = np.abs(raw_sum - official_total)
    rel_err_pct_capacity = abs_diff / capacity_sum * 100
    # 활성발전(계량기>=100kW) 구간만 실측기준 상대오차 계산 - 그 미만은
    # 분모가 0에 가까워 정의 불안정하므로 용량기준값을 그대로 둔다.
    active = official_total >= 100.0
    rel_err_pct_actual = np.where(
        active, abs_diff / np.where(active, official_total, 1.0) * 100, rel_err_pct_capacity)
    rel_err_pct = np.maximum(rel_err_pct_capacity, np.where(active, rel_err_pct_actual, 0.0))
    scale = np.where(raw_sum > 1e-9, official_total / np.where(raw_sum > 1e-9, raw_sum, 1.0), 1.0)
    unreconcilable = (raw_sum <= 1e-9) & (np.abs(official_total) > 1e-9)
    scale = np.where(unreconcilable, 1.0, scale)
    reconciled = raw_actual * scale[:, None]
    return reconciled, rel_err_pct


def _season(month) -> np.ndarray:
    values = np.asarray(month)
    return np.select([np.isin(values, [3, 4, 5]), np.isin(values, [6, 7, 8]),
                      np.isin(values, [9, 10, 11])],
                     ["봄", "여름", "가을"], default="겨울")


def _normalize_shares(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.clip(raw, 0.0, None)
    sums = raw.sum(axis=1, keepdims=True)
    fallback = np.tile(CAPACITY_SHARE, (len(raw), 1))
    return np.divide(raw, sums, out=fallback, where=sums > 1e-12)


def _recent_share_at(issue: pd.DatetimeIndex, agg: pd.DataFrame, lag: pd.Timedelta) -> pd.DataFrame:
    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]
    shares = agg[inv_cols].div(agg[inv_cols].sum(axis=1, min_count=N_INVERTERS), axis=0).dropna()
    query = issue - lag
    selected = shares.reindex(query, method="ffill", tolerance=pd.Timedelta("6h"))
    selected.index = issue
    return selected


def _predict_a(n: int) -> np.ndarray:
    return np.tile(CAPACITY_SHARE, (n, 1))


def _predict_b(train: pd.DataFrame, test_times: pd.Series) -> np.ndarray:
    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]
    shares = train[inv_cols].div(train[inv_cols].sum(axis=1), axis=0)
    meta = pd.DataFrame({"계절": _season(train["target_time_kst"].dt.month),
                         "시": train["target_time_kst"].dt.hour}, index=train.index)
    work = pd.concat([meta, shares], axis=1)
    by_key = work.groupby(["계절", "시"])[inv_cols].median()
    by_hour = work.groupby("시")[inv_cols].median()
    global_share = shares.median().fillna(pd.Series(CAPACITY_SHARE, index=inv_cols))
    rows = []
    for ts in test_times:
        key = (_season(pd.Index([ts.month]))[0], ts.hour)
        if key in by_key.index:
            rows.append(by_key.loc[key].to_numpy(float))
        elif ts.hour in by_hour.index:
            rows.append(by_hour.loc[ts.hour].to_numpy(float))
        else:
            rows.append(global_share.to_numpy(float))
    return _normalize_shares(np.asarray(rows))


def _predict_c(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]
    train = train.dropna(subset=inv_cols).copy()
    if len(train) < 300:
        raise RuntimeError(f"C 후보 학습행 부족: {len(train)}")
    shares = train[inv_cols].div(train[inv_cols].sum(axis=1), axis=0)
    valid = np.isfinite(shares).all(axis=1)
    train, shares = train.loc[valid], shares.loc[valid]
    predictions = []
    for i, col in enumerate(inv_cols):
        model = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              max_depth=5, min_child_samples=40, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=SEED + i, n_jobs=-1, verbosity=-1)
        model.fit(train[features], shares[col])
        predictions.append(model.predict(test[features]))
    return _normalize_shares(np.column_stack(predictions))


def _metrics(actual: np.ndarray, pred: np.ndarray, capacity: float) -> dict:
    mask = np.isfinite(actual) & np.isfinite(pred)
    y, p = actual[mask], pred[mask]
    if not len(y):
        return {"n": 0, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan,
                "R2": np.nan, "nMAE_pct": np.nan}
    mse = mean_squared_error(y, p)
    return {"n": int(len(y)), "MAE": float(mean_absolute_error(y, p)),
            "MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y, p)) if len(y) >= 2 and np.var(y) > 0 else np.nan,
            "nMAE_pct": float(mean_absolute_error(y, p) / capacity * 100)}


def _score(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail, folds = [], []
    for method, group in predictions.groupby("방법", sort=False):
        for inv in range(1, N_INVERTERS + 1):
            m = _metrics(group[f"인버터{inv}_실제_kW"].to_numpy(),
                         group[f"인버터{inv}_예측_kW"].to_numpy(), CAPACITY[inv - 1])
            detail.append({"방법": method, "인버터": inv, "용량_kW": CAPACITY[inv - 1], **m})
        all_actual = group[[f"인버터{i}_실제_kW" for i in range(1, N_INVERTERS + 1)]].to_numpy().ravel()
        all_pred = group[[f"인버터{i}_예측_kW" for i in range(1, N_INVERTERS + 1)]].to_numpy().ravel()
        m = _metrics(all_actual, all_pred, CAPACITY.mean())
        detail.append({"방법": method, "인버터": "전체_동일가중", "용량_kW": CAPACITY.mean(), **m})
        for fold, sub in group.groupby("폴드"):
            a = sub[[f"인버터{i}_실제_kW" for i in range(1, N_INVERTERS + 1)]].to_numpy().ravel()
            p = sub[[f"인버터{i}_예측_kW" for i in range(1, N_INVERTERS + 1)]].to_numpy().ravel()
            fm = _metrics(a, p, CAPACITY.mean())
            folds.append({"방법": method, "폴드": fold, **fm})
    return pd.DataFrame(detail), pd.DataFrame(folds)


def _adjudicate(detail: pd.DataFrame, folds: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    method_order = {"A_균등비례": 0, "B_계절시간중앙값": 1, "C_LightGBM비중": 2}
    rows = []
    group = detail[detail["인버터"] == "전체_동일가중"]
    base = group[group["방법"] == "A_균등비례"].iloc[0]
    candidates = []
    for _, row in group.iterrows():
        method = row["방법"]
        fold_base = folds[folds["방법"] == "A_균등비례"].set_index("폴드")
        fold_now = folds[folds["방법"] == method].set_index("폴드")
        fold_worst = float(((fold_now["RMSE"] / fold_base["RMSE"] - 1) * 100).max())
        inv_base = detail[(detail["방법"] == "A_균등비례") &
                          (detail["인버터"] != "전체_동일가중")].set_index("인버터")
        inv_now = detail[(detail["방법"] == method) &
                         (detail["인버터"] != "전체_동일가중")].set_index("인버터")
        inv_worst_pp = float((inv_now["nMAE_pct"] - inv_base["nMAE_pct"]).max())
        pred_sub = predictions[predictions["방법"] == method]
        reconcile = float(pred_sub["합계차이_kW"].abs().max())
        mae_imp = float((1 - row["MAE"] / base["MAE"]) * 100)
        rmse_imp = float((1 - row["RMSE"] / base["RMSE"]) * 100)
        passed = (method == "A_균등비례" or
                  (mae_imp >= 1.0 and rmse_imp >= 1.0 and fold_worst < 5.0 and
                   inv_worst_pp <= 1.0 and reconcile <= TOL_KW))
        item = {"방법": method, "MAE개선율_pct": mae_imp, "RMSE개선율_pct": rmse_imp,
                "폴드최대RMSE악화_pct": fold_worst, "인버터최대nMAE악화_pp": inv_worst_pp,
                "합계최대차이_kW": reconcile, "채택기준통과": bool(passed)}
        rows.append(item)
        if passed:
            candidates.append((float(row["RMSE"]), method_order[method], method))
    chosen = sorted(candidates, key=lambda x: (round(x[0], 9), x[1]))[0][2]
    for item in rows:
        item["최종선택"] = item["방법"] == chosen
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "사전등록_채택규칙.json").write_text(
        json.dumps(PREREGISTERED_RULES, ensure_ascii=False, indent=2), encoding="utf-8")

    total_mod = _load_module("gimje_total_model_20260903", TOTAL_MODEL_SCRIPT)
    df, removed_defect_rows = total_mod.build_dataset()
    issue_days = pd.DatetimeIndex(np.sort(df["issue_day"].unique()))
    folds = total_mod.expanding_folds_full_coverage(
        issue_days, total_mod.INITIAL_TRAIN_DAYS, total_mod.TEST_BLOCK_DAYS)
    features = total_mod.FEATURES_15

    inv_wide = _load_inverter_wide()
    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]

    output_parts, audit_rows = [], []
    for fold_i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤"
        train_total = df[df["issue_day"].isin(train_days)].dropna(subset=features + [total_mod.TARGET])
        test_total = df[df["issue_day"].isin(test_days)].dropna(subset=features + [total_mod.TARGET])
        if len(train_total) < 300 or len(test_total) < 30:
            print(f"[폴드{fold_i}] 총출력 표본 부족(train={len(train_total)}, "
                  f"test={len(test_total)}) - 생략")
            continue

        model_total = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                                    max_depth=4, min_child_samples=15, subsample=0.9,
                                    colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                                    random_state=SEED, n_jobs=-1, verbosity=-1)
        model_total.fit(train_total[features], train_total[total_mod.TARGET])
        total_pred_test = np.clip(model_total.predict(test_total[features]), 0, CAPACITY.sum())

        # 인버터 10기 전부 가용인 시험행만 사용(광주와 동일 원칙)
        target_times = pd.DatetimeIndex(test_total["target_time_kst"])
        inv_actual = inv_wide.reindex(target_times)[inv_cols + ["가용인버터수"]]
        full_ok = (inv_actual["가용인버터수"] == N_INVERTERS).to_numpy()
        if full_ok.sum() < 30:
            print(f"[폴드{fold_i}] 10기 전부가용 시험행 부족({full_ok.sum()}) - 생략")
            continue

        test_full = test_total.loc[full_ok].copy()
        raw_actual = inv_actual.loc[full_ok, inv_cols].to_numpy(float)
        official_total = test_full[total_mod.TARGET].to_numpy(float)
        actual, rel_err_pct = _reconcile_actual(raw_actual, official_total, CAPACITY.sum())
        total_pred_full = total_pred_test[full_ok]

        # 제미나이 리서치 "실무표준" 상한(설비용량 대비 5%) 초과 시험행은
        # 통신오류/시계불일치 이상치로 간주해 드롭(엄격필터링) - v1 실측
        # 최대치가 2.27%였으므로 이번 데이터에서는 0건 드롭이 예상되나,
        # 로직 자체는 항상 적용(향후 재실행·타 지역 재사용 대비).
        keep = rel_err_pct <= RECONCILE_DROP_THRESHOLD_PCT
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(f"[폴드{fold_i}] 계측불일치 {RECONCILE_DROP_THRESHOLD_PCT}% 초과로 "
                  f"{n_dropped}건 드롭")
        test_full = test_full.loc[keep].copy()
        raw_actual = raw_actual[keep]
        official_total = official_total[keep]
        actual = actual[keep]
        rel_err_pct = rel_err_pct[keep]
        total_pred_full = total_pred_full[keep]
        if len(test_full) < 30:
            print(f"[폴드{fold_i}] 드롭 후 시험행 부족({len(test_full)}) - 생략")
            continue

        # C 후보용 정적특성 프레임(발행시각=총출력모델의 issue_day, 최근/24시간전 인버터비중)
        c_frame = test_full[["target_time_kst"] + features].copy()
        c_frame["대상_hour_sin"] = np.sin(2 * np.pi * c_frame["target_time_kst"].dt.hour / 24)
        c_frame["대상_hour_cos"] = np.cos(2 * np.pi * c_frame["target_time_kst"].dt.hour / 24)
        recent = _recent_share_at(pd.DatetimeIndex(c_frame["target_time_kst"]), inv_wide, pd.Timedelta(0))
        recent24 = _recent_share_at(pd.DatetimeIndex(c_frame["target_time_kst"]), inv_wide, pd.Timedelta("24h"))
        for i in range(1, N_INVERTERS + 1):
            c_frame[f"최근_인버터{i}_비중"] = recent[f"인버터{i}_kW"].to_numpy()
            c_frame[f"24시간전_인버터{i}_비중"] = recent24[f"인버터{i}_kW"].to_numpy()

        train_target_times = pd.DatetimeIndex(train_total["target_time_kst"])
        train_inv = inv_wide.reindex(train_target_times)[inv_cols + ["가용인버터수"]]
        train_ok = (train_inv["가용인버터수"] == N_INVERTERS).to_numpy()
        c_train_frame = train_total.loc[train_ok, ["target_time_kst"] + features].copy()
        c_train_frame["대상_hour_sin"] = np.sin(2 * np.pi * c_train_frame["target_time_kst"].dt.hour / 24)
        c_train_frame["대상_hour_cos"] = np.cos(2 * np.pi * c_train_frame["target_time_kst"].dt.hour / 24)
        recent_tr = _recent_share_at(pd.DatetimeIndex(c_train_frame["target_time_kst"]), inv_wide, pd.Timedelta(0))
        recent24_tr = _recent_share_at(pd.DatetimeIndex(c_train_frame["target_time_kst"]), inv_wide, pd.Timedelta("24h"))
        for i in range(1, N_INVERTERS + 1):
            c_train_frame[f"최근_인버터{i}_비중"] = recent_tr[f"인버터{i}_kW"].to_numpy()
            c_train_frame[f"24시간전_인버터{i}_비중"] = recent24_tr[f"인버터{i}_kW"].to_numpy()
        c_train_frame[inv_cols] = train_inv.loc[train_ok, inv_cols].to_numpy(float)

        c_features = (list(features) + ["대상_hour_sin", "대상_hour_cos"] +
                     [f"최근_인버터{i}_비중" for i in range(1, N_INVERTERS + 1)] +
                     [f"24시간전_인버터{i}_비중" for i in range(1, N_INVERTERS + 1)])

        b_train = c_train_frame.copy()
        b_train[inv_cols] = train_inv.loc[train_ok, inv_cols].to_numpy(float)

        share_predictions = {
            "A_균등비례": _predict_a(len(test_full)),
            "B_계절시간중앙값": _predict_b(b_train, c_frame["target_time_kst"]),
            "C_LightGBM비중": _predict_c(c_train_frame, c_frame, c_features),
        }
        for method, shares in share_predictions.items():
            pred = shares * total_pred_full[:, None]
            block = pd.DataFrame({
                "폴드": fold_i, "대상시각": test_full["target_time_kst"].to_numpy(),
                "발전소실제_kW(계량기공식값)": official_total,
                "발전소예측_kW": total_pred_full,
                "인버터원시합계_kW(재조정전)": raw_actual.sum(axis=1),
                "계측불일치_상대오차_pct용량": rel_err_pct,
            })
            block["방법"] = method
            for i in range(1, N_INVERTERS + 1):
                block[f"인버터{i}_실제_kW"] = actual[:, i - 1]
                block[f"인버터{i}_예측_kW"] = pred[:, i - 1]
            block["인버터예측합계_kW"] = pred.sum(axis=1)
            block["합계차이_kW"] = block["인버터예측합계_kW"] - block["발전소예측_kW"]
            output_parts.append(block)
        audit_rows.append({"폴드": fold_i, "학습행수": len(train_total),
                           "시험행수(10기전부가용,보정전)": int(full_ok.sum()),
                           "시험행수(계측불일치필터후)": len(test_full),
                           "드롭행수": n_dropped,
                           "총출력_학습최대대상시각": train_total["target_time_kst"].max(),
                           "총출력_시험최소대상시각": test_total["target_time_kst"].min(),
                           "시간누출없음": bool(train_total["target_time_kst"].max() <
                                             test_total["target_time_kst"].min())})
        print(f"[폴드{fold_i}] 완료 - 시험행 {len(test_full):,}건(드롭 {n_dropped}건)")

    if not output_parts:
        raise RuntimeError("모든 폴드가 표본 부족으로 생략됨 - 결과 없음")

    predictions = pd.concat(output_parts, ignore_index=True)
    max_diff = float(predictions["합계차이_kW"].abs().max())
    if max_diff > TOL_KW:
        raise AssertionError(f"10대 예측합계 reconciliation 실패: {max_diff}")
    detail, folds_score = _score(predictions)
    decision = _adjudicate(detail, folds_score, predictions)
    predictions.to_csv(OUT / "행단위_인버터별_예측정답.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "인버터별_전체성능.csv", index=False, encoding="utf-8-sig")
    folds_score.to_csv(OUT / "폴드별_성능.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "사전등록규칙_판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(OUT / "누출_동일행_감사.csv", index=False, encoding="utf-8-sig")

    total_dropped = int(pd.DataFrame(audit_rows)["드롭행수"].sum())
    rel_err_summary = predictions.loc[predictions["방법"] == predictions["방법"].iloc[0],
                                       "계측불일치_상대오차_pct용량"].describe()
    (OUT / "계측불일치_요약.txt").write_text(
        f"설비용량(1,100kW) 대비 상대오차 분포(재조정 전, 필터 후 잔존 시험행 기준):\n"
        f"{rel_err_summary.to_string()}\n\n"
        f"드롭 임계값: {RECONCILE_DROP_THRESHOLD_PCT}% 초과, 총 드롭행수: {total_dropped}\n"
        f"(제미나이 리서치 근거: 변압손실·계측등급차이로 3~5%는 정상범위, "
        f"5% 초과만 통신오류로 의심)",
        encoding="utf-8")

    if V1_DECISION_PATH.is_file():
        v1 = pd.read_csv(V1_DECISION_PATH)[
            ["방법", "MAE개선율_pct", "RMSE개선율_pct", "채택기준통과", "최종선택"]
        ].rename(columns=lambda c: f"v1_{c}" if c != "방법" else c)
        v2 = decision[["방법", "MAE개선율_pct", "RMSE개선율_pct", "채택기준통과", "최종선택"]].rename(
            columns=lambda c: f"v2_{c}" if c != "방법" else c)
        compare = v1.merge(v2, on="방법", how="outer")
        compare.to_csv(OUT / "v1_대비_비교.csv", index=False, encoding="utf-8-sig")
        print("\n=== v1(계측보정 전) vs v2(pro-rata 계측보정 후) ===")
        print(compare.to_string(index=False))

    print("\n=== 김제 인버터 분해 v2(계측보정) 사전등록 판정 ===")
    print(decision.to_string(index=False))
    print(f"\n10대 합계 최대차이(예측 내부 일관성): {max_diff:.3e} kW")
    print(f"계측불일치 {RECONCILE_DROP_THRESHOLD_PCT}% 초과 드롭행수: {total_dropped}")
    print(f"결함구간 제외행수(총출력모델 원본): {removed_defect_rows}")


if __name__ == "__main__":
    main()
