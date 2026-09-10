# -*- coding: utf-8 -*-
"""일간 D+1 라이브 예측 실행 드라이버(초단기·단기엔 있었지만 일간엔 없던 것).

## 왜 필요한가(08-27 사용자 지적으로 발견한 공백)
`shadow_readiness_일간_v2_v7연계_2026-08-26.py`는 "입력이 준비됐는가"만
판정하지 실제로 예측을 뽑지 않는다. 초단기·단기는 `assemble_and_predict()`
가 있어 readiness 통과 후 바로 예측까지 가는데, 일간은 이 마지막 단계
자체가 없었다 — readiness가 `ready`여도 실제로는 예측을 실행할 수
없는 상태였다.

## 재구현 없음 — 기존 검증된 조각 두 개를 그대로 이어붙였을 뿐
1. **NWP·ASOS·설비 원자료 조립**: `live_feature_assembler_단기_v1_
   2026-08-26.py::build_live_hourly()`를 그대로 재사용(BSRN 정제·VEC
   원형보간·추정_일조시간_hr까지 이미 검증된 그 함수 그대로). 이
   함수가 만드는 컬럼명이 `e2e_retrain_v5_공식B_v1_2026-08-24.py::
   build_daily_dataset_v5()`가 기대하는 원자료 컬럼명과 정확히 같다
   (DSWRF·DSWRFLX_bsrn정제·DIFSWRF_bsrn정제·TCDC/LCDC/MCDC/HCDC·
   SKY/REH/POP/TMP/WSD·기상청관측_*·EQUIPMENT_COLS) — 그래서 학습 때
   쓴 것과 같은 집계공식(`NWP_AGG_SUM_COLS`/`MEAN`/`MIN`)을 그대로
   가져다 쓸 수 있다.
2. **7일전·2일전·30일 이동통계**: `daily_dataset_v7_라이브연계_로더_
   2026-08-26.py`가 가리키는 v7 parquet(`집계_일간_실제발전량_v5.
   parquet`)의 `일간발전량_kWh`·`부분가용일`을 `corrected_dataset()`과
   똑같은 규칙(`history = raw["일간발전량_kWh"].where(partial<1)`)으로
   읽어, 목표일(D+1) 기준 7일전·2일전 값과 2일전 기준 7일·30일
   이동통계를 계산한다. 이건 계산식 자체가 짧아서(4줄) 함수 호출로
   재사용하기보다 학습 코드와 완전히 동일한 수식을 그대로 복제했다
   (`e2e_retrain_v5_공식B_v1_2026-08-24.py` 303~310행과 1:1 대응 —
   대조 가능하도록 원본 줄번호를 주석에 남김).

## 가짜성공 방지(초단기·단기와 동일 원칙)
`bundle["features"]` 중 하나라도 NaN이면 무조건 `상태:"대기"`를 반환하고
예측을 계산하지 않는다. LightGBM이 NaN을 수학적으로 받을 수 있다는
것과 운영 입력이 실제로 준비됐다는 건 다르다(golden replay·다른
shadow_predict_*와 같은 원칙).

## 사용법
```
python shadow_predict_일간_v1_2026-08-27.py
```
저장 위치는 다른 shadow_predict_*와 같은 DB(`outputs/shadow_predictions/
shadow_predictions.sqlite3`), 같은 `ensure_db()`/스키마 재사용.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "outputs" / "운영모델_v2_2026-08-25" / "운영모델_일간_D+1.joblib"
V7_PARQUET = ROOT / "outputs" / "v7_라이브연계_2026-08-26" / "집계_일간_실제발전량_v5.parquet"
V6_PARQUET = ROOT / "outputs" / "v6_일간구멍보정_2026-08-26" / "집계_일간_실제발전량_v5.parquet"
KST = ZoneInfo("Asia/Seoul")
SITE_ELEVATION_M = 21.0  # e2e 학습 hourly CSV의 site_elevation_dem_m과 동일(상수, 08-27 직접 확인)


def _load(name: str, filename: str, base: Path = ROOT):
    spec = importlib.util.spec_from_file_location(name, base / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


asm = _load("daily_pred_asm", "live_feature_assembler_단기_v1_2026-08-26.py")
shadow_common = _load("daily_pred_shadow_common", "shadow_predict_단기_v1_2026-08-26.py")
e2e = _load("daily_pred_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")  # NWP_AGG_*·EQUIPMENT_COLS 상수 재사용
infer = asm.infer  # production_inference_utils(predict_kw)


def _load_daily_history() -> pd.DataFrame:
    path = V7_PARQUET if V7_PARQUET.is_file() else V6_PARQUET
    if not path.is_file():
        raise RuntimeError("v6/v7 일간 데이터셋이 없다 — 먼저 생성할 것.")
    raw = pd.read_parquet(path)
    raw.index = pd.to_datetime(raw.index)
    return raw.sort_index()


def _daily_agg_for_date(hourly: pd.DataFrame, day: pd.Timestamp) -> dict:
    """build_daily_dataset_v5()의 NWP_AGG_* 집계를 특정 하루에만 적용."""
    day_rows = hourly[hourly.index.normalize() == day]
    out: dict = {}
    for c, _how in e2e.NWP_AGG_SUM_COLS.items():
        out[f"목표일예보_{c}_sum"] = day_rows[c].sum(min_count=1) if c in day_rows else np.nan
    for c in e2e.NWP_AGG_MEAN_COLS:
        out[f"목표일예보_{c}_mean"] = day_rows[c].mean() if c in day_rows else np.nan
    for c, _how in e2e.NWP_AGG_MIN_COLS.items():
        out[f"목표일예보_{c}_min"] = day_rows[c].min() if c in day_rows else np.nan
    for c in ("TCDC", "POP", "WSD", "TMP"):
        out[f"목표일예보_{c}_max"] = day_rows[c].max() if c in day_rows else np.nan
    out["목표일_DIFSWRF_유효개수"] = float(day_rows[e2e.DIF].notna().sum()) if e2e.DIF in day_rows else 0.0
    out[f"목표일예보_{e2e.DIF}_sum_결측여부"] = float(pd.isna(out[f"목표일예보_{e2e.DIF}_sum"]))
    return out


def _equipment_2day_avg(hourly: pd.DataFrame, two_days_before: pd.Timestamp) -> dict:
    """build_daily_dataset_v5() 336~337행과 동일 의미: 목표일 2일전 하루의
    설비·관측 평균을 "2일전평균_*"로 낸다(shift(2)를 짧은 라이브 창에서
    안전하게 재현하려고 groupby+shift 대신 해당 날짜만 직접 필터)."""
    cols = e2e.EQUIPMENT_COLS + ["inverters_available"]
    day_rows = hourly[hourly.index.normalize() == two_days_before]
    means = day_rows[[c for c in cols if c in day_rows.columns]].mean(numeric_only=True)
    return {f"2일전평균_{c}": means.get(c, np.nan) for c in cols}


def build_live_daily_row(bundle: dict, capacity_kw: float) -> tuple[pd.DataFrame, pd.Timestamp]:
    now = pd.Timestamp.now(tz=KST)
    target_day = (now.tz_localize(None) + pd.Timedelta(days=1)).normalize()

    # 72시간 뒤(목표일-2일 전체 커버)~48시간 앞(목표일 전체 커버)으로 넉넉히.
    hourly = asm.build_live_hourly(now, lookback_hours=96, future_hours=48)
    if hourly.empty:
        raise RuntimeError("라이브 hourly 조립 실패(원자료 없음)")

    history = _load_daily_history()
    partial = history.get("부분가용일", pd.Series(0, index=history.index)).fillna(0)
    actual_hist = history["일간발전량_kWh"].where(partial < 1)

    def _h(day: pd.Timestamp) -> float:
        return float(actual_hist.get(day, np.nan))

    lag7_day = target_day - pd.Timedelta(days=7)
    lag2_day = target_day - pd.Timedelta(days=2)
    window7 = [lag2_day - pd.Timedelta(days=d) for d in range(6, -1, -1)]  # lag2-6 ~ lag2
    window30 = [lag2_day - pd.Timedelta(days=d) for d in range(29, -1, -1)]  # lag2-29 ~ lag2
    vals7 = pd.Series([_h(d) for d in window7])
    vals30 = pd.Series([_h(d) for d in window30])

    row: dict = {}
    # ↓ e2e_retrain_v5_공식B_v1_2026-08-24.py 303~310행과 동일 수식
    row["7일전_일간발전량_kWh"] = _h(lag7_day)
    row["7일전_결측여부"] = float(pd.isna(row["7일전_일간발전량_kWh"]))
    row["2일전_일간발전량_kWh"] = _h(lag2_day)
    row["2일전_결측여부"] = float(pd.isna(row["2일전_일간발전량_kWh"]))
    row["2일전기준_7일이동평균_kWh"] = vals7.mean() if vals7.notna().sum() >= 4 else np.nan
    row["2일전기준_30일이동평균_kWh"] = vals30.mean() if vals30.notna().sum() >= 15 else np.nan
    row["2일전기준_30일표준편차_kWh"] = vals30.std() if vals30.notna().sum() >= 15 else np.nan

    row.update(_daily_agg_for_date(hourly, target_day))
    row.update(_equipment_2day_avg(hourly, lag2_day))

    doy = target_day.dayofyear
    row["목표일_연주기_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["목표일_연주기_cos"] = np.cos(2 * np.pi * doy / 365.25)
    row["목표일_월"] = target_day.month
    row["목표일_요일"] = target_day.dayofweek
    row["해발고도_m"] = SITE_ELEVATION_M
    row["설비용량_kW"] = capacity_kw

    frame = pd.DataFrame([row], index=[target_day])
    # bundle 특성 중 위에서 못 채운 게 있으면(안전망) 결측으로라도 만든다
    # — 컬럼 자체가 없으면 KeyError로 죽는 걸 방지(초단기/단기와 동일 패턴).
    for c in bundle["features"]:
        if c not in frame.columns:
            frame[c] = np.nan
    return frame, target_day


def assemble_and_predict(bundle: dict, capacity_kw: float) -> dict:
    try:
        frame, target_day = build_live_daily_row(bundle, capacity_kw)
    except Exception as exc:
        return {"상태": "실패", "사유": f"{type(exc).__name__}: {exc}"}

    row = frame[bundle["features"]]
    missing = [f for f in bundle["features"] if pd.isna(row.iloc[0][f])]
    if missing:
        return {
            "상태": "대기", "사유": "운영 입력 미충족",
            "예측대상일": str(target_day.date()), "특성결측수": len(missing),
            "결측특성": missing, "번들버전": bundle.get("번들버전"),
        }
    pred = infer.predict_kw(bundle, row)
    pred_kwh = float(np.clip(pred[0], 0, bundle["clip_상한"]))
    return {
        "상태": "성공", "예측대상일": str(target_day.date()),
        "예측_kWh": round(pred_kwh, 2), "특성결측수": 0,
        "번들버전": bundle.get("번들버전"),
    }


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    bundle = joblib.load(BUNDLE_PATH)

    result = assemble_and_predict(bundle, capacity_kw)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    conn = shadow_common.ensure_db()
    now = pd.Timestamp.now(tz=KST).isoformat()
    issue = pd.Timestamp.now(tz=KST).tz_localize(None).floor("D") + pd.Timedelta(hours=10)
    target = result.get("예측대상일")
    reason = result.get("사유") or (
        json.dumps(result.get("결측특성", []), ensure_ascii=False) if result.get("결측특성") else None
    )
    conn.execute(
        "UPDATE shadow_predictions SET status='superseded' "
        "WHERE tier='일간' AND status IN ('warming_up','blocked','대기','실패')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO shadow_predictions "
        "(predicted_at,tier,horizon_h,issue_time,target_time,predicted_kw,"
        "n_missing_features,bundle_version,status,reason) "
        "VALUES (?, '일간', 24, ?, ?, ?, ?, ?, ?, ?)",
        (now, str(issue), target, result.get("예측_kWh"),
         result.get("특성결측수"), bundle.get("번들버전"), result["상태"], reason),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
