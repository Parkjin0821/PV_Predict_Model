# -*- coding: utf-8 -*-
"""인버터5 결함구간 처리방침 확정 비교: A_무처리 / B_구간제외 / C_용량가중추정보정.

## 사용자 확정 사항(그대로 구현)
- **공식 주분석은 B_구간제외**다(실제 정지·통신누락을 구분할 근거가
  없어서). C는 **민감도 분석 후보로만** 유지한다.
- **C는 단순 5/가용대수가 아니라 용량가중**: 보정계수 =
  전체정격용량합 / 그 순간 가용 인버터들의 정격용량 합. 정격용량 출처는
  `inverter_capacity_availability_v1_2026-08-21.py`가 만든 표를 그대로
  쓴다(★출처는 Blockdata 등록값 단일 소스뿐 — 원본 설비자료에는 정격용량
  컬럼이 없어 "혼합하지 않을" 두 번째 소스가 존재하지 않는다. 이 사실을
  결과표에 그대로 남긴다).
- **C 산출값은 "복구 실측"이 아니라 "추정 보정값"**이라고 모든 산출물에
  명시한다(정답 파일 자체는 바꾸지 않음, 이 스크립트의 학습 라벨에서만
  임시로 씀).
- **A·B·C를 이상구간을 포함한 연속 6폴드의 완전히 동일한 시험행**에서
  비교한다. 시험행은 항상 **대상시각(미래) 가용인버터수=5**인 행만
  (정책과 무관, 고정) — 유일하게 모호하지 않은 정답.
- **폴드별·시간대별 유효행 수·커버리지, MAE, RMSE, 최대악화폭을 전부
  출력**한다.
- **결함구간이 시험폴드에 포함되지 않아 평가가 우회되면 실패 처리**한다
  — 폴드 "2_이상구간"이 바로 결함구간 자체를 시험창으로 삼도록 6폴드를
  구성했으므로 구조적으로는 포함되나, 그 폴드의 완전가용 시험행이
  0건이면 "그 폴드는 평가 실패"로 명시 보고한다(조용히 건너뛰지 않음).
- **이 스크립트 실행 결과로 기존 공식모델·2차 결과·KPX 예상 정산액을
  갱신하지 않는다** — 정책 확정을 위한 별도 비교 산출물로만 남긴다.

## 데이터 출처(★전부 v5★)
- 초단기(15분): `outputs/v5_복구_2026-08-21/집계_15분_자료_v5.parquet`
- 단기(1시간): `harness.DATASETS["v3_고정tm"]`의 NWP·ASOS는 그대로 두고
  **`plant_output_kw`·`inverters_available` 컬럼만 v5의
  `집계_1시간_자료_v5.parquet`로 치환**한다(그 컬럼만 원래 min_count=5
  결함의 영향을 받았고, NWP·ASOS는 인버터와 무관해 영향 없음 — 이미
  확인함, 두 파일의 시간축 공통 17,033/17,034행).
- 인버터별 가용성·용량가중 보정계수:
  `inverter_capacity_availability_v1_2026-08-21.py` 산출물(먼저 실행 필요).

## 6폴드 정의(KPX 재현 백테스트와 동일 — 일관성 유지)
1_여름 / 2_이상구간(신규포함) / 3_가을 / 4_겨울 / 5_봄 / 6_초여름

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\03_모델학습\\현재_종합파이프라인"
python inverter_capacity_availability_v1_2026-08-21.py   # 선행 필요(아직 안 돌렸으면)
python defect_policy_comparison_v1_2026-08-21.py
```
로컬 재학습만(API 없음). 6폴드×3정책×3(초단기2수평+단기1수평)이라
십수 분 소요될 수 있다.

## 산출물 (`outputs/결함구간_정책비교_v1_2026-08-21/`)
- `커버리지_폴드별.csv` — 폴드×티어×수평: 전체대상행수·완전가용시험행수·
  커버리지%·평가실패여부
- `커버리지_시간대별.csv` — 폴드×티어×시(hour): 완전가용 비율
- `판정표_폴드별.csv` — 폴드×티어×수평×정책: MAE·RMSE
- `판정표_요약.csv` — 티어×수평×정책 가중평균 + 최대악화폭(%, vs A)
- `요약.txt`
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
V5_DIR = ROOT / "outputs" / "v5_복구_2026-08-21"
OUT = ROOT / "outputs" / "결함구간_정책비교_v2_2026-08-24_폴드경계정정"

CAPACITY_SOURCE = "Blockdata 등록값(GET /data/6715, 08-19 스냅샷, 명판·설비대장 미검증) — 원본 설비자료엔 정격용량 컬럼 없음, 혼합할 2차 출처 없음"
N_INVERTERS = 5
POLICIES = ["A_무처리", "B_구간제외(공식)", "C_용량가중추정보정(민감도)"]

FOLD_WINDOWS = [
    ("1_여름", "2025-06-01", "2025-08-14"),
    ("2_이상구간(신규포함)", "2025-08-15", "2025-11-16"),
    ("3_가을_결함종료후", "2025-11-17", "2025-12-14"),
    ("4_겨울", "2025-12-15", "2026-02-14"),
    ("5_봄", "2026-02-15", "2026-04-14"),
    ("6_초여름", "2026-04-15", "2026-08-04"),
]
# ★08-24 정정★: 최초 버전은 이상구간을 08-15~10-20으로, 가을을 10-21
# 시작으로 잘못 나눴다. v5 자체 진단(결측구간_목록.csv)에는 인버터5
# 실제 결함이 08-19~11-16(90일)이라고 정확히 나와 있었는데, 그걸 이
# 폴드 경계에는 반영하지 않은 내 실수였다 — Codex의 08-24
# `fold_hour_prevalidation_v1_2026-08-24.py`가 같은 문제(가을 폴드
# 오염, 초단기 커버리지 69.82%)를 독립적으로 찾아냈다. 경계를 실제
# 결함종료일(11-16) 다음날(11-17)로 고쳐 재실행한다 — 원래 결과(가을
# +9.81% 악화 등)가 이 오염의 산물이었는지 재확인하는 게 목적.

# ★08-24 추가★: Codex가 채택한 DIFSWRF 결측처리 원칙(원값 NaN을
# LightGBM 자체 처리 + 결측여부 플래그, difswrf_missing_strategy_
# comparison_v1_2026-08-24.py의 "C")을 공유 harness에는 아직 안
# 넣었으므로 여기서 로컬로 반영한다 — 안 하면 가을 폴드 저커버리지가
# 폴드경계 문제와 DIFSWRF 문제 중 무엇 때문인지 못 가른다.
DIFSWRF_COL = "DIFSWRF_bsrn정제"
EXTRA_NATIVE_MISSING_OK = {DIFSWRF_COL}


def _load(name: str, filename: str, rel: str = "."):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("dpc_harness", "backtest_harness_v1_2026-08-20밤.py")
ultra = _load("dpc_ultra", "train_ultra_short_official_v1_2026-08-21.py")


def require_v5_outputs() -> None:
    need = ["집계_15분_자료_v5.parquet", "집계_1시간_자료_v5.parquet",
            "용량가중_보정계수_15분.parquet", "용량가중_보정계수_1시간.parquet"]
    missing = [f for f in need if not (V5_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"필요한 v5 산출물이 없다: {missing}\n"
            "먼저 02_전처리/rebuild_plant_v5_recovered_2026-08-21.py와 "
            "inverter_capacity_availability_v1_2026-08-21.py를 실행할 것.")


def load_ultra_frame(horizon: int) -> pd.DataFrame:
    quarter = pd.read_parquet(V5_DIR / "집계_15분_자료_v5.parquet").rename(columns=ultra.RENAME_15MIN)
    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                         parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    frame = ultra.build_ultra_short_frame(quarter, hourly, horizon)
    factor = pd.read_parquet(V5_DIR / "용량가중_보정계수_15분.parquet")["용량가중_보정계수"]
    frame["_목표_가용인버터수"] = quarter["가용인버터수"].shift(-horizon * 4)
    frame["_목표_용량가중보정계수"] = factor.reindex(quarter.index).shift(-horizon * 4).to_numpy()
    if DIFSWRF_COL in frame.columns:
        frame[f"{DIFSWRF_COL}_결측여부"] = frame[DIFSWRF_COL].isna().astype(float)
    return frame


def load_short_frame(horizon: int) -> pd.DataFrame:
    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                         parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    v5_1h = pd.read_parquet(V5_DIR / "집계_1시간_자료_v5.parquet")
    common = hourly.index.intersection(v5_1h.index)
    hourly = hourly.loc[common].copy()
    hourly["plant_output_kw"] = v5_1h.loc[common, "발전출력_kW"]
    hourly["inverters_available"] = v5_1h.loc[common, "가용인버터수"]

    candidate_cols = harness.FEATURE_SETS["전체후보"]
    frame = harness.build_frame(hourly, horizon, candidate_cols)
    factor = pd.read_parquet(V5_DIR / "용량가중_보정계수_1시간.parquet")["용량가중_보정계수"]
    offset = horizon - 1  # harness.build_frame의 시작라벨 (H-1) 규약과 동일
    frame["_목표_가용인버터수"] = hourly["inverters_available"].shift(-offset)
    frame["_목표_용량가중보정계수"] = factor.reindex(hourly.index).shift(-offset).to_numpy()
    if DIFSWRF_COL in frame.columns:
        frame[f"{DIFSWRF_COL}_결측여부"] = frame[DIFSWRF_COL].isna().astype(float)
    return frame


def make_train(daylight: pd.DataFrame, policy: str, capacity_kw: float) -> pd.DataFrame:
    df = daylight.copy()
    if policy == "A_무처리":
        df["_학습타깃_kW"] = df["목표_발전출력_kW"]
    elif policy == "B_구간제외(공식)":
        df = df[df["_목표_가용인버터수"] >= N_INVERTERS].copy()
        df["_학습타깃_kW"] = df["목표_발전출력_kW"]
    elif policy == "C_용량가중추정보정(민감도)":
        df = df.copy()
        scaled = (df["목표_발전출력_kW"] * df["_목표_용량가중보정계수"]).clip(upper=capacity_kw)
        df["_학습타깃_kW"] = np.where(
            df["_목표_가용인버터수"] >= N_INVERTERS, df["목표_발전출력_kW"], scaled)
    else:
        raise ValueError(policy)
    return df


def run_tier(tier: str, horizon, frame: pd.DataFrame, config: dict, capacity_kw: float, seed: int,
             candidate_cols: list[str], target_col: str = "목표_발전출력_kW",
             daylight_col: str = "목표_낮시간") -> tuple[list[dict], list[dict], list[dict]]:
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in (target_col, daylight_col)]
    daylight = frame[frame[daylight_col] > 0]

    coverage_rows, hourly_rows, perf_rows = [], [], []
    for fold, s, e in FOLD_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train_all = daylight[daylight.index < start]
        test_all = daylight[(daylight.index >= start) & (daylight.index < end)]
        test_full = test_all[test_all["_목표_가용인버터수"] >= N_INVERTERS]

        total_target_rows = len(test_all)
        full_rows = len(test_full)
        cov = full_rows / total_target_rows * 100 if total_target_rows else 0.0
        eval_failed = bool("이상구간" in fold and full_rows == 0)
        coverage_rows.append({
            "티어": tier, "수평": horizon, "폴드": fold,
            "전체대상행수": total_target_rows, "완전가용시험행수": full_rows,
            "커버리지_pct": round(cov, 2), "평가실패(결함구간우회)": eval_failed,
        })
        if total_target_rows:
            target_time = test_all.index + (pd.Timedelta(hours=horizon) if tier == "초단기"
                                            else pd.Timedelta(hours=horizon - 1))
            hrs = pd.Series(target_time.hour, index=test_all.index)
            full_flag = (test_all["_목표_가용인버터수"] >= N_INVERTERS)
            for h, idx in hrs.groupby(hrs).groups.items():
                sub_full = full_flag.loc[idx]
                hourly_rows.append({
                    "티어": tier, "수평": horizon, "폴드": fold, "대상시_hour": int(h),
                    "전체행수": len(idx), "완전가용행수": int(sub_full.sum()),
                    "완전가용비율_pct": round(sub_full.mean() * 100, 1),
                })

        if len(train_all) < 300 or len(test_full) < 30:
            print(f"  [{tier} +{horizon} {fold}] 완전가용 시험행 부족({len(test_full)}) — 정책비교 생략"
                  + (" ★평가실패★" if eval_failed else ""))
            continue

        for policy in POLICIES:
            train_variant = make_train(train_all, policy, capacity_kw)
            # select_features_in_fold는 컬럼명 "목표_발전출력_kW"를 타깃으로 고정 참조하므로,
            # 정책별 실제 학습타깃(_학습타깃_kW, 예: C의 스케일업 라벨)을 그 이름으로 넘겨
            # "그 정책이 실제로 쓸 라벨"과의 상관계수로 특성을 고른다.
            sel_input = train_variant.drop(columns=["목표_발전출력_kW"]).assign(
                목표_발전출력_kW=train_variant["_학습타깃_kW"]).dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(sel_input, candidate_cols, 0.3, True, True)
            features = base_cols + chosen
            native_ok = harness.NATIVE_MISSING_OK | EXTRA_NATIVE_MISSING_OK
            required = [c for c in features if c not in native_ok] + ["_학습타깃_kW"]
            train = train_variant.dropna(subset=required)
            test = test_full.dropna(subset=[c for c in features if c not in native_ok]
                                    + [target_col])
            if len(train) < 300 or len(test) < 30:
                continue
            model = (ultra.make_model("LightGBM", seed) if tier == "초단기"
                     else harness.make_model("XGBoost", seed))
            model.fit(train[features], train["_학습타깃_kW"])
            pred = np.clip(model.predict(test[features]), 0, capacity_kw)
            y = test[target_col].to_numpy()
            mae = float(np.mean(np.abs(y - pred)))
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            perf_rows.append({
                "티어": tier, "수평": horizon, "폴드": fold, "정책": policy,
                "학습행수": len(train), "시험행수": len(test),
                "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
            })
        print(f"  [{tier} +{horizon} {fold}] 완료 (시험행 {len(test_full):,}, 커버리지 {cov:.1f}%)")

    return coverage_rows, hourly_rows, perf_rows


def main() -> None:
    require_v5_outputs()
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    print(f"정격용량 출처: {CAPACITY_SOURCE}\n")

    all_cov, all_hourly, all_perf = [], [], []

    print("=== 초단기(+1h·+2h·+3h·+4h — 전 수평) ===")
    ultra_candidates = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
    for horizon in (1, 2, 3, 4):
        frame = load_ultra_frame(horizon)
        cov, hrl, perf = run_tier("초단기", horizon, frame, config, capacity_kw, seed, ultra_candidates)
        all_cov += cov; all_hourly += hrl; all_perf += perf

    print("\n=== 단기(+1h·+24h·+48h — 전 수평) ===")
    short_candidates = harness.FEATURE_SETS["전체후보"]
    for horizon in (1, 24, 48):
        frame = load_short_frame(horizon)
        cov, hrl, perf = run_tier("단기", horizon, frame, config, capacity_kw, seed, short_candidates)
        all_cov += cov; all_hourly += hrl; all_perf += perf

    cov_df = pd.DataFrame(all_cov)
    hourly_df = pd.DataFrame(all_hourly)
    perf_df = pd.DataFrame(all_perf)
    cov_df.to_csv(OUT / "커버리지_폴드별.csv", index=False, encoding="utf-8-sig")
    hourly_df.to_csv(OUT / "커버리지_시간대별.csv", index=False, encoding="utf-8-sig")
    perf_df.to_csv(OUT / "판정표_폴드별.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 커버리지(폴드별) ===")
    print(cov_df.to_string(index=False))

    # ── 요약: 가중평균 + 정책별 최대악화폭(폴드 중 A 대비 RMSE가 가장 나빠진 비율) ──
    summary_rows = []
    for (tier, h), g in perf_df.groupby(["티어", "수평"]):
        piv = g.pivot_table(index="폴드", columns="정책", values="RMSE_kW")
        wpiv = g.pivot_table(index="폴드", columns="정책", values="시험행수")
        for policy in POLICIES:
            if policy not in piv.columns:
                continue
            valid = piv[policy].notna() & (piv.index.isin(wpiv.index))
            w = wpiv.loc[valid.index[valid], policy] if policy in wpiv.columns else None
            rmse_w = np.average(piv.loc[valid, policy], weights=w) if w is not None and len(w) else np.nan
            if "A_무처리" in piv.columns:
                degrade = ((piv[policy] - piv["A_무처리"]) / piv["A_무처리"] * 100).dropna()
                worst = float(degrade.max()) if len(degrade) else np.nan
                worst_fold = degrade.idxmax() if len(degrade) else None
            else:
                worst, worst_fold = np.nan, None
            summary_rows.append({
                "티어": tier, "수평": h, "정책": policy,
                "RMSE_kW_가중평균": round(float(rmse_w), 3) if pd.notna(rmse_w) else None,
                "최대악화폭_vsA_pct": round(worst, 2) if pd.notna(worst) else None,
                "최대악화_폴드": worst_fold,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "판정표_요약.csv", index=False, encoding="utf-8-sig")
    print("\n=== 정책별 요약(가중평균 RMSE + A 대비 최대악화폭) ===")
    print(summary_df.to_string(index=False))

    fails = cov_df[cov_df["평가실패(결함구간우회)"]]
    lines = [
        "인버터5 결함구간 처리방침 확정 비교 — 요약",
        f"정격용량 출처: {CAPACITY_SOURCE}",
        "공식 주분석: B_구간제외 / C는 민감도 분석 후보만(추정 보정값, 복구 실측 아님)",
        f"평가실패(결함구간이 시험폴드에서 우회됨) 건수: {len(fails)}",
        "", "=== 커버리지 ===", cov_df.to_string(index=False),
        "", "=== 정책별 요약 ===", summary_df.to_string(index=False),
        "", "★이 결과로 기존 공식모델·2차 결과·KPX 예상 정산액을 갱신하지 않았음★",
    ]
    (OUT / "요약.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")
    if len(fails):
        print(f"\n★평가실패 {len(fails)}건 — 결함구간이 시험폴드에서 우회됨. 아래 참고★")
        print(fails.to_string(index=False))


if __name__ == "__main__":
    main()
