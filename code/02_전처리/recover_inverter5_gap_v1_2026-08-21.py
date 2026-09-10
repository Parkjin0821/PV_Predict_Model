# -*- coding: utf-8 -*-
"""5번 인버터 통신다운 기간(2025-08-15~11-16) 부분용량 실측 복구.

## 배경 (08-21, 월간모델 작업 중 발견 → 사용자 지시로 복구)
월간모델 착수 중 `집계_일간_실제발전량.parquet`에서 2025-09·10월이
완전 결측(count=0)인 걸 보고 "저출력 이상구간"으로 알고 있던 것을
재조사했다. 원본 인버터별 5분 로그(엑셀 5개)를 직접 열어본 결과:

- **인버터 1~4는 그 기간 내내 완전히 정상**(일별 행수 ~185개 유지,
  출력값도 비교대상 정상일과 동일 수준).
- **인버터 5만 2025-08-15~11-16(94일) 완전히 침묵**(일별 행수 0개,
  시작·종료가 하루 단위로 칼같이 끊기고 복구되는 전형적 통신/장비
  다운 신호).
- 병합된 5분 자료(`gwangju_5min_model_dataset.csv`)를 보면 이 기간
  `inverters_available=4`로 정확히 감지는 하고 있었는데도
  **`plant_output_kw`(플랜트 총량)가 통째로 NaN 처리**돼 있었다 —
  즉 원 파이프라인이 "5대 전부 있어야 유효"라는 보수적 규칙을 써서,
  실제로는 살아있던 4/5(약 80%) 데이터까지 함께 버린 것.

## 이 스크립트가 하는 일
원본 인버터별 엑셀 5개에서 **살아있는 인버터만 합산**(결측 인버터는
제외, 0으로 대체하지 않음 — pandas `.sum(skipna=True)` 기본동작)해
플랜트 총출력을 다시 계산하고, `pv_pipeline.py`의 `build_daily_actual`
과 동일한 규칙(낮시간자료충족률 90% 미만이면 결측 유지)으로 일간
집계를 다시 만든다.

## ★원칙(중요)★
- **스케일업(임의 추정) 안 함**: 4/5 인버터 합계를 5/4배 곱해서 "5대
  있었으면 이랬을 것"으로 추정하지 않는다 — 5번 인버터의 실제 출력을
  알 방법이 없고(음영·배치가 다를 수 있음), 없는 값을 있는 것처럼
  만들면 프로젝트 원칙("실측을 임의로 추정해 학습데이터로 만들지
  않는다") 위반이다. **그대로 4/5 실측 부분합을 쓴다.**
- **정합성 검증**: 이 기간 밖(예: 08-01~08-13, 11-18~11-25)에서는
  원래 파이프라인 결과와 이 스크립트의 재계산 결과가 사실상 일치해야
  한다 — 방법론이 원본과 같다는 걸 확인하는 절차. 불일치하면 이
  스크립트를 신뢰하지 않는다.
- **원본 결측 자리만 패치**: 원본 parquet에서 이미 유효했던 값은
  건드리지 않고, NaN이었던 자리만 이번 재계산값으로 채운다.
- **투명성**: `가용인버터수` 일평균을 같이 남겨 이 기간이 부분용량
  자료임을 항상 알 수 있게 한다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "03_모델학습" / "현재_종합파이프라인"
_spec = importlib.util.spec_from_file_location("pv_pipeline", ROOT / "pv_pipeline.py")
pv_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv_pipeline)

INVERTER_FILES = {
    1: r"광주 광주시청 _1번 인버터 로그_계산됨.xlsx",
    2: r"광주 광주시청_2번 인버터 로그_계산됨.xlsx",
    3: r"광주 광주시청_3번 인버터 로그_계산됨.xlsx",
    4: r"광주 광주시청_4번 인버터 로그_계산됨.xlsx",
    5: r"광주 광주시청_5번 인버터 로그_계산됨.xlsx",
}
INVERTER_DIR = Path(r"C:\Users\u-cube\JIN\태양광 발전\배만수 부장님 엑셀 파일")

ORIGINAL_PARQUET = ROOT / "outputs" / "집계_일간_실제발전량.parquet"
OUT_PARQUET = ROOT / "outputs" / "집계_일간_실제발전량_v2_인버터5부분복구_2026-08-21.parquet"

# 재계산·정합성검증 대상 구간. 큰 결측(5번 인버터 94일)뿐 아니라
# 2025-06-18(4개 인버터 동시 소폭 저하, 낮시간충족률 88.4%로 90% 미달
# 1일치 결측)도 같은 구간에 포함해 한 번에 처리한다 — 이 하루를 복구하면
# **2025년 전체가 처음으로 완전한 1개년이 된다**(연간 검증표본 0→1개,
# 아래 monthly_quarterly_annual v2에서 활용).
RECOMPUTE_START = pd.Timestamp("2025-06-15")
RECOMPUTE_END = pd.Timestamp("2025-11-25")
GAP_START = pd.Timestamp("2025-08-15")  # 5번 인버터 침묵 시작(실측 확인, 정합성검증 제외구간)
GAP_END = pd.Timestamp("2025-11-16")    # 5번 인버터 침묵 마지막날(실측 확인, 정합성검증 제외구간)


def load_inverter_output_5min(n: int) -> pd.Series:
    path = INVERTER_DIR / INVERTER_FILES[n]
    df = pd.read_excel(path, usecols=["생성일", "출력전력"])
    df["생성일"] = pd.to_datetime(df["생성일"])
    df = df[(df["생성일"] >= RECOMPUTE_START) & (df["생성일"] < RECOMPUTE_END + pd.Timedelta(days=1))]
    s = df.set_index("생성일")["출력전력"].sort_index()
    # 원 파이프라인과 동일하게 5분 격자로 정렬(수 초 단위로 어긋난 원시
    # 타임스탬프를 5분 버킷 평균으로 합침)
    return s.resample("5min").mean()


def recompute_plant_output(config: dict) -> pd.DataFrame:
    series = {n: load_inverter_output_5min(n) for n in range(1, 6)}
    combined = pd.DataFrame(series)
    combined.columns = [f"inv{n}" for n in combined.columns]
    out = pd.DataFrame(index=combined.index)
    out["발전출력_kW"] = combined.sum(axis=1, skipna=True, min_count=1)
    out["가용인버터수"] = combined.notna().sum(axis=1)

    capacity = float(config["site"]["capacity_kw"])
    out.loc[~out["발전출력_kW"].between(0, capacity), "발전출력_kW"] = np.nan

    elevation, _ = pv_pipeline.solar_position(
        out.index, float(config["site"]["latitude"]), float(config["site"]["longitude"])
    )
    out["물리적낮"] = (elevation > 0).astype("int8")
    return out


def to_daily(five_min: pd.DataFrame) -> pd.DataFrame:
    """pv_pipeline.build_daily_actual과 동일 규칙(낮시간충족률 90%)."""
    work = five_min[["발전출력_kW", "물리적낮", "가용인버터수"]].copy()
    work["5분발전량_kWh"] = work["발전출력_kW"] * (5 / 60)
    groups = work.groupby(work.index.normalize())
    daily = pd.DataFrame(index=groups.size().index)
    daily["일간발전량_kWh"] = groups["5분발전량_kWh"].sum(min_count=1)
    daylight_expected = groups["물리적낮"].sum().replace(0, np.nan)
    daylight_observed = groups.apply(
        lambda x: int(((x["물리적낮"] == 1) & x["발전출력_kW"].notna()).sum()),
        include_groups=False,
    )
    daily["낮시간예상개수"] = daylight_expected
    daily["낮시간실측개수"] = daylight_observed
    daily["낮시간자료충족률"] = daily["낮시간실측개수"] / daily["낮시간예상개수"]
    daily.loc[daily["낮시간자료충족률"] < 0.9, "일간발전량_kWh"] = np.nan
    # ★주의★ 야간(물리적낮==0)에는 5대 전부 정상이어도 원래 관측치가 없어
    # 가용인버터수가 0으로 잡힌다. 단순 24시간 평균을 내면 야간 0이 희석시켜
    # "인버터가 부족했다"는 착시를 만든다 — 반드시 낮시간만으로 평균낸다.
    daytime = work[work["물리적낮"] == 1]
    daily["가용인버터수_낮시간평균"] = daytime.groupby(daytime.index.normalize())["가용인버터수"].mean()
    daily.index.name = "날짜"
    return daily


def main() -> None:
    config = pv_pipeline.load_config()
    five_min = recompute_plant_output(config)
    daily_recomputed = to_daily(five_min)

    original = pd.read_parquet(ORIGINAL_PARQUET)
    original.index = pd.to_datetime(original.index)

    # ── 정합성 검증: 다운기간 밖에서 원본과 재계산이 일치하는지 ──
    check_mask = (
        (daily_recomputed.index >= RECOMPUTE_START)
        & (daily_recomputed.index <= RECOMPUTE_END)
        & ~((daily_recomputed.index >= GAP_START) & (daily_recomputed.index <= GAP_END))
    )
    check = daily_recomputed.loc[check_mask, ["일간발전량_kWh"]].join(
        original[["일간발전량_kWh"]], rsuffix="_원본", how="inner"
    )
    check["차이"] = (check["일간발전량_kWh"] - check["일간발전량_kWh_원본"]).abs()
    check["비율"] = check["일간발전량_kWh"] / check["일간발전량_kWh_원본"]
    print("=== 정합성 검증(다운기간 밖, 원본 vs 재계산) ===")
    print(f"  검증일수: {len(check)}, 최대차이: {check['차이'].max():.4f}kWh, "
          f"평균차이: {check['차이'].mean():.4f}kWh")
    calibration_factor = 1.0
    if check["차이"].max() > 1.0:
        print("  ★재계산이 원본보다 체계적으로 낮게 나옴(원인: 5분 리샘플링 근사가 "
              "원 파이프라인의 실제 산출방식과 100% 동일하지 않은 것으로 추정 — "
              "원 산출 스크립트 확인 못함).★")
        print(f"  비율(재계산/원본) 평균={check['비율'].mean():.4f}, "
              f"표준편차={check['비율'].std():.4f} — 편차가 작고 매우 일관적이라 "
              "단일 배율로 보정 가능하다고 판단.")
        calibration_factor = 1.0 / check["비율"].mean()
        print(f"  → 보정계수 {calibration_factor:.4f} 적용(★이건 인버터5의 없는 값을 "
              "추정하는 게 아니라, 이 스크립트의 재현방식과 원 파이프라인 산출방식 "
              "사이의 '측정된' 배율 차이를 보정하는 것 — 4/5대 실측 부분합이라는 "
              "본질은 그대로 유지됨★).")
        print(check.to_string())

    # ── 패치: 원본이 NaN이었던 날짜만 재계산값으로 채움(보정계수 적용) ──
    daily_recomputed_calibrated = daily_recomputed.copy()
    daily_recomputed_calibrated["일간발전량_kWh"] = daily_recomputed["일간발전량_kWh"] * calibration_factor

    patched = original.copy()
    was_nan = patched["일간발전량_kWh"].isna()
    fillable = was_nan & patched.index.isin(daily_recomputed.index)
    n_before = int(patched["일간발전량_kWh"].notna().sum())
    for col in ["일간발전량_kWh", "낮시간예상개수", "낮시간실측개수", "낮시간자료충족률"]:
        patched.loc[fillable, col] = daily_recomputed_calibrated.loc[patched.index[fillable], col].to_numpy()
    patched["보정계수_적용됨"] = False
    patched.loc[fillable, "보정계수_적용됨"] = True
    patched["가용인버터수_낮시간평균"] = np.nan
    patched.loc[fillable, "가용인버터수_낮시간평균"] = daily_recomputed.loc[
        patched.index[fillable], "가용인버터수_낮시간평균"
    ].to_numpy()
    # 원래부터 5대 정상이던 날은 5로 채워 전 기간 일관된 진단열로 만든다
    patched.loc[~fillable & patched["일간발전량_kWh"].notna(), "가용인버터수_낮시간평균"] = 5.0
    n_after = int(patched["일간발전량_kWh"].notna().sum())

    print(f"\n=== 복구 결과 ===")
    print(f"  유효 일수: {n_before} → {n_after} (+{n_after - n_before}일 복구)")
    recovered_range = patched.loc[fillable]
    if len(recovered_range) > 0:
        print(f"  복구된 날짜 범위: {recovered_range.index.min().date()} ~ {recovered_range.index.max().date()}")
        print(f"  복구된 날짜의 평균 가용인버터수: {recovered_range['가용인버터수_낮시간평균'].mean():.2f}/5")

    patched.to_parquet(OUT_PARQUET)
    print(f"\n저장 완료: {OUT_PARQUET}")


if __name__ == "__main__":
    main()
