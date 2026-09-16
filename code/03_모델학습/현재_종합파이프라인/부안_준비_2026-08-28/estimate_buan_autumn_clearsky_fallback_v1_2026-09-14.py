# -*- coding: utf-8 -*-
"""부안 가을(9~11월) 월간 발전량 물리모델 잠정 추정 - 데이터 공백 대응.

## 배경
부안 발전소 실측 원본(엑셀)의 최초 기록이 2025-12-12라 2025년 가을은
발전소 자체가 없었고, 2026년 가을(지금, 09-14 기준 진행 중)이 부안의
"첫 가을"이다 - 아직 다 지나지 않아 완전한 표본이 없다(09-08 확정,
재문의 금지 - AGENTS.md 09-08 항목). **이 스크립트는 그 결론을 뒤집는
게 아니라, 실측이 다 쌓이기 전까지 쓸 잠정 추정치를 물리모델로 만든다.**

## 방법 (재구현 없음 - 기존 검증된 조각만 재사용)
`ultra_short_term_v1_buan_2026-09-08.py`의 NOAA 태양고도 근사식 +
Haurwitz(1945) 청천일 GHI 공식을 그대로 가져온다(영광·광주 v1과
동일 공식, 이미 이 프로젝트에서 검증·사용 중).

    가을 예측 발전량(월) = 청천출력(그 달, 순수 물리계산) × 감쇠계수

- **청천출력**: 위도·경도·설비용량만 있으면 데이터 없이 계산 가능
  (임의 데이터 아님 - 태양 기하학).
- **감쇠계수**: 부안에 실제로 있는 봄·여름·겨울 데이터로
  (실제 월간 발전량 / 같은 달 청천출력)을 계산해 평균.

## 학술 근거(국내 KCI, 09-14 사전검색)
- 정영선 외, "한반도의 일사량 추정을 위한 청천일 모델의 비교 평가"
  (KCI, ART002046112) - 5개 청천일 모델(ESRA·Dumortier·MODTRAN·
  Bourges·PdBV) 비교, **PdBV(태양고도 기반 1차원 모델)가 계절과
  무관하게 비교적 일정한 편향(MBE) 유지**를 확인. 단, 단순모델은
  봄~초여름 과소추정, 복잡모델은 겨울 과대추정 경향도 같이 보고됨
  - 완전한 계절무관은 아니라 이 스크립트는 그 불확실성을 manifest에
    명시적으로 남긴다(가을 방향성은 문헌에 없어 임의 보정 안 함).
- "우리나라 지역별 청명도 예측 모델을 이용한 월평균 수평면 일사량
  산출"(KCI, ART001440902) - 청명지수를 구름량·월·위도·경도의
  함수(3차 다항식)로 일반화 - 특정 지역·월의 실측 없이도 월평균
  일사량 산출이 국내에서 이미 쓰이는 접근임을 뒷받침.

## 이 추정치의 위상 (반드시 지킬 것)
- **운영 승격 절대 금지** - 실측이 아닌 물리모델 추정치다.
- manifest에 `status: "가을_미검증_물리모델추정"`을 명시하고, 대시보드에
  노출할 경우 반드시 이 라벨을 같이 보여준다.
- 2026년 9~11월 실측이 다 쌓이면(11월 말~12월 초, 09-08 절 예정대로)
  이 추정치를 실측 검증으로 교체한다 - 그 전까지만 쓰는 임시값이다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "부안_가을_청천모델_잠정추정_2026-09-14"

# 이번 예측 대상: 부안의 첫 가을(2026년), 아직 진행 중 - 9~11월 통째로
AUTUMN_MONTHS = [(2026, 9), (2026, 10), (2026, 11)]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없음: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clearsky_kwh_at(m: "module", index: pd.DatetimeIndex) -> float:
    """주어진 시각 목록에 대해서만 청천출력(kWh)을 물리계산으로 합산.
    실측 데이터 불필요 - 위도/경도/설비용량과 시각만 있으면 됨."""
    if len(index) == 0:
        return 0.0
    elev = m.solar_elevation_deg(index)
    ghi = m.haurwitz_clearsky_ghi_wm2(elev)
    power_kw = m.CAPACITY_KW * ghi / 1000.0
    return float(np.sum(power_kw) * (5.0 / 60.0))  # kW * h(5분=1/12시간) 합산 = kWh


def month_clearsky_kwh(m: "module", year: int, month: int) -> float:
    """해당 연/월 전체(캘린더 풀) 5분 격자 청천출력 - 가을처럼 실측이
    아예 없는 달을 통째로 추정할 때만 쓴다."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd(1)) + pd.Timedelta(days=1)
    grid = pd.date_range(start, end, freq="5min", inclusive="left")
    return clearsky_kwh_at(m, pd.DatetimeIndex(grid))


def main() -> int:
    m = load_module("buan_ultra_base", HERE / "ultra_short_term_v1_buan_2026-09-08.py")
    backtest = load_module(
        "buan_backtest_base", HERE / "ultra_short_historical_backtest_v1_2026-08-31.py")

    plant_5min = backtest.load_plant_5min_series()  # 실측(봄·여름·겨울만 존재)
    plant_hourly = plant_5min.resample("1h").mean(numeric_only=True) if hasattr(
        plant_5min, "resample") else plant_5min

    valid = plant_5min.dropna()
    real_months = sorted({(ts.year, ts.month) for ts in valid.index})
    # ★09-14 수정★: 첫 계산에서 8월(데이터 시작 04일까지만)·12월(12일부터
    # 시작)이 "부분월"인데 청천출력은 그 달 전체(풀캘린더)로 나눠버려서
    # 8월 비율이 0.097로 12월(0.38)보다도 낮게 나오는 등 물리적으로
    # 말이 안 되는 결과가 나왔었다(부분월 편향, 실제 날씨 효과 아님).
    # 청천출력도 "실측이 실제로 있는 그 시각들"에만 맞춰 계산하도록
    # 고쳐서 부분월도 공정하게 비교되게 한다.
    MIN_VALID_SLOTS = 500  # 대략 낮시간 며칠치 이상 - 너무 적은 표본 제외
    monthly_actual_kwh: dict[str, float] = {}
    monthly_clear_kwh: dict[str, float] = {}
    monthly_ratio: dict[str, float] = {}
    monthly_coverage: dict[str, dict] = {}
    for year, month in real_months:
        key = f"{year}-{month:02d}"
        month_idx = valid.index[(valid.index.year == year) & (valid.index.month == month)]
        n_slots = len(month_idx)
        days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
        coverage_pct = round(n_slots / (days_in_month * 288) * 100, 1)
        monthly_coverage[key] = {"valid_slots": n_slots, "coverage_pct": coverage_pct}
        if n_slots < MIN_VALID_SLOTS:
            continue  # 표본이 너무 적어 감쇠계수 추정에 부적합(제외 사유 monthly_coverage에 남김)
        actual_kwh = float(valid.loc[month_idx].sum() * (5.0 / 60.0))
        clear_kwh = clearsky_kwh_at(m, month_idx)  # 같은 시각들만 - 부분월도 공정 비교
        if actual_kwh <= 0 or clear_kwh <= 0:
            continue
        monthly_actual_kwh[key] = round(actual_kwh, 2)
        monthly_clear_kwh[key] = round(clear_kwh, 2)
        monthly_ratio[key] = round(actual_kwh / clear_kwh, 4)

    if not monthly_ratio:
        raise RuntimeError("사용 가능한 실측 월(봄/여름/겨울)이 하나도 없음 - 감쇠계수 계산 불가")

    ratios = np.array(list(monthly_ratio.values()))
    avg_ratio = float(np.mean(ratios))
    ratio_std = float(np.std(ratios))

    autumn_estimate: dict[str, dict] = {}
    for year, month in AUTUMN_MONTHS:
        key = f"{year}-{month:02d}"
        clear_kwh = month_clearsky_kwh(m, year, month)
        autumn_estimate[key] = {
            "clearsky_kwh": round(clear_kwh, 2),
            "estimated_actual_kwh": round(clear_kwh * avg_ratio, 2),
            "estimated_range_kwh": [
                round(clear_kwh * max(avg_ratio - ratio_std, 0.0), 2),
                round(clear_kwh * (avg_ratio + ratio_std), 2),
            ],
        }

    manifest = {
        "region": "부안", "status": "가을_미검증_물리모델추정",
        "method": "청천출력(Haurwitz 1945, NOAA 태양고도 근사) × 봄여름겨울 평균 감쇠계수",
        "warning": (
            "이 값은 실측이 아니라 물리모델 추정치다. 운영 승격 금지. "
            "2026년 9~11월 실측이 다 쌓이면(예정: 11월말~12월초) 교체할 것."
        ),
        "academic_basis_kci": [
            {"title": "한반도의 일사량 추정을 위한 청천일 모델의 비교 평가",
             "kci_id": "ART002046112",
             "note": "PdBV(태양고도기반 1차원모델)가 계절무관 비교적 일정한 MBE 유지 "
                     "- 단, 단순모델 봄초여름 과소추정·복잡모델 겨울 과대추정 경향 있음 "
                     "(가을 방향성은 문헌에 없어 별도 보정 안 함)"},
            {"title": "우리나라 지역별 청명도 예측 모델을 이용한 월평균 수평면 일사량 산출",
             "kci_id": "ART001440902",
             "note": "청명지수를 구름량·월·위도·경도 함수로 일반화 - 실측 없는 지역/월도 산출 가능함을 뒷받침"},
        ],
        "capacity_kw": m.CAPACITY_KW, "latitude": m.LATITUDE, "longitude": m.LONGITUDE,
        "reference_months_used": {
            "months": list(monthly_ratio.keys()),
            "monthly_actual_kwh": monthly_actual_kwh,
            "monthly_clearsky_kwh": monthly_clear_kwh,
            "monthly_ratio": monthly_ratio,
            "note": "청천출력은 그 달 실측이 존재하는 시각에만 맞춰 계산함(부분월 편향 방지)",
        },
        "all_months_coverage_including_excluded": monthly_coverage,
        "avg_ratio": round(avg_ratio, 4), "ratio_std": round(ratio_std, 4),
        "autumn_2026_estimate": autumn_estimate,
        "created_at_kst": datetime.now(tz=KST).isoformat(timespec="seconds"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
