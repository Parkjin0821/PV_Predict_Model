"""미래 일조시간 산출 — 직접 예보 항목이 없어 수치예보 직달복사(DSWRFLX)로부터
계산해 만드는 파생값이다 (AGENTS.md 5번 작업).

## 정의 (기상청 공식 정의를 그대로 채택)
기상청 ASOS의 일조시간은 "태양 직달일사강도가 120W/m^2 이상인 시간의 누적"으로
정의된다. 이 스크립트는 같은 임계값을, 실측이 아니라 수치예보모델의 직달복사
(DSWRFLX, KIMR)에 적용한다. 그래서 산출값은 실측 일조시간이 아니라 "예보 기반
추정 일조시간"이며 컬럼명에 `추정_`을 붙여 구분한다.

## 2026-08-20 밤 전면 재작업 이유
DSWRFLX가 "그 시각의 순간값"이 아니라 "직전 3시간 평균값"으로 보인다는 사실이
밝혀졌다(AGENTS.md "★★★신규 발견★★★" 절). 기존 버전은 3시간 값을 순간값처럼
선형보간해 임계값 교차를 계산했는데, 이는 위 발견과 어긋난 방법론이었다.
`interpolate_nwp_3h_to_1h_v1.py`(청천모양 보존적 재분배)로 새로 만든 **진짜
1시간 해상도** DSWRFLX를 입력으로 바꿔, 각 시각이 임계값(120W/m^2) 이상인지
직접 판정해 하루 합산 시간(정수 시간 단위)으로 집계한다 — 더 이상 3시간
구간 내 선형보간 교차계산이 필요 없다(이미 1시간 단위이므로).

## 결측 정책 변경 (핵심)
기존에는 "발표일 8개 시점 중 하나라도 결측이면 그 날 전체를 결측 처리"했다.
BSRN 정제 후 결측률이 직달 21%대인데, 이를 시간단위로 그대로 적용하면
낮시간(하루 약 9~14시간) 중 단 1시간만 빠져도 그 날 전체가 버려져
사실상 거의 모든 날이 탈락한다(pigeonhole). 대신 **낮시간
(태양고도>0) 대비 유효(non-NaN) 비율이 90% 이상인 날만 채택**하는
방식으로 바꿨다 — `pv_pipeline.py`의 `aggregate_power()`가 이미 쓰고 있는
"자료충족률 임계값" 관행과 동일한 원칙이다. 90% 미만인 날은 여전히
임의로 채우지 않고 결측 처리한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NWP_V2_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\nwp_hourly_interpolation_v2_2026-08-20밤"
    r"\광주_예보_1시간재구성_710일.csv"
)
OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\nwp_hourly_interpolation_v2_2026-08-20밤"
)
OUT_CSV = OUT_DIR / "광주_추정_일조시간_1시간기반_710일.csv"

THRESHOLD_W_M2 = 120.0
DAYLIGHT_COVERAGE_MIN = 0.90


def main() -> int:
    df = pd.read_csv(NWP_V2_CSV, parse_dates=["time"], low_memory=False).set_index("time")
    day = df.index.tz_localize(None).normalize() if df.index.tz is not None else df.index.normalize()
    df = df.copy()
    df["날짜"] = day
    daylight = df[df["태양고도_deg"] > 0]

    rows = []
    for date, g in daylight.groupby("날짜"):
        n_total = len(g)
        n_valid = g["DSWRFLX_bsrn정제"].notna().sum()
        coverage = n_valid / n_total if n_total else 0.0
        if coverage >= DAYLIGHT_COVERAGE_MIN:
            hours = (g["DSWRFLX_bsrn정제"].dropna() >= THRESHOLD_W_M2).sum()
        else:
            hours = None
        rows.append({
            "날짜": date, "추정_일조시간_hr": hours,
            "낮시간_유효비율": round(coverage, 3), "낮시간_전체시수": n_total,
        })

    out = pd.DataFrame(rows).sort_values("날짜")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    n_valid = out["추정_일조시간_hr"].notna().sum()
    print(f"저장 완료: {OUT_CSV} ({len(out)}행)")
    print(f"임계값: 직달복사(1시간 재구성) {THRESHOLD_W_M2}W/m^2 이상, 낮시간 유효비율 >= {DAYLIGHT_COVERAGE_MIN:.0%}인 날만 채택")
    print(f"산출 성공: {n_valid}/{len(out)}일 ({100*n_valid/len(out):.1f}%)")
    valid = out["추정_일조시간_hr"].dropna()
    if len(valid):
        print(f"평균 추정 일조시간: {valid.mean():.2f}시간, 최대: {valid.max():.2f}시간")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
