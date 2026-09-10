# -*- coding: utf-8 -*-
"""영광 위성(GK2A VI006/IR105) - 실측 발전량 예비 상관분석(09-07).

## 배경(★사용자 확정: "위성 결합까지만 해가지고 진행을 해보는 느낌으로"★)
방향①(ramp)·방향③(regime분리) 둘 다 확정적 개선 없어서 v1(단일모델)을
현재 공식 잠정모델로 유지하기로 하고, 남은 방향②(위성 결합)를 실제로
착수해본다. 단, 09-07 확인 결과 **실시간 10분 간격 위성 수집 자체가
이미 중단된 상태**(org키 방식은 사용기간 09-04 만료로 자동중단, 공유
쿼터 방식은 09-03 16:35 이후 로그가 아예 끊김 - 프로세스가 죽어있음).
그래서 "특성으로 바로 학습"은 아직 불가능하고, 지금 가진 3시간 간격
과거아카이브(`sat_features_4regions_continuous.csv`, 2024-08-25~
현재, 5,560행)로 **먼저 상관성만 확인**한다 - Gimje 인근일사량 때도
"상관 확인 → 유의미하면 그 다음에 모델화" 순서를 지켰던 것과 동일한
원칙([[pv-track-sources-and-rationale]]).

VI006(가시광 반사도, 값이 클수록 구름 두껍고 밝음)·IR105(적외선
밝기온도, 값이 작을수록 구름꼭대기가 높고 차가움 - 두꺼운 구름)를
영광 실제 발전량(같은 3시간 슬롯, 태양고도>0인 낮 시간만)과 대조한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SAT_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
    r"\01_데이터수집\위성\sat_features_4regions_continuous.csv"
)
PLANT_5MIN = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\시간집계_v1_2026-09-01"
    r"\영광_발전소_5분_공식후보.parquet"
)
OUT_DIR = HERE / "outputs" / "영광_위성상관_예비검증_2026-09-07"
CAPACITY_KW = 634.0
TOLERANCE = pd.Timedelta("15min")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sat = pd.read_csv(SAT_CSV)
    sat["datetime_kst"] = pd.to_datetime(sat["datetime_kst"])
    sat = sat[["datetime_kst", "Yeonggwang_vi006", "Yeonggwang_ir105"]].sort_values("datetime_kst")
    sat_meta = {
        "위성_행수": int(len(sat)),
        "위성_기간": [str(sat["datetime_kst"].min()), str(sat["datetime_kst"].max())],
        "위성_간격": "3시간(과거 아카이브 - 실시간 10분 수집 아님, 아래 상태 참고)",
    }

    plant = pd.read_parquet(PLANT_5MIN, columns=[
        "grid_time_kst", "plant_ac_power_kw", "solar_elevation_deg", "quality_status"])
    plant["grid_time_kst"] = pd.to_datetime(plant["grid_time_kst"])
    plant = plant.sort_values("grid_time_kst")
    valid_q = {"complete_observed", "complete_with_short_interpolation",
              "complete_with_night_zero", "complete_with_idle_zero"}
    plant.loc[~plant["quality_status"].isin(valid_q), "plant_ac_power_kw"] = np.nan

    merged = pd.merge_asof(sat, plant, left_on="datetime_kst", right_on="grid_time_kst",
                           direction="nearest", tolerance=TOLERANCE)
    merged = merged.dropna(subset=["plant_ac_power_kw", "Yeonggwang_vi006", "Yeonggwang_ir105"])
    daylight = merged[merged["solar_elevation_deg"] > 0].copy()

    corr_vi006 = float(daylight["Yeonggwang_vi006"].corr(daylight["plant_ac_power_kw"]))
    corr_ir105 = float(daylight["Yeonggwang_ir105"].corr(daylight["plant_ac_power_kw"]))

    # 청천지수(kt) 개념으로 태양고도 효과 어느정도 통제 - 같은 고도대만 비교
    daylight["elev_bin"] = pd.cut(daylight["solar_elevation_deg"], bins=[0, 20, 40, 60, 90])
    partial_corr = {}
    for b, grp in daylight.groupby("elev_bin", observed=True):
        if len(grp) >= 30:
            partial_corr[str(b)] = {
                "n": int(len(grp)),
                "vi006_상관": round(float(grp["Yeonggwang_vi006"].corr(grp["plant_ac_power_kw"])), 3),
                "ir105_상관": round(float(grp["Yeonggwang_ir105"].corr(grp["plant_ac_power_kw"])), 3),
            }

    result = {
        **sat_meta,
        "수집상태_★중요★": {
            "실시간_10분_수집": "중단됨(09-04부로) - org키(apihub-org, 2TB/일)는 사용기간(~09-04) "
                              "만료로 자동중단, 공유쿼터 버전(collect_gk2a_continuous_10min_v1)도 "
                              "09-03 16:35 이후 로그가 끊겨 있음(프로세스 미실행 확인, 09-07 실측)",
            "raw_nc_최신파일": "gk2a_ami_le1b_vi006_ko_202609041440.nc(09-04 14:40 KST가 마지막)",
            "결론": "지금 이 상관분석은 '중단 전까지 쌓인 3시간 간격 과거아카이브'로 한 예비검증 - "
                  "실제 초단기(+1~4h) 모델 특성으로 쓰려면 고빈도(10분) 수집 재개가 선행되어야 함",
        },
        "매칭_결과행수(낮시간만)": int(len(daylight)),
        "전체_상관(태양고도>0)": {
            "VI006_vs_실제발전kW": round(corr_vi006, 3),
            "IR105_vs_실제발전kW": round(corr_ir105, 3),
        },
        "태양고도구간별_상관(고도효과 일부 통제)": partial_corr,
        "capacity_kw": CAPACITY_KW,
        "_방법론출처": "Ha 외(2024, GEO DATA 6(4)) GK2A+ASOS 결합 방법론의 '위성 구름신호가 태양광"
                     "산출과 관련있다'는 전제를 영광 실측으로 예비 검증(모델화 이전 상관확인 단계).",
        "_판정": "예비 상관분석 - promote_to_official 대상 아님. 모델 특성 채택 여부는 "
                "실시간 수집 재개 후 walk-forward 검증으로 별도 결정.",
    }
    (OUT_DIR / "영광_위성상관_예비검증_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
