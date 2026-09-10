# -*- coding: utf-8 -*-
"""다음 단계 우선순위 ①(08-20): 오늘까지 만든 신규 파생자료를 시간단위
학습용 통합 데이터셋 하나로 합친다.

기존 상태: `환경요인_기상청_ASOS/gwangju_1hour_model_dataset_kma_observed.csv`
(2024-08-25~2026-08-04, 17,034행)에는 KMA ASOS 실측 + 장비텔레메트리는
있었지만, ① 08-18 확정 기준상 금지된 `open_meteo_forecast_*` 컬럼 20개가
그대로 남아 있었고, ② 장비 특성(주파수·온도)이 08-20에 재정의(정상 인버터만
평균)되기 전 값이었고, ③ 오늘 새로 만든 미래 NWP 일사량·운량·풍향·모듈온도·
일조시간이 전혀 병합돼 있지 않았다. 이 스크립트가 그 세 가지를 처리한다.

처리 내용
1. Open-Meteo 예보 컬럼 20개 전부 제거(공식 학습·검증에서 배제 원칙, AGENTS.md
   "2026-08-18 최신 확정 기준" 절 근거).
2. `mean_frequency_hz`·`mean_inverter_temperature_c`를 08-20 재정의 값
   (`redefine_inverter_equipment_features_v1.py` 산출물, 정상 인버터만 평균)
   으로 교체. 기존(5대 전체 평균, 오염) 값은 이 파일에는 남기지 않는다 —
   비교용 원본은 ablation 산출물 폴더에 이미 보존돼 있음(AGENTS.md 참고).
3. 다음 미래 NWP·파생 자료를 "발표일=D → 대상일 D+1의 KST 00,03,...,21시"
   관례로 시각 인덱스에 맞춰 병합한다: DSWRF·TCDC(원본), DSWRFLX·DIFSWRF는
   BSRN 정제본, LCDC·MCDC·HCDC, VEC(풍향), 추정 모듈표면온도·출력온도.
   전부 3시간 간격 8개 시점만 있으므로 그 외 시각은 NaN이 정상이다(자료
   결측이 아니라 원 자료 해상도의 자연스러운 결과 — 15분/1시간 촘촘한
   보간·전개는 별도 판단 필요, 아직 하지 않음).
4. 추정 일조시간(일 단위 1개 값)은 날짜 기준으로 그 날짜의 모든 시각에
   동일하게 채운다(하루 단일 값이므로 시각별 구분이 없음).

출력: `환경요인_기상청_ASOS/processed/gwangju_1hour_model_dataset_official_v1_2026-08-20.csv`
(기존 파일은 그대로 보존, 새 파일로 저장 — 이력 추적을 위해 덮어쓰지 않음).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS"
    r"\gwangju_1hour_model_dataset_kma_observed.csv"
)
NWP_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_nwp_solar_cloud_710d_v1_2026-08-19"
)
SOLAR_CLOUD_CSV = NWP_DIR / "광주_수치예보_DSWRF_DSWRFLX_DIFSWRF_TCDC_710일.csv"
BSRN_CSV = NWP_DIR / "광주_수치예보_DSWRFLX_DIFSWRF_BSRN정제_710일.csv"
LAYER_CSV = NWP_DIR / "광주_수치예보_LCDC_MCDC_HCDC_710일.csv"
SUNSHINE_CSV = NWP_DIR / "광주_추정_일조시간_710일.csv"
MODULE_TEMP_CSV = NWP_DIR / "광주_추정_모듈온도_710일.csv"
VEC_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\grid_forecast_vec_710d_v1_2026-08-19"
    r"\광주_격자예보_VEC_710일.csv"
)
INVERTER_REDEFINED_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\inverter_equipment_feature_redefinition_v1_2026-08-20"
    r"\광주_인버터_주파수온도_재정의_1시간_710일.csv"
)
OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
)
OUT_CSV = OUT_DIR / "gwangju_1hour_model_dataset_official_v1_2026-08-20.csv"

FCST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def wide_3hourly_to_hourly_index(path: Path, value_prefixes: list[str], suffix: str = "") -> pd.DataFrame:
    """`발표일` + `{prefix}_{hh}h{suffix}` 컬럼들을 실제 대상 시각 인덱스로 펼친다.

    suffix: 모듈온도 파일처럼 `추정_모듈표면온도_00h_C`같이 시간 뒤에 단위가
    더 붙는 경우 "_C"를 넘긴다(기본은 빈 문자열).
    """
    df = pd.read_csv(path, dtype={"발표일": str})
    records = []
    for row in df.to_dict("records"):
        issue_day = datetime.strptime(row["발표일"], "%Y%m%d")
        target_day = issue_day + timedelta(days=1)
        for hour in FCST_HOURS:
            rec = {"time": target_day + timedelta(hours=hour)}
            for prefix in value_prefixes:
                rec[prefix] = row.get(f"{prefix}_{hour:02d}h{suffix}")
            records.append(rec)
    return pd.DataFrame(records).drop_duplicates("time").set_index("time").sort_index()


def daily_value_to_hourly_index(path: Path, value_col: str, out_col: str, base_index) -> pd.Series:
    df = pd.read_csv(path, dtype={"발표일": str})
    day_map = {}
    for row in df.to_dict("records"):
        issue_day = datetime.strptime(row["발표일"], "%Y%m%d")
        target_day = (issue_day + timedelta(days=1)).date()
        day_map[target_day] = row.get(value_col)
    return pd.Series(
        [day_map.get(ts.date()) for ts in base_index], index=base_index, name=out_col
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    before_cols = len(base.columns)

    open_meteo_cols = [c for c in base.columns if c.startswith("open_meteo_forecast_")]
    base = base.drop(columns=open_meteo_cols)
    print(f"[1] Open-Meteo 예보 컬럼 {len(open_meteo_cols)}개 제거 (공식 배제 원칙)")

    inverter_redef = pd.read_csv(
        INVERTER_REDEFINED_CSV, parse_dates=["time"], low_memory=False
    ).set_index("time")
    base = base.drop(columns=["mean_frequency_hz", "mean_inverter_temperature_c"])
    base["mean_frequency_hz"] = inverter_redef["재정의_주파수평균_Hz_정상만"]
    base["mean_inverter_temperature_c"] = inverter_redef["재정의_인버터평균온도_C_정상만"]
    print("[2] mean_frequency_hz/mean_inverter_temperature_c를 08-20 재정의(정상 인버터만)로 교체")

    solar_cloud = wide_3hourly_to_hourly_index(SOLAR_CLOUD_CSV, ["DSWRF", "TCDC"])
    bsrn = wide_3hourly_to_hourly_index(BSRN_CSV, ["DSWRFLX", "DIFSWRF"]).rename(
        columns={"DSWRFLX": "DSWRFLX_bsrn정제", "DIFSWRF": "DIFSWRF_bsrn정제"}
    )
    layers = wide_3hourly_to_hourly_index(LAYER_CSV, ["LCDC", "MCDC", "HCDC"])
    vec = wide_3hourly_to_hourly_index(VEC_CSV, ["VEC"])
    module_temp = wide_3hourly_to_hourly_index(
        MODULE_TEMP_CSV, ["추정_모듈표면온도", "추정_출력온도"], suffix="_C"
    )

    nwp_all = solar_cloud.join([bsrn, layers, vec, module_temp], how="outer")
    print(f"[3] 미래 NWP 계열 {len(nwp_all.columns)}개 컬럼, {len(nwp_all)}개 시각(3시간 간격 8점/일) 준비")

    merged = base.join(nwp_all, how="left")

    sunshine = daily_value_to_hourly_index(
        SUNSHINE_CSV, "추정_일조시간_hr", "추정_일조시간_hr", merged.index
    )
    merged["추정_일조시간_hr"] = sunshine
    print("[4] 추정_일조시간_hr(일 단위 값)을 날짜 기준으로 채움")

    merged.index.name = "time"
    merged.to_csv(OUT_CSV, encoding="utf-8-sig")

    print(f"\n행수: {len(merged)} (기존 {len(base)}행과 동일해야 정상)")
    print(f"컬럼수: 기존 {before_cols} → 신규 {len(merged.columns)}")
    for c in nwp_all.columns.tolist() + ["추정_일조시간_hr"]:
        n_valid = merged[c].notna().sum()
        print(f"  {c}: 유효값 {n_valid}/{len(merged)} ({n_valid/len(merged)*100:.1f}%, 3시간 해상도 자료라 나머지는 자연 결측)")
    print(f"\n저장 완료: {OUT_CSV}")


if __name__ == "__main__":
    main()
