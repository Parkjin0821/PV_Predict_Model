"""부안 도담 인버터 8대 과거 Excel 시간정렬 전처리.

실행 시 API를 호출하지 않는다. 원본 Excel을 읽어 다음 산출물만 만든다.
- 부안_인버터별_5분정렬.parquet
- 부안_발전소_5분정렬.parquet
- 부안_시간정렬_감사_인버터별.csv
- 부안_시간정렬_규칙및요약.json

사전동결 규칙
1) 원시 측정시각을 KST 로컬시각으로 보존한다.
2) 최근접 5분 시각(round)으로 귀속한다. 단순 floor는 사용하지 않는다.
3) 같은 인버터·5분 슬롯에 여러 행이면 기준시각과 가장 가까운 행을 쓴다.
   거리가 같으면 더 늦게 측정된 행을 쓴다.
4) 5분 격자화 후 내부공백 최대 2칸(10분)만 시간선형보간한다.
5) 10분 초과 공백, 가장자리 공백은 보간하지 않는다.
6) 8대 완전가용일 때만 공식 발전소 총출력을 만든다. 부분합 스케일업은 금지한다.
7) 야간 0은 이 단계에서 만들지 않는다. 부안 좌표 기반 태양고도를 결합하는
   후속 단계에서만 별도 품질플래그와 함께 적용한다.
8) 모든 원시·정렬·보간 여부와 정렬 오프셋을 보존한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

_PEER_MODULE_PATH = Path(__file__).resolve().parents[1] / "공통" / "peer_comparison_quality_v1_2026-09-01.py"
_spec = importlib.util.spec_from_file_location("ucube_peer_comparison_quality", _PEER_MODULE_PATH)
peer_quality = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = peer_quality
_spec.loader.exec_module(peer_quality)


PLANT_ID = 16783
EXPECTED_INVERTERS = tuple(range(1, 9))
INVERTER_CAPACITY_KW = 125.0
PLANT_AC_CAPACITY_KW = 1000.0
PLANT_DC_CAPACITY_KW = 998.715
KST_LABEL = "Asia/Seoul"
# 부안 실측좌표(build_buan_time_aggregates_v1_2026-08-28.py와 동일 출처).
PLANT_LAT = 35.7874617462152
PLANT_LON = 126.73000042548799

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\배만수 부장님 엑셀 파일"
    r"\태양광발전소 4개소 과거 데이터\부안 도담"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\시간정렬_v1_2026-08-28"
)

# Excel 컬럼은 파일별로 동일한 30열 구조다. 헤더 인코딩에 의존하지 않고
# 검증된 위치만 읽는다. 위치가 바뀌면 아래 구조검사가 즉시 중단시킨다.
COL = {
    "measurement_time_kst": 0,
    "inverter_number": 1,
    "comm_error": 2,
    "dc_voltage_v": 3,
    "dc_current_a": 4,
    "dc_power_kw": 5,
    "ac_voltage_r_v": 6,
    "ac_voltage_s_v": 7,
    "ac_voltage_t_v": 8,
    "ac_current_r_a": 12,
    "ac_current_s_a": 13,
    "ac_current_t_a": 14,
    "ac_power_kw": 15,
    "frequency_hz": 17,
    "power_factor_pct": 20,
    "temperature_c": 21,
    "daily_energy_kwh": 24,
    "total_energy_kwh": 25,
}
NUMERIC_COLUMNS = tuple(c for c in COL if c not in {"measurement_time_kst", "inverter_number", "comm_error"})
COMM_ERROR_VALUE = "발생"  # 원본 값 그대로(숫자 아님) - 수치변환 금지, 문자열로만 비교
POWER_COLUMNS = ("ac_power_kw", "dc_power_kw")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="부안 도담 8대 인버터 과거자료 5분 시간정렬")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def load_one_excel(path: Path, expected_number: int) -> pd.DataFrame:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    ws.reset_dimensions()  # 원본 파일에 worksheet dimension이 없는 경우 대응
    rows = ws.iter_rows(values_only=True)
    header = next(rows, None)
    if header is None or len(header) < 30:
        wb.close()
        raise RuntimeError(f"30열 원본 구조가 아님: {path}")

    records: list[dict] = []
    for excel_row, row in enumerate(rows, start=2):
        if row is None or len(row) <= max(COL.values()):
            continue
        rec = {name: row[pos] for name, pos in COL.items()}
        rec["source_file"] = path.name
        rec["source_excel_row"] = excel_row
        records.append(rec)
    wb.close()
    if not records:
        raise RuntimeError(f"데이터 행이 없음: {path}")

    df = pd.DataFrame.from_records(records)
    df["measurement_time_kst"] = pd.to_datetime(df["measurement_time_kst"], errors="coerce")
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # ★08-31 이전 결함 수정★: comm_error는 "발생"/공백 같은 문자열이라
    # 수치변환하면 전부 NaN이 돼서 값이 사라진다 - 문자열로만 비교해 플래그화.
    df["comm_error_flag"] = df["comm_error"].astype(str).str.strip() == COMM_ERROR_VALUE
    if df["measurement_time_kst"].isna().any():
        bad = int(df["measurement_time_kst"].isna().sum())
        raise RuntimeError(f"측정시각 파싱 실패 {bad}행: {path}")

    found = sorted(df["inverter_number"].dropna().astype(int).unique().tolist())
    if found != [expected_number]:
        raise RuntimeError(f"파일명 인버터 {expected_number}번과 내부 번호 불일치: {found} / {path}")
    df["inverter_number"] = expected_number
    return df.sort_values("measurement_time_kst").reset_index(drop=True)


def align_one(raw: pd.DataFrame, number: int) -> tuple[pd.DataFrame, dict]:
    x = raw.copy()
    x["grid_time_kst"] = x["measurement_time_kst"].dt.round("5min")
    x["alignment_offset_sec"] = (
        x["measurement_time_kst"] - x["grid_time_kst"]
    ).dt.total_seconds()
    x["alignment_abs_sec"] = x["alignment_offset_sec"].abs()

    # 가까운 행 우선, 동률이면 늦은 측정행 우선
    x = x.sort_values(
        ["grid_time_kst", "alignment_abs_sec", "measurement_time_kst"],
        ascending=[True, True, False],
    )
    duplicates_removed = int(x.duplicated("grid_time_kst", keep="first").sum())
    chosen = x.drop_duplicates("grid_time_kst", keep="first").set_index("grid_time_kst").sort_index()

    grid = pd.date_range(chosen.index.min(), chosen.index.max(), freq="5min")
    aligned = chosen.reindex(grid)
    aligned.index.name = "grid_time_kst"
    aligned["plant_id"] = PLANT_ID
    aligned["inverter_number"] = number
    aligned["inverter_capacity_kw"] = INVERTER_CAPACITY_KW
    aligned["was_observed"] = aligned["measurement_time_kst"].notna()

    before = aligned[list(NUMERIC_COLUMNS)].isna()
    interpolated = aligned[list(NUMERIC_COLUMNS)].interpolate(
        method="time", limit=2, limit_area="inside"
    )
    after = interpolated.isna()
    for c in NUMERIC_COLUMNS:
        aligned[c] = interpolated[c]
    aligned["was_interpolated"] = (before & ~after).any(axis=1)
    aligned["interpolated_feature_count"] = (before & ~after).sum(axis=1).astype("int16")
    aligned["long_or_edge_gap"] = aligned["ac_power_kw"].isna()
    aligned["quality_status"] = np.select(
        [aligned["was_observed"], aligned["was_interpolated"]],
        ["observed", "interpolated_le10min"],
        default="missing_gt10min_or_edge",
    )

    audit = {
        "inverter_number": number,
        "source_rows": int(len(raw)),
        "start": str(raw["measurement_time_kst"].min()),
        "end": str(raw["measurement_time_kst"].max()),
        "duplicates_removed": duplicates_removed,
        "aligned_rows": int(len(aligned)),
        "observed_rows": int(aligned["was_observed"].sum()),
        "interpolated_rows": int(aligned["was_interpolated"].sum()),
        "remaining_ac_power_missing_rows": int(aligned["ac_power_kw"].isna().sum()),
        "alignment_abs_sec_p50": float(x["alignment_abs_sec"].quantile(.50)),
        "alignment_abs_sec_p95": float(x["alignment_abs_sec"].quantile(.95)),
        "alignment_abs_sec_max": float(x["alignment_abs_sec"].max()),
        "ac_power_max_kw": float(raw["ac_power_kw"].max()),
        "ac_power_over_125_rows": int((raw["ac_power_kw"] > INVERTER_CAPACITY_KW).sum()),
    }
    return aligned.reset_index(), audit


def build_plant(aligned_all: pd.DataFrame) -> pd.DataFrame:
    ac = aligned_all.pivot(index="grid_time_kst", columns="inverter_number", values="ac_power_kw")
    dc = aligned_all.pivot(index="grid_time_kst", columns="inverter_number", values="dc_power_kw")
    observed = aligned_all.pivot(index="grid_time_kst", columns="inverter_number", values="was_observed")
    interpolated = aligned_all.pivot(index="grid_time_kst", columns="inverter_number", values="was_interpolated")
    for n in EXPECTED_INVERTERS:
        if n not in ac.columns:
            raise RuntimeError(f"인버터 {n}번 정렬자료 누락")

    out = pd.DataFrame(index=ac.index)
    out["plant_id"] = PLANT_ID
    out["plant_ac_capacity_kw"] = PLANT_AC_CAPACITY_KW
    out["plant_dc_capacity_kw"] = PLANT_DC_CAPACITY_KW
    out["available_inverter_count"] = ac.notna().sum(axis=1).astype("int8")
    out["observed_inverter_count"] = observed.fillna(False).sum(axis=1).astype("int8")
    out["interpolated_inverter_count"] = interpolated.fillna(False).sum(axis=1).astype("int8")
    out["complete_8_inverters"] = out["available_inverter_count"].eq(8)
    out["plant_ac_power_kw"] = ac.sum(axis=1, min_count=8)
    out["plant_dc_power_kw"] = dc.sum(axis=1, min_count=8)
    out["partial_ac_sum_reference_kw"] = ac.sum(axis=1, min_count=1)
    out["quality_status"] = np.select(
        [
            out["complete_8_inverters"] & out["interpolated_inverter_count"].eq(0),
            out["complete_8_inverters"],
        ],
        ["complete_observed", "complete_with_short_interpolation"],
        default="incomplete_no_official_target",
    )
    return out.reset_index()


def main() -> None:
    args = parse_args()
    files = sorted(p for p in args.input_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if len(files) != 8:
        raise RuntimeError(f"부안 Excel은 정확히 8개여야 함: 발견 {len(files)}개")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_parts=[]
    audits=[]
    for number in EXPECTED_INVERTERS:
        matches=[p for p in files if f"_{number}번 " in p.name]
        if len(matches) != 1:
            raise RuntimeError(f"인버터 {number}번 파일 식별 실패: {matches}")
        raw=load_one_excel(matches[0], number)
        aligned,audit=align_one(raw, number)
        aligned_parts.append(aligned)
        audits.append(audit)
        print(f"[정렬] 인버터 {number}: 원본={len(raw):,} 정렬={len(aligned):,} 보간={audit['interpolated_rows']:,}")

    all_aligned=pd.concat(aligned_parts, ignore_index=True).sort_values(["grid_time_kst","inverter_number"])

    # ★09-01 추가: 공통_품질정책_v1_2026-08-31.json 구현 - 동료 인버터 대조로
    # comm_error_flag=True인 observed 행을 재분류(영광에서 검증한 방법론).
    peer_result = peer_quality.classify_with_peer_comparison(
        all_aligned, lat=PLANT_LAT, lon=PLANT_LON,
    )
    all_aligned["quality_status"] = peer_result.quality_status
    print(f"[동료대조] 재검토 대상(observed+comm_error_flag) {peer_result.reclassified_count:,}행 "
          f"-> {peer_result.bucket_counts}")

    plant=build_plant(all_aligned)
    if not np.allclose(
        plant.loc[plant["complete_8_inverters"], "plant_ac_power_kw"].to_numpy(),
        all_aligned.pivot(index="grid_time_kst", columns="inverter_number", values="ac_power_kw")
        .sum(axis=1, min_count=8).dropna().to_numpy(),
        rtol=0, atol=1e-9,
    ):
        raise RuntimeError("인버터 8대 합계 검증 실패")

    all_aligned.to_parquet(args.output_dir / "부안_인버터별_5분정렬.parquet", index=False)
    plant.to_parquet(args.output_dir / "부안_발전소_5분정렬.parquet", index=False)
    pd.DataFrame(audits).to_csv(args.output_dir / "부안_시간정렬_감사_인버터별.csv", index=False, encoding="utf-8-sig")
    summary={
        "status":"pre_registered_time_alignment_v1",
        "executed_at":pd.Timestamp.now(tz=KST_LABEL).isoformat(),
        "rules":{
            "grid_assignment":"nearest_5min_round",
            "duplicate_selection":"minimum_absolute_offset_then_latest_measurement",
            "interpolation":"time_linear_inside_only_max_2_slots_10min",
            "night_zero":"not_applied_in_this_stage",
            "official_plant_target":"sum_only_when_all_8_available",
            "partial_scaling":"forbidden",
        },
        "plant":{
            "plant_id":PLANT_ID,"ac_capacity_kw":PLANT_AC_CAPACITY_KW,
            "dc_capacity_kw":PLANT_DC_CAPACITY_KW,"inverter_count":8,
        },
        "period":{"start":str(plant["grid_time_kst"].min()),"end":str(plant["grid_time_kst"].max())},
        "rows":{
            "inverter_aligned":int(len(all_aligned)),"plant_5min":int(len(plant)),
            "complete_8":int(plant["complete_8_inverters"].sum()),
            "complete_8_pct":float(100*plant["complete_8_inverters"].mean()),
            "complete_observed":int((plant["quality_status"]=="complete_observed").sum()),
            "complete_with_short_interpolation":int((plant["quality_status"]=="complete_with_short_interpolation").sum()),
        },
        "limitations":[
            "이 단계는 ASOS/NWP를 결합하지 않음",
            "태양고도 기반 야간 0은 후속 단계에서만 적용",
            "2026-08-05 이후 API 라이브 자료 이어붙이기는 별도 단계",
            "이 파일의 발전소 5분정렬 롤업(부안_발전소_5분정렬.parquet)은 참고용 - "
            "공식 게이트(valid_quality 필터링)는 build_buan_time_aggregates가 "
            "인버터별 parquet의 quality_status를 읽어 별도로 적용함",
        ],
        "peer_comparison_reclassification": {
            "reclassified_candidate_rows(observed+comm_error_flag)": peer_result.reclassified_count,
            "bucket_counts": peer_result.bucket_counts,
            "method": "공통_품질정책_v1_2026-08-31.json, 영광에서 검증한 임계값 재사용",
        },
    }
    (args.output_dir / "부안_시간정렬_규칙및요약.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
