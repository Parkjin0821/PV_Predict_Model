# -*- coding: utf-8 -*-
"""영광 김삿갓1 인버터 13대 과거 Excel 시간정렬 전처리(09-01).

김제(prepare_gimje_time_alignment_v1_2026-08-31.py) 템플릿을 그대로 재사용
- 헤더가 정상 파싱되는 구조(부안처럼 위치기반 읽기 아님)라 김제 패턴이
  그대로 맞는다. 처음부터 공통 동료대조 모듈(peer_comparison_quality_v1_
  2026-09-01.py)을 사용 - 부안·김제처럼 나중에 패치하지 않고 시작부터 반영.

## 영광이 부안·김제와 다른 점(영광_컬럼구조_실측확인_v1_2026-08-31.json 근거)
- 30열(김제 22열보다 +8: 출력전압R/S/T(최대)·주파수(최대)/(최소)·
  누설전류·절연저항·출력전력(최대)).
- 13대, 용량 불균일(12대 50kW + 1대 34kW 유력후보 - Blockdata 인버터별
  내역 없어 명판·정식조회 전까지 미확정, capacity_verified=False 유지).
- 온도 컬럼이 13대 전부 상시 0(신호 없음) - 사용 불가로 그대로 통과.
- 주파수(최소)에 655.35 센티널(uint16 오버플로 추정) - 김제 온도센티널과
  같은 유형으로 NaN 처리.
- 22:00~03:59에는 원본 로그 자체가 없음(부안·김제는 24시간 로깅) - grid
  reindex가 자연히 결측으로 잡고, 집계단계 태양고도 야간0채움이 처리.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PEER_MODULE_PATH = Path(__file__).resolve().parents[1] / "공통" / "peer_comparison_quality_v1_2026-09-01.py"
_spec = importlib.util.spec_from_file_location("ucube_peer_comparison_quality", _PEER_MODULE_PATH)
peer_quality = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = peer_quality
_spec.loader.exec_module(peer_quality)


PLANT_ID = 7912
EXPECTED_INVERTERS = tuple(range(1, 14))
# ★13번=34kW는 유력후보일 뿐 미확정★(관측 최대출력 33.8kW 근거, Blockdata
# 인버터별 내역 없음) - 명판·정식조회 전까지 이 값을 확정 취급하지 않는다.
INVERTER_CAPACITY_KW = {**{i: 50.0 for i in range(1, 13)}, 13: 34.0}
INVERTER_CAPACITY_VERIFIED = False
PLANT_REGISTERED_CAPACITY_AC_CANDIDATE_KW = 634.0  # 50*12+34*1, 인버터AC 추정
PLANT_REGISTERED_CAPACITY_MODULE_DC_KW = 639.94  # Blockdata 확정(08-31), 모듈 합산 DC
KST_LABEL = "Asia/Seoul"
TEMP_SENTINELS = ()  # 영광은 온도 자체가 상시 0(신호없음) - 별도 센티널 없음
FREQ_MIN_SENTINEL = 655.35  # 65535/100 uint16 오버플로 추정(실측 08-31)
PLANT_LAT = 35.323668890815696
PLANT_LON = 126.486825627353

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\배만수 부장님 엑셀 파일"
    r"\태양광발전소 4개소 과거 데이터\영광 김삿갓1"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광"
    r"\시간정렬_v1_2026-09-01"
)

COL_NAMES = {
    "measurement_time_kst": "생성일", "inverter_number": "번호", "comm_error": "통신에러",
    "dc_voltage_v": "입력전압", "dc_current_a": "입력전류", "dc_power_kw": "입력전력",
    "ac_voltage_r_v": "출력전압R", "ac_voltage_s_v": "출력전압S", "ac_voltage_t_v": "출력전압T",
    "ac_voltage_r_max_v": "출력전압R(최대)", "ac_voltage_s_max_v": "출력전압S(최대)", "ac_voltage_t_max_v": "출력전압T(최대)",
    "ac_current_r_a": "출력전류R", "ac_current_s_a": "출력전류S", "ac_current_t_a": "출력전류T",
    "ac_power_kw": "출력전력", "ac_power_max_kw": "출력전력(최대)",
    "frequency_hz": "주파수", "frequency_max_hz": "주파수(최대)", "frequency_min_hz": "주파수(최소)",
    "power_factor_pct": "역률", "temperature_c": "온도",
    "leakage_current": "누설전류", "insulation_resistance": "절연저항",
    "daily_value": "일일", "cumulative_kwh": "누적",
    "calc_method": "계산방식", "status": "상태", "status_code": "상태값",
    "period_generation_kwh": "주기별발전량",
}
NUMERIC_COLUMNS = tuple(
    k for k in COL_NAMES
    if k not in {"measurement_time_kst", "inverter_number", "calc_method", "status", "status_code", "comm_error"}
)
EXPECTED_COL_COUNT = 30
COMM_ERROR_VALUE = "발생"  # 원본 값 그대로(숫자 아님) - 수치변환 금지, 문자열로만 비교


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="영광 김삿갓1 13대 인버터 과거자료 5분 시간정렬")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--build", action="store_true",
                    help="실제 시간정렬본(parquet)을 확정 저장한다. 없으면 감사(audit-only)만 수행.")
    p.add_argument("--interpolate-max-slots", type=int, default=2,
                    help="내부공백 최대 몇 슬롯(5분단위)까지 시간선형보간할지. 0=보간 끔. --build일 때만 적용.")
    p.add_argument("--overwrite", action="store_true",
                    help="기존 최종 산출물이 있어도 덮어쓴다. 없으면 기존 파일 존재 시 중단.")
    p.add_argument("--max-rows", type=int, default=None,
                    help="인버터 파일당 앞 N행만 읽어 소규모 실행검사(스모크테스트)를 한다.")
    return p.parse_args()


def load_one_excel(path: Path, expected_number: int, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_excel(path, nrows=max_rows)
    if df.shape[1] != EXPECTED_COL_COUNT:
        raise RuntimeError(f"{EXPECTED_COL_COUNT}열 원본 구조가 아님(실제 {df.shape[1]}열): {path}")
    missing_headers = [v for v in COL_NAMES.values() if v not in df.columns]
    if missing_headers:
        raise RuntimeError(f"예상 헤더 누락 {missing_headers}: {path}")

    df = df.rename(columns={v: k for k, v in COL_NAMES.items()})
    df["source_file"] = path.name
    df["measurement_time_kst"] = pd.to_datetime(df["measurement_time_kst"], errors="coerce")
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 부안·김제와 동일 원칙: comm_error는 문자열로만 비교(수치변환 금지).
    df["comm_error_flag"] = df["comm_error"].astype(str).str.strip() == COMM_ERROR_VALUE

    if df["measurement_time_kst"].isna().any():
        bad = int(df["measurement_time_kst"].isna().sum())
        raise RuntimeError(f"측정시각 파싱 실패 {bad}행: {path}")

    found = sorted(df["inverter_number"].dropna().astype(int).unique().tolist())
    if found != [expected_number]:
        raise RuntimeError(f"파일명 인버터 {expected_number}번과 내부 번호 불일치: {found} / {path}")
    df["inverter_number"] = expected_number
    return df.sort_values("measurement_time_kst").reset_index(drop=True)


def apply_frequency_min_sentinel(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """주파수(최소) 655.35 센티널 -> NaN(임의보간 없음, 김제 온도센티널과 동일 원칙)."""
    mask = np.isclose(df["frequency_min_hz"], FREQ_MIN_SENTINEL, atol=1e-6)
    n = int(mask.sum())
    df = df.copy()
    df.loc[mask, "frequency_min_hz"] = np.nan
    return df, n


def detect_meter_resets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("measurement_time_kst").copy()
    d["cumulative_diff"] = d["cumulative_kwh"].diff()
    resets = d[d["cumulative_diff"] < 0][
        ["inverter_number", "measurement_time_kst", "cumulative_kwh", "cumulative_diff",
         "period_generation_kwh", "source_file"]
    ]
    return resets


def grid_assign(raw: pd.DataFrame, number: int, freq_sentinel_count: int) -> tuple[pd.DataFrame, dict]:
    """감사·빌드 공통 1단계: 센티널 처리 + 5분격자 배정 + 중복해소.
    보간은 여기서 하지 않는다(그건 build 단계 전용)."""
    x = raw.copy()
    x["grid_time_kst"] = x["measurement_time_kst"].dt.round("5min")
    x["alignment_offset_sec"] = (x["measurement_time_kst"] - x["grid_time_kst"]).dt.total_seconds()
    x["alignment_abs_sec"] = x["alignment_offset_sec"].abs()

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
    aligned["inverter_capacity_kw"] = INVERTER_CAPACITY_KW[number]
    aligned["inverter_capacity_verified"] = INVERTER_CAPACITY_VERIFIED
    aligned["was_observed"] = aligned["measurement_time_kst"].notna()

    gap_mask = ~aligned["was_observed"]
    gap_run_id = (aligned["was_observed"]).cumsum()
    gap_sizes = aligned[gap_mask].groupby(gap_run_id[gap_mask]).size() if gap_mask.any() else pd.Series(dtype=int)
    gaps_le_10min = int((gap_sizes <= 2).sum())
    gaps_gt_10min = int((gap_sizes > 2).sum())

    audit = {
        "inverter_number": number,
        "capacity_kw": INVERTER_CAPACITY_KW[number],
        "source_rows": int(len(raw)),
        "start": str(raw["measurement_time_kst"].min()),
        "end": str(raw["measurement_time_kst"].max()),
        "frequency_min_sentinel_rows": freq_sentinel_count,
        "duplicates_removed": duplicates_removed,
        "aligned_rows": int(len(aligned)),
        "observed_rows": int(aligned["was_observed"].sum()),
        "missing_slots_total": int(gap_mask.sum()),
        "gap_runs_le_10min(<=2슬롯)": gaps_le_10min,
        "gap_runs_gt_10min(>2슬롯,보간대상아님)": gaps_gt_10min,
        "alignment_abs_sec_p50": float(x["alignment_abs_sec"].quantile(.50)),
        "alignment_abs_sec_p95": float(x["alignment_abs_sec"].quantile(.95)),
        "alignment_abs_sec_max": float(x["alignment_abs_sec"].max()),
        "ac_power_max_kw": float(raw["ac_power_kw"].max()),
        "ac_power_over_capacity_rows": int((raw["ac_power_kw"] > INVERTER_CAPACITY_KW[number]).sum()),
        "comm_error_flag_rows": int(raw["comm_error_flag"].sum()),
    }
    return aligned.reset_index(), audit


def apply_interpolation(aligned: pd.DataFrame, max_slots: int) -> pd.DataFrame:
    aligned = aligned.set_index("grid_time_kst")
    interp_targets = ["dc_power_kw", "ac_power_kw"]
    if max_slots <= 0:
        aligned["was_interpolated"] = False
    else:
        before = aligned[interp_targets].isna()
        interpolated = aligned[interp_targets].interpolate(
            method="time", limit=max_slots, limit_area="inside")
        after = interpolated.isna()
        for c in interp_targets:
            aligned[c] = interpolated[c]
        aligned["was_interpolated"] = (before & ~after).any(axis=1)
    aligned["long_or_edge_gap"] = aligned["ac_power_kw"].isna()
    aligned["quality_status"] = np.select(
        [aligned["was_observed"], aligned["was_interpolated"]],
        ["observed", f"interpolated_le{max_slots*5}min"],
        default="missing_gt_interp_window_or_edge",
    )
    return aligned.reset_index()


def build_plant(aligned_all: pd.DataFrame) -> pd.DataFrame:
    ac = aligned_all.pivot(index="grid_time_kst", columns="inverter_number", values="ac_power_kw")
    dc = aligned_all.pivot(index="grid_time_kst", columns="inverter_number", values="dc_power_kw")
    observed = aligned_all.pivot(index="grid_time_kst", columns="inverter_number", values="was_observed")
    interpolated = aligned_all.pivot(index="grid_time_kst", columns="inverter_number", values="was_interpolated")
    for n in EXPECTED_INVERTERS:
        if n not in ac.columns:
            raise RuntimeError(f"인버터 {n}번 정렬자료 누락")

    n_inv = len(EXPECTED_INVERTERS)
    out = pd.DataFrame(index=ac.index)
    out["plant_id"] = PLANT_ID
    out["plant_registered_capacity_ac_candidate_kw"] = PLANT_REGISTERED_CAPACITY_AC_CANDIDATE_KW
    out["plant_registered_capacity_module_dc_kw"] = PLANT_REGISTERED_CAPACITY_MODULE_DC_KW
    out["inverter_capacity_sum_kw"] = sum(INVERTER_CAPACITY_KW.values())
    out["available_inverter_count"] = ac.notna().sum(axis=1).astype("int8")
    out["observed_inverter_count"] = observed.fillna(False).sum(axis=1).astype("int8")
    out["interpolated_inverter_count"] = interpolated.fillna(False).sum(axis=1).astype("int8")
    out[f"complete_{n_inv}_inverters"] = out["available_inverter_count"].eq(n_inv)
    out["plant_ac_power_kw"] = ac.sum(axis=1, min_count=n_inv)
    out["plant_dc_power_kw"] = dc.sum(axis=1, min_count=n_inv)
    out["partial_ac_sum_reference_kw"] = ac.sum(axis=1, min_count=1)
    out["quality_status"] = np.select(
        [
            out[f"complete_{n_inv}_inverters"] & out["interpolated_inverter_count"].eq(0),
            out[f"complete_{n_inv}_inverters"],
        ],
        ["complete_observed", "complete_with_short_interpolation"],
        default="incomplete_no_official_target",
    )
    return out.reset_index()


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def check_overwrite_guard(output_dir: Path, filenames: list[str], overwrite: bool) -> None:
    existing = [f for f in filenames if (output_dir / f).exists()]
    if existing and not overwrite:
        raise RuntimeError(
            f"기존 산출물이 이미 있음(덮어쓰기 방지): {existing} - "
            f"의도한 거라면 --overwrite를 명시할 것."
        )


def main() -> None:
    args = parse_args()
    files = sorted(p for p in args.input_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if len(files) != len(EXPECTED_INVERTERS):
        raise RuntimeError(f"영광 Excel은 정확히 {len(EXPECTED_INVERTERS)}개여야 함: 발견 {len(files)}개")

    n_inv = len(EXPECTED_INVERTERS)
    mode = "build" if args.build else "audit_only"
    build_filenames = ["영광_인버터별_5분정렬.parquet", "영광_발전소_5분정렬.parquet"]
    audit_filenames = ["영광_시간정렬_감사_인버터별.csv", "영광_계량기리셋_감사.csv",
                       "영광_시간정렬_규칙및요약.json"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_overwrite_guard(
        args.output_dir, build_filenames if args.build else audit_filenames, args.overwrite
    )

    aligned_parts, audits, all_resets = [], [], []
    for number in EXPECTED_INVERTERS:
        matches = [p for p in files if f"_{number}번 " in p.name]
        if len(matches) != 1:
            raise RuntimeError(f"인버터 {number}번 파일 식별 실패: {matches}")
        raw = load_one_excel(matches[0], number, args.max_rows)
        raw, freq_sentinel_count = apply_frequency_min_sentinel(raw)
        all_resets.append(detect_meter_resets(raw))
        aligned, audit = grid_assign(raw, number, freq_sentinel_count)
        aligned_parts.append(aligned)
        audits.append(audit)
        print(f"[{mode}] 인버터 {number}: 원본={len(raw):,} 정렬={len(aligned):,} "
              f"주파수최소센티널={freq_sentinel_count:,} 통신플래그={audit['comm_error_flag_rows']:,} "
              f"10분초과공백런={audit['gap_runs_gt_10min(>2슬롯,보간대상아님)']:,}")

    all_aligned = pd.concat(aligned_parts, ignore_index=True).sort_values(["grid_time_kst", "inverter_number"])
    resets_df = pd.concat(all_resets, ignore_index=True) if all_resets else pd.DataFrame()

    audit_dir = args.output_dir
    atomic_write_csv(pd.DataFrame(audits), audit_dir / "영광_시간정렬_감사_인버터별.csv")
    atomic_write_csv(resets_df, audit_dir / "영광_계량기리셋_감사.csv")

    summary_base = {
        "mode": mode,
        "executed_at": pd.Timestamp.now(tz=KST_LABEL).isoformat(),
        "interpolate_max_slots_requested": args.interpolate_max_slots,
        "max_rows_per_file": args.max_rows,
        "plant": {
            "plant_id": PLANT_ID, "inverter_count": n_inv,
            "inverter_capacity_kw_by_number": INVERTER_CAPACITY_KW,
            "inverter_capacity_verified": INVERTER_CAPACITY_VERIFIED,
            "inverter_capacity_source": "12대 50kW·1대(13번 유력후보) 34kW 관측치기반 추정, 명판·Blockdata 인버터별 내역 미확인",
            "plant_registered_capacity_ac_candidate_kw": PLANT_REGISTERED_CAPACITY_AC_CANDIDATE_KW,
            "plant_registered_capacity_module_dc_kw": PLANT_REGISTERED_CAPACITY_MODULE_DC_KW,
            "plant_capacity_source": "Blockdata plant info API 확정(08-31): 640kW 모듈 합산 DC, AC는 인버터 추정치일 뿐",
        },
        "rows": {
            "frequency_min_sentinel_total_rows": sum(a["frequency_min_sentinel_rows"] for a in audits),
            "meter_reset_events_total": int(len(resets_df)),
        },
        "rules": {
            "grid_assignment": "nearest_5min_round",
            "duplicate_selection": "minimum_absolute_offset_then_latest_measurement",
            "temperature": "상시 0(신호없음) - 센티널 없음, 사용불가 컬럼으로 그대로 통과",
            "frequency_min_sentinel": f"{FREQ_MIN_SENTINEL} -> NaN(임의보간 없음)",
            "meter_reset": "누적발전량 감소 구간은 감사 CSV로 별도 보존, 타깃(plant_ac_power_kw)은 출력전력 합산이라 이 문제와 무관",
            "completeness_basis": "통신에러 열이 아니라 실제 관측 타임스탬프 존재 여부",
            "night_zero": "not_applied_in_this_stage(집계단계에서 태양고도로 적용)",
            "official_plant_target": f"sum_only_when_all_{n_inv}_available",
            "partial_scaling": "forbidden",
            "peer_comparison": "공통_품질정책_v1_2026-08-31.json - build 단계에서 comm_error_flag=True인 observed 행을 동료대조로 재분류",
        },
        "limitations": [
            "이 단계는 ASOS/NWP/GRID를 결합하지 않음",
            "태양고도 기반 야간 0은 후속 단계에서만 적용(build_yeonggwang_time_aggregates에서)",
            "인버터별 정격용량(12x50kW+1x34kW)은 관측치 기반 추정 - 명판·Blockdata 인버터별 조회 전까지 미확정",
            "발전소 registered capacity(AC 634kW 후보 / 모듈DC 639.94kW 확정)는 서로 다른 기준값 - 혼동 금지",
            "22:00~03:59 구간은 원본 로그 자체가 없음(부안·김제와 다른 구조) - 결측 아니라 이 사이트 정상패턴으로 추정",
        ],
    }

    if not args.build:
        summary_base["note"] = ("--audit-only 모드 - 보간·최종 발전소 총출력 parquet은 만들지 않았음. "
                                "감사 CSV·JSON을 확인한 뒤 --build --interpolate-max-slots N 으로 재실행할 것.")
        atomic_write_json(summary_base, audit_dir / "영광_시간정렬_규칙및요약.json")
        print(json.dumps(summary_base, ensure_ascii=False, indent=2))
        return

    # --build: 보간 적용 + 동료대조 재분류 + 최종 산출물 확정.
    built_parts = [apply_interpolation(a, args.interpolate_max_slots) for a in aligned_parts]
    all_built = pd.concat(built_parts, ignore_index=True).sort_values(["grid_time_kst", "inverter_number"])

    peer_result = peer_quality.classify_with_peer_comparison(
        all_built, lat=PLANT_LAT, lon=PLANT_LON,
    )
    all_built["quality_status"] = peer_result.quality_status
    print(f"[동료대조] 재검토 대상(observed+comm_error_flag) {peer_result.reclassified_count:,}행 "
          f"-> {peer_result.bucket_counts}")

    plant = build_plant(all_built)

    expected_complete = all_built.pivot(
        index="grid_time_kst", columns="inverter_number", values="ac_power_kw"
    ).sum(axis=1, min_count=n_inv).dropna()
    if not np.allclose(
        plant.loc[plant[f"complete_{n_inv}_inverters"], "plant_ac_power_kw"].to_numpy(),
        expected_complete.to_numpy(), rtol=0, atol=1e-9,
    ):
        raise RuntimeError(f"인버터 {n_inv}대 합계 검증 실패(저장 전)")

    inv_path = args.output_dir / "영광_인버터별_5분정렬.parquet"
    plant_path = args.output_dir / "영광_발전소_5분정렬.parquet"
    atomic_write_parquet(all_built, inv_path)
    atomic_write_parquet(plant, plant_path)

    # ★왕복검증★ 방금 쓴 parquet을 다시 읽어 완전가용 행수가 메모리 계산치와 같은지 확인.
    reread_plant = pd.read_parquet(plant_path)
    mem_complete = int(plant[f"complete_{n_inv}_inverters"].sum())
    disk_complete = int(reread_plant[f"complete_{n_inv}_inverters"].sum())
    if mem_complete != disk_complete:
        raise RuntimeError(
            f"왕복검증 실패: 메모리상 완전가용행수({mem_complete}) != 저장후 재읽기 행수({disk_complete})"
        )

    summary_base["peer_comparison_reclassification"] = {
        "reclassified_candidate_rows(observed+comm_error_flag)": peer_result.reclassified_count,
        "bucket_counts": peer_result.bucket_counts,
        "method": "공통_품질정책_v1_2026-08-31.json",
    }
    summary_base["limitations"].append(
        "영광_발전소_5분정렬.parquet의 available_inverter_count/plant_ac_power_kw는 "
        "quality_status로 게이팅하지 않은 참고용 롤업 - 공식 게이트는 "
        "후속 집계 스크립트가 인버터별 quality_status를 읽어 별도 적용할 것(부안·김제와 동일 구조)."
    )
    summary_base["note"] = "--build 모드 - 최종 parquet 확정 저장 완료, 왕복검증 통과."
    summary_base["rows"]["inverter_aligned"] = int(len(all_built))
    summary_base["rows"]["plant_5min"] = int(len(plant))
    summary_base["rows"][f"complete_{n_inv}"] = mem_complete
    summary_base["rows"][f"complete_{n_inv}_pct"] = float(100 * plant[f"complete_{n_inv}_inverters"].mean())
    summary_base["rows"]["roundtrip_verified"] = True
    summary_base["period"] = {"start": str(plant["grid_time_kst"].min()), "end": str(plant["grid_time_kst"].max())}

    atomic_write_json(summary_base, audit_dir / "영광_시간정렬_규칙및요약.json")
    print(json.dumps(summary_base, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
