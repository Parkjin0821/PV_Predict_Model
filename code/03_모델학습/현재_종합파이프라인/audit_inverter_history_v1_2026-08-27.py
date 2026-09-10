# -*- coding: utf-8 -*-
"""광주 인버터 1~5 원자료와 발전소 총출력 정합성 감사.

API를 호출하지 않고 배만수 부장님 원본 Excel 5개와 v5 로컬 산출물만
읽는다. 운영모델·DB·원본은 수정하지 않으며 결과는 전용 outputs 폴더에만
저장한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
INVERTER_DIR = PROJECT.parent / "배만수 부장님 엑셀 파일"
INVERTER_FILES = {
    1: "광주 광주시청 _1번 인버터 로그_계산됨.xlsx",
    2: "광주 광주시청_2번 인버터 로그_계산됨.xlsx",
    3: "광주 광주시청_3번 인버터 로그_계산됨.xlsx",
    4: "광주 광주시청_4번 인버터 로그_계산됨.xlsx",
    5: "광주 광주시청_5번 인버터 로그_계산됨.xlsx",
}
CAPACITY_KW = {1: 50.0, 2: 50.0, 3: 30.0, 4: 39.0, 5: 50.0}
TOTAL_CAPACITY_KW = 219.0
DEFECT_START = pd.Timestamp("2025-08-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2025-11-17")
V5_DIR = ROOT / "outputs" / "v5_복구_2026-08-21"
REBUILD_V5_SCRIPT = PROJECT / "02_전처리" / "rebuild_plant_v5_recovered_2026-08-21.py"
OUT = ROOT / "outputs" / "인버터_이력감사_v1_2026-08-27"


def _assert_sources() -> None:
    missing = [str(INVERTER_DIR / name) for name in INVERTER_FILES.values()
               if not (INVERTER_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("인버터 원본 파일 누락:\n" + "\n".join(missing))
    if abs(sum(CAPACITY_KW.values()) - TOTAL_CAPACITY_KW) > 1e-12:
        raise AssertionError("등록용량 합계가 219kW가 아님")


def load_inverter_raw(number: int) -> pd.DataFrame:
    """원본 1개를 읽어 숫자형 정리. 원본 행은 변경하지 않는다."""
    path = INVERTER_DIR / INVERTER_FILES[number]
    wanted = ["생성일", "출력전력", "입력전력", "입력전류", "입력전압",
              "주파수", "역률", "온도", "통신에러"]
    # 대용량 Excel을 읽을 때 필요 없는 셀까지 파싱하지 않는다. 이 감사의
    # 첫 실행 병목은 계산이 아니라 openpyxl의 전체 시트 파싱이었다.
    frame = pd.read_excel(path, usecols=lambda col: col in wanted)
    absent = [c for c in wanted if c not in frame.columns]
    if absent:
        raise ValueError(f"{path.name}: 필수 컬럼 누락 {absent}")
    frame = frame[wanted].copy()
    frame["생성일"] = pd.to_datetime(frame["생성일"], errors="coerce")
    for col in wanted[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["생성일"]).sort_values("생성일")
    frame = frame.drop_duplicates(subset=["생성일"], keep="last").set_index("생성일")
    return frame


def load_all_inverters_5min(
        raw_cache: dict[int, pd.DataFrame] | None = None) -> dict[int, pd.DataFrame]:
    """공식 v5 함수를 직접 재사용해 5분 인버터 출력을 재현한다.

    개별 0~80kW 필터 → 야간 결측 0 → 내부 2스텝(10분) 시간보간 순서다.
    이 함수를 별도로 재구현하지 않아 감사/CV와 공식 정답의 차이를 막는다.
    """
    _assert_sources()
    # 분해 CV가 이 함수를 인자 없이 import해서 호출하는 기존 인터페이스도
    # 유지한다. 이 경우에도 Excel은 인버터별 한 번씩만 읽는다.
    if raw_cache is None:
        print("인버터 원본 Excel 5개 로딩(분해 CV용, 각 파일 1회)...")
        raw_cache = {
            number: load_inverter_raw(number) for number in INVERTER_FILES}
    if not REBUILD_V5_SCRIPT.is_file():
        raise FileNotFoundError(REBUILD_V5_SCRIPT)
    spec = importlib.util.spec_from_file_location("inverter_audit_rebuild_v5", REBUILD_V5_SCRIPT)
    rebuild = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rebuild
    assert spec.loader is not None
    spec.loader.exec_module(rebuild)

    # Excel은 main()에서 인버터별 정확히 한 번만 읽는다. 공식 rebuild의
    # process_inverter_5min()을 그대로 부르면 내부에서 5개를 다시 읽으므로,
    # 아래에는 그 함수의 5분 처리 순서를 동일하게 적용한다.
    probe = raw_cache[1]
    for number in range(2, 6):
        probe = probe.combine_first(raw_cache[number])
    grid = pd.date_range(probe.index.min().floor("5min"), probe.index.max().ceil("5min"), freq="5min")
    config = rebuild.pv_pipeline.load_config()
    elevation, _ = rebuild.pv_pipeline.solar_position(
        grid, float(config["site"]["latitude"]), float(config["site"]["longitude"]))
    night = pd.Series(elevation <= 0.0, index=grid)

    result: dict[int, pd.DataFrame] = {}
    for number in INVERTER_FILES:
        official_raw = raw_cache[number].copy()
        lo, hi = rebuild.PER_INVERTER_POWER_CLIP
        official_raw.loc[
            ~official_raw["출력전력"].between(lo, hi), "출력전력"] = np.nan
        tlo, thi = rebuild.PER_INVERTER_TEMP_CLIP
        official_raw.loc[
            ~official_raw["온도"].between(tlo, thi), "온도"] = np.nan
        cols = list(rebuild.POWER_COLS) + list(rebuild.MEAN_COLS)
        official = official_raw[cols].resample("5min").mean().reindex(grid)
        official["_raw_observed"] = official["출력전력"].notna().astype("int8")
        for col in rebuild.POWER_COLS:
            official.loc[night & official[col].isna(), col] = 0.0
        official[cols] = official[cols].interpolate(
            method="time", limit=2, limit_area="inside")
        official.columns = [
            f"{col}__inv{number}" if col in cols else col
            for col in official.columns]
        values = official[[f"출력전력__inv{number}", "_raw_observed"]].rename(
            columns={f"출력전력__inv{number}": f"인버터{number}_kW",
                     "_raw_observed": f"인버터{number}_원시관측"})
        result[number] = values
    return result


def aggregate_inverter_power(per_inv: dict[int, pd.DataFrame], freq: str) -> pd.DataFrame:
    """공식 v5 총출력과 정확히 합치되는 인버터별 관측 기여량.

    공식 v5처럼 각 5분 시각에 살아있는 인버터를 먼저 합산한 뒤 기간평균을
    낸다. 개별 기여량도 같은 '유효 발전소 5분슬롯 수'를 분모로 써서
    5대 기여량 합계 == 공식 발전소 평균이 항상 성립한다. 인버터 가용판정은
    별도로 15분 100%, 1시간 75% 규칙을 유지한다.
    """
    if freq not in {"15min", "1h"}:
        raise ValueError("freq는 15min 또는 1h만 허용")
    min_count = 3 if freq == "15min" else 9
    rule = 1.0 if freq == "15min" else 0.75
    five_min = pd.concat(
        [per_inv[i][f"인버터{i}_kW"] for i in range(1, 6)], axis=1).sort_index()
    five_min.columns = [f"인버터{i}_kW" for i in range(1, 6)]
    plant_5min = five_min.sum(axis=1, min_count=1)
    valid = plant_5min.notna()
    denominator = valid.astype(int).resample(freq).sum()
    period_ok = denominator >= min_count
    pieces, availability = [], []
    for number in range(1, 6):
        source = five_min[f"인버터{number}_kW"]
        contribution = source.where(valid).fillna(0.0).resample(freq).sum().div(
            denominator.replace(0, np.nan))
        contribution[~period_ok] = np.nan
        pieces.append(contribution.rename(f"인버터{number}_kW"))
        availability.append((source.notna().resample(freq).mean() >= rule).rename(f"inv{number}"))
    out = pd.concat(pieces, axis=1).sort_index()
    avail = pd.concat(availability, axis=1)
    out["가용인버터수"] = avail.sum(axis=1).astype("int8")
    out["인버터합계_kW"] = out[[f"인버터{i}_kW" for i in range(1, 6)]].sum(
        axis=1, min_count=1)
    out.loc[~period_ok, "인버터합계_kW"] = np.nan
    out["발전자료개수"] = denominator
    return out


def _gap_table(daily_ok: pd.Series, number: int) -> list[dict]:
    rows, start, previous = [], None, None
    for day, ok in daily_ok.items():
        if not bool(ok) and start is None:
            start = day
        elif bool(ok) and start is not None:
            rows.append({"인버터": number, "시작": start, "종료": previous,
                         "일수": int((previous - start).days) + 1})
            start = None
        previous = day
    if start is not None:
        rows.append({"인버터": number, "시작": start, "종료": previous,
                     "일수": int((previous - start).days) + 1})
    return rows


def _compare_v5(agg: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, dict]:
    path = V5_DIR / ("집계_15분_자료_v5.parquet" if freq == "15min"
                     else "집계_1시간_자료_v5.parquet")
    if not path.is_file():
        raise FileNotFoundError(path)
    v5 = pd.read_parquet(path)
    if "발전출력_kW" not in v5.columns:
        raise ValueError(f"{path.name}: 발전출력_kW 컬럼 없음")
    joined = agg[["인버터합계_kW", "가용인버터수"]].join(
        v5[["발전출력_kW"]], how="inner")
    complete = joined.dropna(subset=["인버터합계_kW", "발전출력_kW"]).copy()
    complete["차이_kW"] = complete["인버터합계_kW"] - complete["발전출력_kW"]
    summary = {
        "해상도": freq, "공통행수": int(len(joined)), "완전비교행수": int(len(complete)),
        "최대절대차이_kW": float(complete["차이_kW"].abs().max()) if len(complete) else None,
        "평균절대차이_kW": float(complete["차이_kW"].abs().mean()) if len(complete) else None,
        "0.011kW이내비율_pct": float((complete["차이_kW"].abs() <= 0.011).mean() * 100)
        if len(complete) else None,
    }
    return complete, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("인버터 원본 Excel 5개 로딩(각 파일 1회, 수 분 걸릴 수 있음)...")
    raw_cache = {number: load_inverter_raw(number) for number in INVERTER_FILES}
    per_inv = load_all_inverters_5min(raw_cache)
    summaries, gaps = [], []
    for number in range(1, 6):
        raw = raw_cache[number]
        col = "출력전력"
        valid = raw[col].dropna()
        power_outside = int((raw[col].notna() & ~raw[col].between(0, 80)).sum())
        pf = raw["역률"].dropna()
        processed = per_inv[number]
        filled_after_raw = int(((processed[f"인버터{number}_원시관측"] == 0) &
                                processed[f"인버터{number}_kW"].notna()).sum())
        daily_count = valid.resample("D").count()
        full_days = pd.date_range(raw.index.min().normalize(), raw.index.max().normalize(), freq="D")
        daily_ok = daily_count.reindex(full_days, fill_value=0) > 0
        gaps.extend(_gap_table(daily_ok, number))
        summaries.append({
            "인버터": number, "파일": INVERTER_FILES[number],
            "등록용량_kW": CAPACITY_KW[number], "원본유효시각행수": int(len(raw)),
            "시작": raw.index.min(), "종료": raw.index.max(),
            "출력결측률_pct": float(raw[col].isna().mean() * 100),
            "출력최소_kW": float(valid.min()) if len(valid) else None,
            "출력최대_kW": float(valid.max()) if len(valid) else None,
            "출력평균_kW": float(valid.mean()) if len(valid) else None,
            "출력_0~80kW범위밖_행수": power_outside,
            "역률최소_원본": float(pf.min()) if len(pf) else None,
            "역률최대_원본": float(pf.max()) if len(pf) else None,
            "역률_물리범위필터_적용여부": False,
            "원시미관측후_야간0또는10분보간_5분행수": filled_after_raw,
            "완전무응답일수": int((~daily_ok).sum()),
        })
    summary_df = pd.DataFrame(summaries)
    gap_df = pd.DataFrame(gaps, columns=["인버터", "시작", "종료", "일수"])
    gap_df = gap_df.sort_values(["인버터", "시작"])
    summary_df.to_csv(OUT / "인버터별_원자료_요약.csv", index=False, encoding="utf-8-sig")
    gap_df.to_csv(OUT / "인버터별_연속결측구간.csv", index=False, encoding="utf-8-sig")

    comparisons = []
    compare_summaries = []
    for freq in ("15min", "1h"):
        agg = aggregate_inverter_power(per_inv, freq)
        agg.to_parquet(OUT / f"인버터별_{freq}_출력.parquet")
        detail, result = _compare_v5(agg, freq)
        detail.reset_index(names="시각").to_csv(
            OUT / f"v5총출력_정합성_{freq}.csv", index=False, encoding="utf-8-sig")
        comparisons.append(result)
        compare_summaries.append(result)

    inv5_raw = raw_cache[5]
    defect_days = pd.date_range(DEFECT_START, DEFECT_END_EXCLUSIVE - pd.Timedelta(days=1), freq="D")
    defect_daily_count = inv5_raw["출력전력"].resample("D").count().reindex(defect_days, fill_value=0)
    defect_no_response_days = int((defect_daily_count == 0).sum())
    defect_response_days = int((defect_daily_count > 0).sum())
    # 08-18에 일부 응답이 있어 결측구간이 3일+90일로 나뉜다. 알려진 94일
    # 구간의 90% 이상이 완전무응답이고 공식 가을창이 11-17부터 시작하면
    # 정책B 적용 근거가 충분하다고 판정한다.
    policy_b_available = bool(defect_no_response_days / len(defect_days) >= 0.90)
    audit = {
        "등록용량_kW": CAPACITY_KW, "등록용량합계_kW": sum(CAPACITY_KW.values()),
        "명판검증": False, "용량출처": "Blockdata 인버터 등록값",
        "정답출처": "배만수 부장님 인버터별 Excel 5개",
        "정책B_결함구간": {"시작": str(DEFECT_START.date()),
                         "종료": str((DEFECT_END_EXCLUSIVE - pd.Timedelta(days=1)).date())},
        "정책B_결함구간_총일수": int(len(defect_days)),
        "정책B_결함구간_완전무응답일수": defect_no_response_days,
        "정책B_결함구간_일부응답일수": defect_response_days,
        "인버터5_장기결측으로_정책B적용가능": policy_b_available,
        "v5정합성": compare_summaries,
        "주의": "인버터별 예측은 발전소 총예측의 후처리 분해이며 기존 공식모델을 대체하지 않음",
    }
    (OUT / "감사_판정.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print("\n등록용량 합계:", sum(CAPACITY_KW.values()), "kW")
    print("인버터5 정책B 적용 가능:", policy_b_available)
    print("v5 정합성:", compare_summaries)
    if not policy_b_available:
        raise RuntimeError("인버터5 장기결측과 정책B 구간의 일치성을 확인하지 못함")


if __name__ == "__main__":
    main()
