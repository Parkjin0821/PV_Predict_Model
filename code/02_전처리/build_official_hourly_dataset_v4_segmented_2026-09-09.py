# -*- coding: utf-8 -*-
"""공식 시간단위 데이터셋 v4 — 710일 구간(~2026-08-04) + 신규 라이브 구간
(2026-08-26~) 세그먼트 병합, 08-05~08-25 결측 구간은 절대 보간하지 않음.

09-09 배경: 광주 "단기"(D+1/+24h/+48h) 모델을 최신 데이터로 재학습하기로
했으나, 원본 입력(710일 CSV)이 2026-08-04에서 멈춘 정적 스냅샷이고
2026-08-05~08-25는 발전량·NWP 원자료 자체가 없다(ASOS만 있음). 사용자
(및 코덱스)가 지정한 안전조건을 그대로 구현한다:

1. 전체 기간을 연속 1시간 인덱스로 먼저 reindex
2. 08-05~08-25는 실제 NaN 유지 - 보간·전방채움 금지
3. shift·rolling은 reindex 이후, **segment_id로 그룹핑해서** 계산
   (세그먼트를 넘어서는 shift/rolling이 구조적으로 불가능하게 만듦 -
   단순 reindex+shift만으로는 rolling(min_periods=...)이 결측 직후에도
   일부값을 만들 수 있다는 코덱스 지적을 완전히 막지 못하므로, groupby
   자체로 원천 차단한다)
4. 결측 경계를 넘는 ffill 금지(신규 세그먼트 NWP도 forward-fill 없음)
5. 경계 이후 최대 lag/rolling 길이(24시간)만큼 추가 제외
6. 기존 710일과 신규 구간의 학습·평가 행수를 매니페스트에 별도 기록
7. 산출물은 "공식 후보"(정식 아님, 08-05~08-25 백필 후 최종 재검증
   필요)로 명시

특성 범위: `live_feature_assembler_단기_v1_2026-08-26.py`의
`build_live_hourly()`(golden replay 검증된 라이브 재조립 함수)를 신규
구간(08-26~) 원자료 재구성에 그대로 재사용한다. 이 함수로 최근 16일을
직접 뽑아 컬럼별 결측률을 실측한 결과, 다음 컬럼은 지금 라이브
파이프라인에서 사실상 재현 불가능함을 확인했다(전부 100% 또는 계절
전체 결측):
  DSWRFLX, DIFSWRF, DSWRFLX_bsrn정제, DIFSWRF_bsrn정제,
  추정_일조시간_hr, 추정_모듈표면온도, 추정_출력온도,
  mean_inverter_temperature_c, mean_communication_ok
→ 이 컬럼들은 신규(v4) 특성 셋에서 제외한다(구모델이 이 컬럼들을
요구해서 09-03부터 라이브 예측이 단 한 번도 성공하지 못한 것과 정확히
일치하는 원인).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

KST = ZoneInfo("Asia/Seoul")

OLD_MASTER_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v3_fixed_tm_2026-08-21.csv"
)
ASSEMBLER_PY = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
    r"\03_모델학습\현재_종합파이프라인\live_feature_assembler_단기_v1_2026-08-26.py"
)
OUT_DIR = Path(r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed")
OUT_CSV = OUT_DIR / "gwangju_1hour_model_dataset_official_v4_segmented_candidate_2026-09-09.csv"
OUT_MANIFEST = OUT_DIR / "gwangju_1hour_model_dataset_official_v4_segmented_candidate_2026-09-09.manifest.json"

# 09-09 실측: 구간 경계(양쪽 다 실제 유효데이터가 있는 마지막/처음 시각)
OLD_SEGMENT_END = pd.Timestamp("2026-08-04 22:00:00")   # 710일 CSV 마지막 유효 행
NEW_SEGMENT_START = pd.Timestamp("2026-08-26 00:00:00")  # 블록데이터 첫날(08-25, 부분일) 다음 첫 완전일
NEW_SEGMENT_END = pd.Timestamp("2026-09-09 15:00:00")    # 오늘 최신 라이브 수신 시각까지

MAX_WINDOW_H = 24  # lag/rolling 최대 창(발전출력_24시간전_kW, roll_24h 등) - 경계 추가제외 폭

# build_live_hourly()가 만드는 컬럼 중, 지금 라이브 파이프라인에서 사실상
# 재현 불가능(전부/거의 전부 결측)임을 09-09 실측으로 확인해 제외하는 것들.
DEAD_LIVE_COLUMNS = [
    "DSWRFLX", "DIFSWRF", "DSWRFLX_bsrn정제", "DIFSWRF_bsrn정제",
    "추정_일조시간_hr", "추정_모듈표면온도", "추정_출력온도",
    "mean_inverter_temperature_c", "mean_communication_ok",
    "WSD",  # NWP발 WSD는 100% 결측 - GRID_WSD로 대체
]


def load_assembler():
    spec = importlib.util.spec_from_file_location("assembler_단기", ASSEMBLER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_new_segment() -> pd.DataFrame:
    mod = load_assembler()
    lookback_hours = int((NEW_SEGMENT_END - NEW_SEGMENT_START).total_seconds() / 3600) + 48
    raw = mod.build_live_hourly(NEW_SEGMENT_END, lookback_hours=lookback_hours)
    raw.index = pd.DatetimeIndex(raw.index)
    seg = raw.loc[(raw.index >= NEW_SEGMENT_START) & (raw.index <= NEW_SEGMENT_END)].copy()
    drop_cols = [c for c in DEAD_LIVE_COLUMNS if c in seg.columns]
    seg = seg.drop(columns=drop_cols)
    return seg


def load_old_segment() -> pd.DataFrame:
    df = pd.read_csv(OLD_MASTER_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    df = df.loc[df.index <= OLD_SEGMENT_END].copy()
    drop_cols = [c for c in DEAD_LIVE_COLUMNS if c in df.columns]
    # 구세그먼트는 이 컬럼들이 실제로 있으므로 지우지 않는다 - 대신 세그먼트별로
    # 사용가능한 컬럼이 다르다는 사실 자체를 매니페스트에 남긴다.
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    old = load_old_segment()
    new = build_new_segment()

    common_cols = sorted(set(old.columns) & set(new.columns))
    old_only_cols = sorted(set(old.columns) - set(new.columns))
    new_only_cols = sorted(set(new.columns) - set(old.columns))
    print(f"[1] 구세그먼트 {len(old)}행({old.index.min()}~{old.index.max()}), "
          f"신세그먼트 {len(new)}행({new.index.min() if len(new) else '-'}~{new.index.max() if len(new) else '-'})")
    print(f"    공통 컬럼 {len(common_cols)}개, 구세그먼트 전용 {len(old_only_cols)}개, "
          f"신세그먼트 전용 {len(new_only_cols)}개")

    # 2) 안전조건 1·4: 전체 기간을 연속 1시간 인덱스로 reindex, ffill 금지
    #    (reindex만 하고 어떤 값도 채우지 않음 - 08-05~08-25는 자동으로 전부 NaN)
    full_index = pd.date_range(old.index.min(), new.index.max() if len(new) else old.index.max(), freq="1h")
    old_r = old.reindex(full_index)
    new_r = new.reindex(full_index)

    merged = old_r.combine_first(new_r) if len(new) else old_r
    # combine_first는 첫 프레임(old_r) 값을 우선하고 결측만 new_r로 채운다.
    # 구간이 겹치지 않으므로(구:~08-04, 신:08-26~) 실질적으로는 이어붙이기와 동일.

    # 3) segment_id 부여 - 세그먼트를 넘는 shift/rolling을 구조적으로 차단하기 위함
    merged["segment_id"] = np.where(
        merged.index <= OLD_SEGMENT_END, "seg1_710d",
        np.where(merged.index >= NEW_SEGMENT_START, "seg2_live_0826", "gap_no_data"),
    )
    n_gap = int((merged["segment_id"] == "gap_no_data").sum())
    n_seg1 = int((merged["segment_id"] == "seg1_710d").sum())
    n_seg2 = int((merged["segment_id"] == "seg2_live_0826").sum())
    gap_start = OLD_SEGMENT_END + pd.Timedelta(hours=1)
    gap_end = NEW_SEGMENT_START - pd.Timedelta(hours=1)
    print(f"[2] 연속 인덱스 reindex 완료: 전체 {len(merged)}행 "
          f"(seg1={n_seg1}, gap={n_gap}, seg2={n_seg2})")
    print(f"    결측 경계(실제 시각 기준): {gap_start} ~ {gap_end} "
          f"({(gap_end - gap_start).total_seconds() / 3600 + 1:.0f}시간)")

    # 4) 안전조건 5: 경계 이후 최대 lag/rolling 길이만큼 추가 제외
    #    (segment_id 그룹핑으로 이미 세그먼트 간 오염은 불가능하지만, 각
    #    세그먼트 "시작" 직후 MAX_WINDOW_H시간은 그 세그먼트 내부에서도
    #    rolling 창이 완전히 안 찬 상태이므로 학습에서 별도 표시해 제외한다)
    merged["boundary_excluded"] = False
    for seg_name in ["seg1_710d", "seg2_live_0826"]:
        idx = merged.index[merged["segment_id"] == seg_name]
        if len(idx) == 0:
            continue
        seg_start = idx.min()
        excl_end = seg_start + pd.Timedelta(hours=MAX_WINDOW_H - 1)
        merged.loc[(merged["segment_id"] == seg_name) & (merged.index <= excl_end), "boundary_excluded"] = True
    n_boundary_excl = int(merged["boundary_excluded"].sum())
    print(f"[3] 세그먼트 시작 경계 추가제외(각 세그먼트 첫 {MAX_WINDOW_H}시간): {n_boundary_excl}행")

    merged = merged.reset_index().rename(columns={"index": "time"})
    merged.to_csv(OUT_CSV, index=False)

    manifest = {
        "status": "공식_후보(정식 아님) - 08-05~08-25 실백필 후 최종 재검증 필요",
        "생성시각_kst": pd.Timestamp.now(tz=KST).isoformat(),
        "세그먼트": {
            "seg1_710d": {"기간": [str(old.index.min()), str(OLD_SEGMENT_END)], "행수": n_seg1},
            "gap_no_data": {"기간": [str(gap_start), str(gap_end)],
                            "시간수": round((gap_end - gap_start).total_seconds() / 3600 + 1, 1),
                            "행수": n_gap, "비고": "발전량·NWP 원자료 없음 - 보간 금지, 전부 NaN 유지"},
            "seg2_live_0826": {"기간": [str(NEW_SEGMENT_START), str(new.index.max()) if len(new) else None],
                               "행수": n_seg2},
        },
        "경계_추가제외": {"창_시간": MAX_WINDOW_H, "제외행수": n_boundary_excl,
                     "비고": "각 세그먼트 시작 직후 lag/rolling이 완전히 안 찬 구간"},
        "컬럼_비교": {
            "공통": common_cols,
            "구세그먼트_전용(신규구간_재현불가로_확인됨)": old_only_cols,
            "신세그먼트_전용": new_only_cols,
        },
        "안전조건_구현": [
            "1) 전체기간 연속 1시간 reindex 완료",
            "2) 08-05~08-25 실제 NaN 유지, 보간/ffill 없음(reindex만 수행)",
            "3) shift/rolling은 이 파일 이후 별도 학습스크립트에서 segment_id로 groupby 후 계산 예정",
            "4) NWP ffill 없음(reindex만, DSWRF 등 실제결측 그대로)",
            "5) 세그먼트 시작 후 24시간 boundary_excluded=True로 표시",
            "6) 세그먼트별 행수 위에 기록",
            "7) status를 공식_후보로 명시",
        ],
        "출력_csv": str(OUT_CSV),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4] 저장 완료: {OUT_CSV}")
    print(f"    매니페스트: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
