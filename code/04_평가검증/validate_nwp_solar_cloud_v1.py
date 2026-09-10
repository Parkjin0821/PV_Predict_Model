"""수치예보모델 미래 일사량·운량(DSWRF/DSWRFLX/DIFSWRF/TCDC)을 ASOS 실측
일사량·전운량과 같은 시각 기준으로 비교해 MAE·상관계수를 산출한다.

이 스크립트는 `collect_gwangju_nwp_solar_cloud_710d_v1.py`의 수집 결과가
있어야 실행할 수 있다. 아직 수집이 끝나지 않았다면 안내 메시지를 출력하고
종료한다 (AGENTS.md 2026-08-19 절 참고).

비교 대상
- DSWRF(지면 하향단파복사, W/m^2) vs ASOS 일사량_W_m2
- TCDC(전운량, 0~1) vs ASOS 전운량_pct/100
- 참고용: DSWRFLX+DIFSWRF(직달+산란) 합이 DSWRF와 대략 같은 규모인지도 기록

주의
- 실측 기준 성능표가 아니라 "예보 대 실측 오차" 비교이므로, 이 결과를
  발전량 예측모델의 실측/예보 성능표와 혼동해 인용하지 않는다.
- ASOS 값은 정시(KST) 관측이므로 NWP 쪽도 동일한 KST 정시 타임스탬프로
  맞춘 뒤에만 비교한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


NWP_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_nwp_solar_cloud_710d_v1_2026-08-19"
)
NWP_CSV = NWP_DIR / "광주_수치예보_DSWRF_DSWRFLX_DIFSWRF_TCDC_710일.csv"

ASOS_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\기상청_ASOS156_시간환경_모델용.csv"
)

OUT_DIR = NWP_DIR / "검증"
FCST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def load_nwp_long() -> pd.DataFrame:
    df = pd.read_csv(NWP_CSV, dtype={"발표일": str})
    records = []
    for row in df.to_dict("records"):
        issue_day = datetime.strptime(row["발표일"], "%Y%m%d")
        target_day = issue_day + timedelta(days=1)
        for hour in FCST_HOURS:
            ts = target_day + timedelta(hours=hour)
            records.append(
                {
                    "timestamp": ts,
                    "DSWRF_fcst": row.get(f"DSWRF_{hour:02d}h"),
                    "DSWRFLX_fcst": row.get(f"DSWRFLX_{hour:02d}h"),
                    "DIFSWRF_fcst": row.get(f"DIFSWRF_{hour:02d}h"),
                    "TCDC_fcst": row.get(f"TCDC_{hour:02d}h"),
                }
            )
    long_df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    long_df["DSWRFLX_DIFSWRF_sum"] = long_df["DSWRFLX_fcst"] + long_df["DIFSWRF_fcst"]
    return long_df


def load_asos_hourly() -> pd.DataFrame:
    # 08-20 수정: 이 파일의 "시각" 컬럼이 이미 완전한 타임스탬프
    # ("YYYY-MM-DD HH:MM:SS")다. 별도 날짜 컬럼("관측날짜")은 없고
    # "목록순서"는 단순 일련번호라 원래 가정이 틀렸었다.
    df = pd.read_csv(ASOS_CSV)
    df["timestamp"] = pd.to_datetime(df["시각"])
    keep = df[["timestamp", "일사량_W_m2", "전운량_pct"]].copy()
    keep["전운량_frac"] = keep["전운량_pct"] / 100.0
    return keep


def summarize(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"n": 0, "mae": None, "corr": None}
    diff = (y_true[mask] - y_pred[mask]).abs()
    return {
        "n": int(mask.sum()),
        "mae": float(diff.mean()),
        "corr": float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1]),
    }


def main() -> int:
    if not NWP_CSV.exists():
        print(f"아직 수집 결과가 없습니다: {NWP_CSV}")
        print("collect_gwangju_nwp_solar_cloud_710d_v1.py 실행이 끝난 뒤 다시 실행하세요.")
        return 1

    nwp = load_nwp_long()
    asos = load_asos_hourly()
    merged = nwp.merge(asos, on="timestamp", how="inner")

    n_expected_days = pd.read_csv(NWP_CSV, dtype={"발표일": str})["발표일"].nunique()
    print(f"수집된 발표일 수: {n_expected_days} / 710")
    print(f"매칭된 시각 수(교집합): {len(merged)}")

    result_dswrf = summarize(merged["일사량_W_m2"], merged["DSWRF_fcst"])
    result_tcdc = summarize(merged["전운량_frac"], merged["TCDC_fcst"])
    result_sum_vs_dswrf = summarize(merged["DSWRF_fcst"], merged["DSWRFLX_DIFSWRF_sum"])

    print("\n[DSWRF vs ASOS 일사량_W_m2]")
    print(f"  n={result_dswrf['n']}, MAE={result_dswrf['mae']}, corr={result_dswrf['corr']}")
    print("[TCDC vs ASOS 전운량_frac]")
    print(f"  n={result_tcdc['n']}, MAE={result_tcdc['mae']}, corr={result_tcdc['corr']}")
    print("[참고: DSWRFLX+DIFSWRF 합 vs DSWRF (같은 모델 내부 정합성 점검)]")
    print(f"  n={result_sum_vs_dswrf['n']}, MAE={result_sum_vs_dswrf['mae']}, corr={result_sum_vs_dswrf['corr']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / "nwp_vs_asos_matched_timestamps.csv", index=False, encoding="utf-8-sig")
    summary_path = OUT_DIR / "nwp_vs_asos_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"수집된 발표일 수: {n_expected_days} / 710",
                f"매칭된 시각 수: {len(merged)}",
                f"DSWRF vs ASOS 일사량_W_m2: n={result_dswrf['n']}, MAE={result_dswrf['mae']}, corr={result_dswrf['corr']}",
                f"TCDC vs ASOS 전운량_frac: n={result_tcdc['n']}, MAE={result_tcdc['mae']}, corr={result_tcdc['corr']}",
                f"DSWRFLX+DIFSWRF 합 vs DSWRF: n={result_sum_vs_dswrf['n']}, MAE={result_sum_vs_dswrf['mae']}, corr={result_sum_vs_dswrf['corr']}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
