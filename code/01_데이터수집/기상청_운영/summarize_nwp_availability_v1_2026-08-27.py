# -*- coding: utf-8 -*-
"""nwp_가용성_실측_로그.csv를 일자별·런별 요약(Codex — 데이터 쌓인 뒤 아무 때나 실행).

`measure_nwp_availability_v1_2026-08-27.py`를 여러 날 반복 실행해 로그가
쌓인 뒤, 하루라도 실행할 때마다 이 스크립트로 현재까지의 그림을 본다.
재구현 없음 — 로그 CSV만 읽어 집계, 새 API 호출 0건.

## 산출
- 런(03시런/09시런)·목표일별로 "DSWRF가 처음 결측→값으로 바뀐 확인시각"
  (그 날 그 런의 최소 준비시각 추정치, 정확한 발표시각은 아니고 어디까지나
  "이 시각엔 확인해보니 있었다"는 하한선)
- 09시런이 17:00 KST 이전 확인 기록 중 값이 있었던 날 목록(2차 마감 전
  가용 여부의 직접 증거)
- DIFSWRF(대조군)가 항상 결측인지 확인(아니면 대조군 자체가 무효)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = Path(__file__).resolve().parent / "nwp_가용성_실측_로그.csv"


def main() -> None:
    if not LOG_PATH.exists():
        raise RuntimeError(f"로그 없음: {LOG_PATH} — measure_nwp_availability_v1_2026-08-27.py를 먼저 실행할 것.")
    df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    df["확인시각_KST"] = pd.to_datetime(df["확인시각_KST"])
    df["확인일"] = df["확인시각_KST"].dt.date
    df["확인시각_시분"] = df["확인시각_KST"].dt.strftime("%H:%M")

    dswrf = df[df["변수"] == "DSWRF"].copy()
    dswrf["가용"] = ~dswrf["결측_09KST"].astype(bool)  # 09KST(주간) 값 기준 — 밤은 0이라 결측여부 무관

    print("=== 대조군 확인: DIFSWRF는 항상 결측이어야 정상(아니면 대조군 무효) ===")
    dif = df[df["변수"] == "DIFSWRF"]
    print(f"DIFSWRF 결측 아닌 행수: {int((~dif['결측_09KST'].astype(bool)).sum())} / {len(dif)} (0이어야 정상)")

    print("\n=== 런·확인일별 최초 가용 확인시각(하한선 추정) ===")
    first_ok = (dswrf[dswrf["가용"]]
                .groupby(["런", "확인일"])["확인시각_시분"].min()
                .reset_index().rename(columns={"확인시각_시분": "최초가용확인시각"}))
    print(first_ok.to_string(index=False) if len(first_ok) else "아직 가용 확인 기록 없음")

    print("\n=== 09시런이 17:00 이전에 가용했던 날(2차 마감 안전 증거) ===")
    nine = dswrf[dswrf["런"].str.contains("09시런")]
    before_17 = nine[(nine["가용"]) & (nine["확인시각_KST"].dt.hour < 17)]
    if len(before_17):
        print(before_17.groupby("확인일")["확인시각_시분"].min().to_string())
    else:
        print("아직 없음(17시 이전 09시런 가용 확인 기록 0건) — 계속 측정 필요")

    print("\n=== 03시런 최초 가용 확인시각(1차 10시 마감 안전마진 계산용) ===")
    three = dswrf[dswrf["런"].str.contains("03시런")]
    three_ok = three[three["가용"]]
    if len(three_ok):
        print(three_ok.groupby("확인일")["확인시각_시분"].min().to_string())
    else:
        print("아직 없음")


if __name__ == "__main__":
    main()
