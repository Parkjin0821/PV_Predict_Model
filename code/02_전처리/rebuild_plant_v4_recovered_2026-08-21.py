# -*- coding: utf-8 -*-
"""★v4 전체 재생성★ 인버터 부분가용 복구를 15분·1시간·일간에 일괄 적용
   + 완결성 감사표 생성 (사용자 확정 재작업 순서 1~3번).

## 왜 필요한가 (08-21 발견)
`sum(min_count=5)`("5대 전부 있어야 유효") 규칙 때문에 **인버터 1대만
빠져도 발전소 총출력이 통째로 NaN**이 된다. 08-21에 이 문제를
**일간 자료에서만** 복구했고(`집계_일간_실제발전량_v2_인버터5부분복구`),
**15분·1시간 자료에는 반영하지 않았다.** 그 결과:

| 구간 | 낮시간 `plant_output_kw` 결측 | 시간대별 양상 |
|---|---:|---|
| 2025-08-15~10-20 | **80.2%** | 06~08시 0% / 09시 70% / **10~19시 99~100%** |
| 2025-10-21~12-14 | **39.0%** | 10~17시 약 50% |

즉 **발전량·오차가 가장 큰 정오대가 통째로 빠진 채** 초단기·단기 모델이
학습·평가돼 왔다. 2차 공식 E2E의 폴드별 실제 커버리지도 심하게
불균형하다(단기 +24h: 가을 287행 vs 초여름 1,550행 — 하루평균 5.4시각
vs 14.0시각). "동일 5계절 rolling-origin"이라는 서술이 실질적으로
성립하지 않는다.

## 이 스크립트가 하는 일 (재작업 순서 1~3번을 한 번에)
1. **원인·정답출처 확정**: 원본 인버터별 엑셀 5개를 **전 기간** 직접 읽어
   인버터별 가용성을 일·시간 단위로 진단하고, **인버터5 외 다른 결측
   구간도 전부** 찾아낸다. 발전소 총출력의 정답 출처를 "원본 엑셀 5개의
   살아있는 인버터 합"으로 확정한다.
2. **15분·1시간·일간을 같은 복구 원칙으로 재생성**(v4).
3. **날짜×시간대×인버터 가용대수 완결성 감사표** 생성.
4. (추가) 기존 v3 대비 **정합성 검증** — 결측이 아니었던 구간에서는
   v4와 v3가 사실상 일치해야 한다. 불일치하면 v4를 신뢰하지 않는다.

## ★복구 원칙(기존 일간 복구와 동일 — 새로 만들지 않음)★
- **살아있는 인버터만 합산**(`sum(skipna=True, min_count=1)`).
  결측 인버터를 0으로 대체하지 않는다.
- **스케일업(임의 추정) 금지**: 4/5 합계를 5/4배 해서 "5대였다면"으로
  추정하지 않는다. 5번 인버터의 실제 출력을 알 방법이 없다(음영·배치가
  다를 수 있음). **그대로 부분합을 쓰고, 부분합임을 컬럼으로 표시한다.**
- **평균계열(전압·주파수·역률·온도)은 살아있는 인버터의 평균**으로 재계산
  (합이 아니라 평균이므로 부분가용이어도 의미가 유지된다).
- 집계 규칙은 `pv_pipeline.aggregate_power`/`build_daily_actual`과
  **동일**하게 유지한다(15분 충족률·1시간 충족률·낮시간 90% 규칙).
- **투명성**: 모든 산출물에 `가용인버터수`와 `부분가용여부`를 남긴다.
  모델 학습 시 이 구간을 어떻게 다룰지는 **별도 판단**이며, 이
  스크립트는 "값을 되살리고 그 성격을 표시"하는 데까지만 한다.

## ★주의: 부분가용 구간을 그대로 학습에 넣으면 안 된다★
4/5 인버터 부분합은 **같은 일사량에서도 총출력이 구조적으로 낮다**.
아무 표시 없이 학습·평가에 넣으면 모델이 "이 시기엔 원래 덜 나온다"로
잘못 배운다. 그래서 `부분가용여부`·`가용인버터수`를 반드시 남기고,
**후속 모델링 단계에서 ①제외 ②용량정규화(가용대수 비례) ③특성으로 투입
중 무엇을 쓸지 별도 실험으로 정한다**(이 스크립트는 정하지 않는다).

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\02_전처리"
python rebuild_plant_v4_recovered_2026-08-21.py
```
API 호출 없음(로컬 엑셀 읽기·재집계만). 엑셀 5개(약 64만행) 파싱이라
수 분 소요될 수 있다.

## 산출물 (`03_모델학습/현재_종합파이프라인/outputs/v4_복구_2026-08-21/`)
- `인버터_가용성_일별.csv` — 날짜×인버터별 관측행수·가용여부
- `결측구간_목록.csv` — 인버터별 연속 결측구간(시작·종료·일수)
- `완결성_감사표_날짜시간대.csv` — 날짜×시(hour)×가용대수×결측률
- `집계_15분_자료_v4.parquet` / `집계_1시간_자료_v4.parquet`
- `집계_일간_실제발전량_v4.parquet`
- `정합성검증_v3대비.csv` — 비결측 구간에서 v3와 일치하는지
- `요약.txt`
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PIPE_DIR = Path(__file__).resolve().parents[1] / "03_모델학습" / "현재_종합파이프라인"
_spec = importlib.util.spec_from_file_location("pv_pipeline", PIPE_DIR / "pv_pipeline.py")
pv_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["pv_pipeline"] = pv_pipeline
_spec.loader.exec_module(pv_pipeline)

INVERTER_DIR = Path(r"C:\Users\u-cube\JIN\태양광 발전\배만수 부장님 엑셀 파일")
INVERTER_FILES = {
    1: "광주 광주시청 _1번 인버터 로그_계산됨.xlsx",
    2: "광주 광주시청_2번 인버터 로그_계산됨.xlsx",
    3: "광주 광주시청_3번 인버터 로그_계산됨.xlsx",
    4: "광주 광주시청_4번 인버터 로그_계산됨.xlsx",
    5: "광주 광주시청_5번 인버터 로그_계산됨.xlsx",
}
OUT = PIPE_DIR / "outputs" / "v4_복구_2026-08-21"

# 합산 대상(발전소 총량) / 평균 대상(대표값)
SUM_COLS = {"출력전력": "발전출력_kW", "입력전력": "입력전력_kW", "입력전류": "입력전류_A"}
MEAN_COLS = {"입력전압": "입력전압평균_V", "주파수": "주파수평균_Hz",
             "역률": "역률평균", "온도": "인버터평균온도_C"}
N_INVERTERS = 5


def load_inverter(n: int) -> pd.DataFrame:
    """원본 엑셀 1개 → 5분 격자 정렬. **이게 발전소 총출력의 정답 출처다.**"""
    usecols = ["생성일"] + list(SUM_COLS) + list(MEAN_COLS) + ["통신에러"]
    df = pd.read_excel(INVERTER_DIR / INVERTER_FILES[n], usecols=usecols)
    df["생성일"] = pd.to_datetime(df["생성일"], errors="coerce")
    df = df.dropna(subset=["생성일"]).set_index("생성일").sort_index()
    for c in list(SUM_COLS) + list(MEAN_COLS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 원 파이프라인과 동일하게 5분 버킷 평균으로 정렬(원시 타임스탬프가 초 단위로 어긋남)
    out = df[list(SUM_COLS) + list(MEAN_COLS)].resample("5min").mean()
    out["_관측행수"] = df[list(SUM_COLS)[:1]].resample("5min").count().iloc[:, 0]
    return out


def find_gaps(daily_ok: pd.Series, label: str) -> list[dict]:
    """연속 결측구간을 (시작, 종료, 일수)로 뽑는다."""
    gaps, run_start, prev = [], None, None
    for day, ok in daily_ok.items():
        if not ok and run_start is None:
            run_start = day
        elif ok and run_start is not None:
            gaps.append({"대상": label, "시작": run_start, "종료": prev,
                         "일수": int((prev - run_start).days) + 1})
            run_start = None
        prev = day
    if run_start is not None:
        gaps.append({"대상": label, "시작": run_start, "종료": prev,
                     "일수": int((prev - run_start).days) + 1})
    return gaps


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = pv_pipeline.load_config()
    capacity = float(config["site"]["capacity_kw"])

    print("=== 1) 원본 인버터 엑셀 5개 로드(정답 출처 확정) ===")
    per_inv = {}
    for n in range(1, N_INVERTERS + 1):
        per_inv[n] = load_inverter(n)
        s = per_inv[n]
        print(f"  인버터{n}: {len(s):,}개 5분버킷, {s.index.min()} ~ {s.index.max()}")

    grid = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in per_inv.values()])))
    grid = pd.date_range(grid.min(), grid.max(), freq="5min")

    # ── 인버터별 가용성 진단(일별) ──
    print("\n=== 2) 인버터별 가용성 진단 + 결측구간 전수 탐색 ===")
    avail_rows, gaps = [], []
    for n in range(1, N_INVERTERS + 1):
        cnt = per_inv[n]["_관측행수"].reindex(grid).fillna(0)
        daily = cnt.groupby(cnt.index.normalize()).sum()
        for day, c in daily.items():
            avail_rows.append({"날짜": day, "인버터": n, "관측행수": int(c), "가용": bool(c > 0)})
        gaps += find_gaps(daily > 0, f"인버터{n}")
        bad = int((daily == 0).sum())
        print(f"  인버터{n}: 완전무응답 {bad}일")
    avail_df = pd.DataFrame(avail_rows)
    avail_df.to_csv(OUT / "인버터_가용성_일별.csv", index=False, encoding="utf-8-sig")
    gaps_df = pd.DataFrame(gaps).sort_values(["대상", "시작"]) if gaps else pd.DataFrame()
    gaps_df.to_csv(OUT / "결측구간_목록.csv", index=False, encoding="utf-8-sig")
    if len(gaps_df):
        print("\n  ★발견된 결측구간(3일 이상만 표시)★")
        big = gaps_df[gaps_df["일수"] >= 3]
        print(big.to_string(index=False) if len(big) else "    (3일 이상 없음)")

    # ── 5분 총출력 재계산(살아있는 인버터만, 스케일업 없음) ──
    print("\n=== 3) 5분 발전소 집계 재생성(min_count=1) ===")
    five = pd.DataFrame(index=grid)
    for src, dst in SUM_COLS.items():
        stacked = pd.concat([per_inv[n][src].reindex(grid) for n in range(1, N_INVERTERS + 1)], axis=1)
        five[dst] = stacked.sum(axis=1, skipna=True, min_count=1)   # ★핵심★ 1대만 살아도 합산
    for src, dst in MEAN_COLS.items():
        stacked = pd.concat([per_inv[n][src].reindex(grid) for n in range(1, N_INVERTERS + 1)], axis=1)
        five[dst] = stacked.mean(axis=1, skipna=True)
    avail_stack = pd.concat(
        [per_inv[n]["_관측행수"].reindex(grid).fillna(0) > 0 for n in range(1, N_INVERTERS + 1)], axis=1)
    five["가용인버터수"] = avail_stack.sum(axis=1).astype("int8")
    five["부분가용여부"] = ((five["가용인버터수"] > 0) & (five["가용인버터수"] < N_INVERTERS)).astype("int8")
    five["통신정상비율"] = five["가용인버터수"] / N_INVERTERS

    # 물리범위 필터(기존 파이프라인과 동일)
    five.loc[~five["발전출력_kW"].between(0, capacity), "발전출력_kW"] = np.nan
    five.loc[five["입력전력_kW"] < 0, "입력전력_kW"] = np.nan
    five.loc[~five["인버터평균온도_C"].between(-50, 80), "인버터평균온도_C"] = np.nan

    elev, azi = pv_pipeline.solar_position(
        five.index, float(config["site"]["latitude"]), float(config["site"]["longitude"]))
    five["태양고도_deg"] = elev
    five["태양방위각_deg"] = azi
    five["물리적낮"] = (elev > 0).astype("int8")
    five["핵심낮시간"] = (elev > 10).astype("int8")
    print(f"  5분 {len(five):,}행 / 발전출력 결측 {five['발전출력_kW'].isna().mean()*100:.1f}%"
          f" / 부분가용 {five['부분가용여부'].mean()*100:.1f}%")

    # ── 15분·1시간 집계(기존 규칙 동일) ──
    print("\n=== 4) 15분·1시간·일간 재집계(기존 충족률 규칙 유지) ===")
    # ★충족률은 반드시 원 파이프라인 `build_all()`과 동일해야 한다★
    # (15분=1.0, 1시간=0.75). 다르게 두면 v3와 비교 자체가 성립하지 않고
    # "복구 효과"와 "규칙 변경 효과"가 뒤섞인다.
    f15 = pv_pipeline.aggregate_power(five, "15min", required_fraction=1.0)
    f1h = pv_pipeline.aggregate_power(five, "1h", required_fraction=0.75)
    for f, freq in ((f15, "15min"), (f1h, "1h")):
        f["가용인버터수"] = five["가용인버터수"].resample(freq).max()
        f["부분가용여부"] = (five["부분가용여부"].resample(freq).max()).astype("int8")
    daily = pv_pipeline.build_daily_actual(five)
    day_only = five[five["물리적낮"] == 1]
    daily["가용인버터수_낮시간평균"] = day_only.groupby(day_only.index.normalize())["가용인버터수"].mean()
    daily["부분가용일"] = (daily["가용인버터수_낮시간평균"] < N_INVERTERS).astype("int8")

    f15.to_parquet(OUT / "집계_15분_자료_v4.parquet")
    f1h.to_parquet(OUT / "집계_1시간_자료_v4.parquet")
    daily.to_parquet(OUT / "집계_일간_실제발전량_v4.parquet")
    print(f"  15분 {len(f15):,}행(결측 {f15['발전출력_kW'].isna().mean()*100:.1f}%)"
          f" / 1시간 {len(f1h):,}행(결측 {f1h['발전출력_kW'].isna().mean()*100:.1f}%)"
          f" / 일간 {len(daily):,}일(유효 {daily['일간발전량_kWh'].notna().sum()}일)")

    # ── 완결성 감사표(날짜 × 시간대) ──
    print("\n=== 5) 완결성 감사표(날짜×시간대×가용대수) ===")
    a = f1h.copy()
    a["날짜"] = a.index.normalize()
    a["시"] = a.index.hour
    audit = a.reset_index()[["날짜", "시", "발전출력_kW", "가용인버터수",
                             "부분가용여부", "물리적낮", "발전자료충족률"]]
    audit["타깃결측"] = audit["발전출력_kW"].isna().astype("int8")
    audit.to_csv(OUT / "완결성_감사표_날짜시간대.csv", index=False, encoding="utf-8-sig")
    day_audit = audit[audit["물리적낮"] == 1]
    print("  낮시간 타깃결측률(시간대별):")
    print("   ", dict(day_audit.groupby("시")["타깃결측"].apply(lambda x: round(x.mean() * 100, 1))))

    # ── v3 대비 정합성 검증 ──
    print("\n=== 6) 기존 v3 대비 정합성 검증(비결측 구간은 일치해야 정상) ===")
    checks = []
    old15 = PIPE_DIR / "outputs" / "집계_15분_자료.parquet"
    old1h = PIPE_DIR / "outputs" / "집계_1시간_자료.parquet"
    for label, oldp, new in (("15분", old15, f15), ("1시간", old1h, f1h)):
        if not oldp.exists():
            continue
        old = pd.read_parquet(oldp)["발전출력_kW"]
        common = old.index.intersection(new.index)
        both = old.reindex(common).notna() & new["발전출력_kW"].reindex(common).notna()
        d = (old.reindex(common)[both] - new["발전출력_kW"].reindex(common)[both]).abs()
        newly = int(old.reindex(common).isna().sum() - new["발전출력_kW"].reindex(common).isna().sum())
        checks.append({"대상": label, "공통시각": len(common), "양쪽유효": int(both.sum()),
                       "최대절대차_kW": round(float(d.max()) if len(d) else 0.0, 4),
                       "평균절대차_kW": round(float(d.mean()) if len(d) else 0.0, 4),
                       "중앙비율_v4대v3": round(float((new['발전출력_kW'].reindex(common)[both]
                                                  / old.reindex(common)[both].replace(0, np.nan)).median()), 4),
                       "신규복구_시각수": newly})
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "정합성검증_v3대비.csv", index=False, encoding="utf-8-sig")
    print(chk.to_string(index=False))

    lines = [
        "v4 복구 재생성 요약",
        f"정답 출처: 원본 인버터별 엑셀 5개(살아있는 인버터만 합산, min_count=1, 스케일업 없음)",
        f"5분 {len(five):,}행 / 15분 {len(f15):,}행 / 1시간 {len(f1h):,}행 / 일간 {len(daily):,}일",
        f"1시간 발전출력 결측률 {f1h['발전출력_kW'].isna().mean()*100:.2f}%",
        f"부분가용(1~4대) 5분버킷 비율 {five['부분가용여부'].mean()*100:.2f}%",
        "",
        "★후속 모델링에서 반드시 결정할 것★",
        "부분가용 구간(4/5 등)은 같은 일사량에서도 총출력이 구조적으로 낮다.",
        "①제외 ②가용대수 비례 정규화 ③특성으로 투입 중 무엇을 쓸지 실험으로 정할 것.",
        "이 스크립트는 값을 되살리고 성격을 표시하는 데까지만 한다.",
    ]
    (OUT / "요약.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
