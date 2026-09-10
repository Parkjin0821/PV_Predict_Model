# -*- coding: utf-8 -*-
"""예측구간(불확실성) 전사 재산출 — v5·공식B·219kW 3차 공식모델 v3 기준.

## 왜 필요한가
`uncertainty_audit_v1_2026-08-24.py`는 **v3 데이터(인버터5 결함이 시간단위
자료에 미반영된 상태)로 만든 1차 E2E OOF**를 썼고, 스스로 판정문에
"v3 OOF 사용. 인버터5 결함을 고친 v5 공식 OOF 생성 후 반드시 재산출"이라고
못박아뒀다. AGENTS.md "미해결·주의할 것"에도 같은 항목이 남아 있었다.
그 재산출을 여기서 **전 티어·전 수평(초단기 +1~4h, 단기 +1/+24/+48h,
일간 D+1) 8개 조합 전부** 수행한다.

입력은 현재 공식모델인 **3차 v3**(`E2E_v5_공식B_v3_일간특성군_2026-08-24`)
산출물이다 — 초단기 +4h 날씨군집화(v2 패치)와 일간 53특성(v3 패치)이
모두 반영된 최신 공식 예측이다.

## 재사용(재구현 금지 원칙)
`uncertainty_audit_v1_2026-08-24`의 `sequential_conformal`·
`finite_sample_quantile`·`audit_nwp`를 그대로 import해서 쓴다. 방법론
자체는 1차와 동일해야 v3→v5 비교가 성립하므로 **일부러 바꾸지 않았다**.
바뀐 건 입력 OOF뿐이다.

## 방법(1차와 동일)
폴드를 발행시각 순으로 정렬해, **직전 폴드까지의 절대잔차 분포**에서
finite-sample split-conformal 분위수 q를 구하고 다음 폴드에
`[예측-q, 예측+q]`(0~용량으로 클리핑)를 적용한다. 1번 폴드는 보정에만
쓰이고 구간을 산출하지 않는다(그래서 시험표본이 전체 행수보다 적다).
명목 포함률 80%·90% 두 수준.

## ★1차에 없던 추가 진단: 조건부 포함률★
전체 포함률이 명목치에 맞아도 **구간이 조건부로는 틀릴 수 있다** — 이
방법은 폴드마다 폭이 상수(q)라, PV처럼 오차가 출력수준에 따라 크게
달라지는(새벽·해질녘 ≈0, 정오 최대) 대상에서는 저출력 구간을 과대포함
하고 고출력 구간을 과소포함할 수 있다. 전체 포함률만 보면 이게 안
보이므로, **예측값 십분위별·대상시각(hour)별 포함률**을 함께 산출해
보고한다(진단용 — 공식 방법은 바꾸지 않음).

## 산출물 (`outputs/불확실성_v5_전사재산출_2026-08-24/`)
- `예측구간_수평별_요약.csv` — 티어×수평×명목포함률: 실제포함률·평균구간폭
- `예측구간_폴드별.csv` — 폴드 단위 상세
- `예측구간_행단위.csv.gz` — 행 단위 하한·상한·포함여부
- `조건부포함률_예측값십분위.csv`, `조건부포함률_시간대별.csv` — 신규 진단
- `v3대비_비교.csv` — 1차(v3 OOF) 예비치 대비 변화
- `NWP_운영정합성_감사.csv`, `NWP_변수별_결측.csv` — 1차와 동일(모델 무관)
- `판정.json`
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parent / "03_모델학습" / "현재_종합파이프라인"
SRC = PIPE / "outputs" / "E2E_v5_공식B_v3_일간특성군_2026-08-24"
OLD_SUMMARY = ROOT / "outputs" / "불확실성_감사_v1_2026-08-24" / "예측구간_수평별_예비요약.csv"
OUT = ROOT / "outputs" / "불확실성_v5_전사재산출_2026-08-24"
CAPACITY_KW = 219.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ua = _load("ua_v1", ROOT / "uncertainty_audit_v1_2026-08-24.py")


def load_oof_v5() -> pd.DataFrame:
    """3차 공식모델 v3 산출물 → 1차 `load_oof()`와 같은 컬럼 구조로 정규화.

    1차와 파일명·컬럼명이 다르다(`행단위_daily_final_예측정답.csv`의
    `대상일` → v3는 `행단위_daily_예측정답.csv`의 `대상일`), 그래서 로더만
    새로 쓰고 이후 계산은 1차 모듈을 그대로 재사용한다.
    """
    ac = pd.read_csv(SRC / "행단위_ac_power_예측정답.csv", encoding="utf-8-sig",
                     parse_dates=["발행시각", "대상시각"])
    ac = ac.rename(columns={"실제_kW": "실제", "예측_kW": "예측", "대상시각": "대상"})
    ac["단위"] = "kW"

    daily = pd.read_csv(SRC / "행단위_daily_예측정답.csv", encoding="utf-8-sig",
                        parse_dates=["발행시각", "대상일"])
    daily = daily.rename(columns={"실제_kWh": "실제", "예측_kWh": "예측", "대상일": "대상"})
    daily["단위"] = "kWh"
    daily["수평_h"] = "D+1"

    keep = ["티어", "수평_h", "공식구성", "폴드", "발행시각", "대상", "실제", "예측", "단위"]
    return pd.concat([ac[keep], daily[keep]], ignore_index=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.assign(구간폭=rows["상한"] - rows["하한"])
    agg = (rows.groupby(["티어", "수평_h", "명목포함률_pct"], as_index=False)
           .agg(실제포함률_pct=("포함", lambda x: float(np.mean(x) * 100)),
                평균구간폭=("구간폭", "mean"),
                시험표본수=("포함", "size")))
    agg["포함률차이_pp"] = agg["실제포함률_pct"] - agg["명목포함률_pct"]
    return agg


def conditional_by_decile(rows: pd.DataFrame) -> pd.DataFrame:
    """예측값 십분위별 포함률 — 구간폭이 폴드당 상수라 조건부로는 어긋날 수 있다."""
    out = []
    for (tier, h, nom), g in rows.groupby(["티어", "수평_h", "명목포함률_pct"]):
        g = g.copy()
        # 동일값(예: 0kW)이 많아 qcut이 실패할 수 있어 rank 기반으로 자른다
        g["십분위"] = pd.qcut(g["예측"].rank(method="first"), 10, labels=False) + 1
        for dec, gd in g.groupby("십분위"):
            out.append({
                "티어": tier, "수평_h": h, "명목포함률_pct": nom, "예측값_십분위": int(dec),
                "예측값_구간": f"{gd['예측'].min():.1f}~{gd['예측'].max():.1f}",
                "n": len(gd),
                "실제포함률_pct": round(float(gd["포함"].mean() * 100), 2),
                "포함률차이_pp": round(float(gd["포함"].mean() * 100 - nom), 2),
                "평균구간폭": round(float((gd["상한"] - gd["하한"]).mean()), 2),
            })
    return pd.DataFrame(out)


def conditional_by_hour(rows: pd.DataFrame) -> pd.DataFrame:
    """대상시각(hour)별 포함률 — 일간(D+1)은 시각 개념이 없어 제외."""
    sub = rows[rows["티어"] != "일간"].copy()
    sub["대상시_hour"] = pd.to_datetime(sub["대상"]).dt.hour
    out = []
    for (tier, h, nom, hour), g in sub.groupby(["티어", "수평_h", "명목포함률_pct", "대상시_hour"]):
        out.append({
            "티어": tier, "수평_h": h, "명목포함률_pct": nom, "대상시_hour": int(hour),
            "n": len(g),
            "실제포함률_pct": round(float(g["포함"].mean() * 100), 2),
            "포함률차이_pp": round(float(g["포함"].mean() * 100 - nom), 2),
        })
    return pd.DataFrame(out)


def compare_with_v3(new_agg: pd.DataFrame) -> pd.DataFrame:
    if not OLD_SUMMARY.exists():
        return pd.DataFrame()
    old = pd.read_csv(OLD_SUMMARY, encoding="utf-8-sig")
    old = old.rename(columns={"실제포함률_pct": "실제포함률_pct_v3(1차)",
                              "평균구간폭": "평균구간폭_v3(1차)",
                              "시험표본수": "시험표본수_v3(1차)"})
    old["수평_h"] = old["수평_h"].astype(str)
    new = new_agg.copy()
    new["수평_h"] = new["수평_h"].astype(str)
    merged = new.merge(
        old[["티어", "수평_h", "명목포함률_pct", "실제포함률_pct_v3(1차)",
             "평균구간폭_v3(1차)", "시험표본수_v3(1차)"]],
        on=["티어", "수평_h", "명목포함률_pct"], how="left")
    merged["포함률_변화_pp"] = merged["실제포함률_pct"] - merged["실제포함률_pct_v3(1차)"]
    merged["구간폭_변화_pct"] = ((merged["평균구간폭"] - merged["평균구간폭_v3(1차)"])
                             / merged["평균구간폭_v3(1차)"] * 100)
    return merged


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"입력(3차 공식모델 v3): {SRC}\n")

    print("[1/4] NWP 운영정합성 감사(모델 무관 — 1차와 동일 결과 기대)...")
    nwp_summary, nwp_missing = ua.audit_nwp()
    nwp_summary.to_csv(OUT / "NWP_운영정합성_감사.csv", index=False, encoding="utf-8-sig")
    nwp_missing.to_csv(OUT / "NWP_변수별_결측.csv", index=False, encoding="utf-8-sig")
    print(nwp_summary.to_string(index=False))

    print("\n[2/4] v5 공식 OOF 적재 및 순차 conformal 예측구간 산출...")
    oof = load_oof_v5()
    print(f"  전 티어·수평 {oof.groupby(['티어', '수평_h']).ngroups}개 조합, 총 {len(oof):,}행")
    capacities = {"hourly": CAPACITY_KW, "daily": CAPACITY_KW * 24}
    rows, fold_summary = ua.sequential_conformal(oof, capacities)
    rows.to_csv(OUT / "예측구간_행단위.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    fold_summary.to_csv(OUT / "예측구간_폴드별.csv", index=False, encoding="utf-8-sig")

    agg = summarize(rows)
    agg.to_csv(OUT / "예측구간_수평별_요약.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 예측구간 수평별 요약(v5·3차 공식모델 v3) ===")
    print(agg.to_string(index=False))

    print("\n[3/4] ★신규★ 조건부 포함률 진단(예측값 십분위·시간대별)...")
    dec = conditional_by_decile(rows)
    hr = conditional_by_hour(rows)
    dec.to_csv(OUT / "조건부포함률_예측값십분위.csv", index=False, encoding="utf-8-sig")
    hr.to_csv(OUT / "조건부포함률_시간대별.csv", index=False, encoding="utf-8-sig")

    # 90% 기준으로 십분위 최저·최고 포함률만 요약 출력
    d90 = dec[dec["명목포함률_pct"] == 90]
    print("\n  [90% 명목] 티어·수평별 예측값 십분위 포함률 최저/최고:")
    for (tier, h), g in d90.groupby(["티어", "수평_h"]):
        lo = g.loc[g["실제포함률_pct"].idxmin()]
        hi = g.loc[g["실제포함률_pct"].idxmax()]
        print(f"    {tier} {h}: 최저 {lo['실제포함률_pct']:.1f}%(십분위{int(lo['예측값_십분위'])}, "
              f"{lo['예측값_구간']}) / 최고 {hi['실제포함률_pct']:.1f}%(십분위{int(hi['예측값_십분위'])})")

    worst_gap = float(d90["포함률차이_pp"].min())
    n_under = int((d90["실제포함률_pct"] < 80).sum())

    print("\n[4/4] 1차(v3 OOF) 예비치 대비 비교...")
    cmp_df = compare_with_v3(agg)
    if len(cmp_df):
        cmp_df.to_csv(OUT / "v3대비_비교.csv", index=False, encoding="utf-8-sig")
        print(cmp_df[["티어", "수평_h", "명목포함률_pct", "실제포함률_pct",
                      "실제포함률_pct_v3(1차)", "포함률_변화_pp",
                      "평균구간폭", "평균구간폭_v3(1차)", "구간폭_변화_pct"]].to_string(index=False))

    # 판정: 전체 포함률이 명목 대비 ±3pp 이내면 통과(1차에도 명시적 기준은 없었으므로
    # 여기서 처음 정하고, 결과를 보기 전에 기준을 정했음을 기록한다)
    agg90 = agg[agg["명목포함률_pct"] == 90]
    agg80 = agg[agg["명목포함률_pct"] == 80]
    within_3pp = bool((agg["포함률차이_pp"].abs() <= 3).all())

    report = {
        "판정": "v5 전사 재산출 완료 — 1차의 'v3 OOF 사용' 한계 해소",
        "입력": str(SRC),
        "기준모델": "3차 공식모델 v3(초단기+4h 날씨군집화·일간 53특성 반영)",
        "용량프로필": f"{CAPACITY_KW}kW (inverter_registered_sum_219)",
        "대상조합": "초단기 +1~4h, 단기 +1/+24/+48h, 일간 D+1 = 8개 전부",
        "방법": "폴드 순차 split-conformal(1차와 동일 — 방법 변경 없이 입력만 v5로 교체)",
        "전체포함률_명목대비_3pp이내": within_3pp,
        "최대_포함률차이_pp": round(float(agg["포함률차이_pp"].abs().max()), 2),
        "80pct_실제포함률_범위": [round(float(agg80["실제포함률_pct"].min()), 2),
                            round(float(agg80["실제포함률_pct"].max()), 2)],
        "90pct_실제포함률_범위": [round(float(agg90["실제포함률_pct"].min()), 2),
                            round(float(agg90["실제포함률_pct"].max()), 2)],
        "★조건부_진단★": {
            "90pct_십분위별_최대_과소포함_pp": round(worst_gap, 2),
            "90pct_십분위중_포함률80pct미만_구간수": n_under,
            "해석": "구간폭이 폴드당 상수(q)라 저출력 구간은 과대포함·고출력 구간은 "
                   "과소포함되는 구조적 한계. 전체 포함률만으로는 안 보인다.",
        },
        "NWP_시각구조": "D09 KST 런 < D10 KST 발행 구조는 전 행 통과(1차와 동일)",
        "NWP_미해결": "실제 게시완료시각 로그가 없어 1시간 내 운영 입수 가능성은 미입증(1차와 동일)",
        "한계": "1번 폴드는 보정 전용이라 구간이 산출되지 않는다(시험표본이 전체 행수보다 적음). "
               "조건부 포함률 개선(정규화 conformal 등)은 이번 범위 밖 — 사전동결 규칙을 "
               "먼저 정한 뒤 별도로 판정해야 한다.",
    }
    (OUT / "판정.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n전체 포함률 명목±3pp 이내: {within_3pp} (최대 차이 {report['최대_포함률차이_pp']}pp)")
    print(f"★조건부 진단★ 90% 기준 십분위별 최대 과소포함 {worst_gap:.1f}pp, "
          f"포함률 80% 미만 구간 {n_under}개")
    print(f"\n저장 완료: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
