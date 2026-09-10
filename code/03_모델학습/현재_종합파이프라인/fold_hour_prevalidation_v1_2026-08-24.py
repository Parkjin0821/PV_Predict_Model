# -*- coding: utf-8 -*-
"""v5 기반 공식 B(결함구간 제외) 폴드별·시간대별 사전검증.

모델을 학습하기 전에 시험창의 날짜/시간대/타깃/인버터 가용성/입력특성
완결성을 감사한다. 초단기 +1~+4h와 단기 +1/+24/+48h를 동일한 연속
6폴드에서 점검한다. 결함구간은 결과에서 숨기지 않고 명시적 평가불가로
남기며, 공식 정책 B는 학습행에서 대상시각 완전가용(5대)만 허용한다.
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
OUT = ROOT / "outputs" / "폴드시간대_사전검증_v1_2026-08-24"
N_INVERTERS = 5
MIN_TEST_ROWS = 30
MIN_COVERAGE_PCT = 80.0
NOON_HOURS = set(range(10, 16))
DIF = "DIFSWRF_bsrn정제"
MASK = "DIFSWRF_결측여부"
AUDIT_WINDOWS = dpc.FOLD_WINDOWS if "dpc" in globals() else []


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dpc = _load("precheck_dpc", ROOT / "defect_policy_comparison_v1_2026-08-21.py")
harness = dpc.harness
ultra = dpc.ultra
AUDIT_WINDOWS = dpc.FOLD_WINDOWS
OFFICIAL_B_WINDOWS = [
    ("1_여름", "2025-06-01", "2025-08-14"),
    # 인버터5 결함은 11-16까지이므로 기존 10-21 시작 가을창을 그대로 쓰면 안 된다.
    ("2_가을_결함종료후", "2025-11-17", "2025-12-14"),
    ("3_겨울", "2025-12-15", "2026-02-14"),
    ("4_봄", "2026-02-15", "2026-04-14"),
    ("5_초여름", "2026-04-15", "2026-08-04"),
]


def selected_features(frame: pd.DataFrame, tier: str, train: pd.DataFrame) -> list[str]:
    candidates = (ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
                  if tier == "초단기" else harness.FEATURE_SETS["전체후보"])
    # 공식 B: 결함/부분가용 타깃은 학습에서 제외한다.
    train_b = train[train["_목표_가용인버터수"] >= N_INVERTERS].copy()
    if len(train_b) < 300:
        return []
    return harness.select_features_in_fold(train_b, candidates, 0.3, True, True)


def audit_one(tier: str, horizon: int, frame: pd.DataFrame, scheme: str, windows):
    daylight = frame[frame["목표_낮시간"] > 0].copy()
    if tier == "초단기" and DIF in daylight.columns:
        # 08-24 비교에서 채택한 C: 원값 NaN 유지 + 결측여부 표시.
        daylight[MASK] = daylight[DIF].isna().astype("int8")
    fold_rows, hour_rows, missing_rows = [], [], []
    for fold, s, e in windows:
        start = pd.Timestamp(s)
        end = pd.Timestamp(e) + pd.Timedelta(days=1)
        train = daylight[daylight.index < start]
        test = daylight[(daylight.index >= start) & (daylight.index < end)].copy()
        chosen = selected_features(frame, tier, train)
        native_ok = set(harness.NATIVE_MISSING_OK)
        # 08-25 단기 연결 재감사 후 공통화: DIFSWRF C전략은 초단기뿐
        # 아니라 단기 실행 경로에도 적용한다. 현재 단기 공식 폴드에서는
        # 원본 DIF가 0/15회 선택되어 수치 영향은 없지만, 향후 선택될 때
        # 이 사전검증만 결측행을 다시 삭제하는 불일치를 막는다.
        native_ok.add(DIF)
        required = [c for c in chosen if c not in native_ok]

        full = test["_목표_가용인버터수"] >= N_INVERTERS
        target_ok = test["목표_발전출력_kW"].notna()
        feature_ok = (test[required].notna().all(axis=1) if required
                      else pd.Series(True, index=test.index))
        ready = full & target_ok & feature_ok

        offset = pd.Timedelta(hours=horizon if tier == "초단기" else horizon - 1)
        target_time = test.index + offset
        hours = pd.Series(target_time.hour, index=test.index)
        target_dates = pd.Series(target_time.normalize(), index=test.index)
        noon = hours.isin(NOON_HOURS)

        n = len(test)
        n_ready = int(ready.sum())
        coverage = 100.0 * n_ready / n if n else 0.0
        explicit_failure = (n_ready < MIN_TEST_ROWS or coverage < MIN_COVERAGE_PCT)
        calendar_days = (pd.Timestamp(e) - pd.Timestamp(s)).days + 1
        ready_dates = int(target_dates[ready].nunique())
        fold_rows.append({
            "검증구성": scheme, "티어": tier, "수평_h": horizon, "폴드": fold,
            "시험기간_시작": s, "시험기간_종료": e,
            "달력일수": calendar_days, "평가날짜수": ready_dates,
            "날짜_커버리지_pct": round(100.0 * ready_dates / calendar_days, 2),
            "전체_낮시간행": n,
            "타깃존재행": int(target_ok.sum()),
            "완전가용_5대행": int(full.sum()),
            "모델준비완료행": n_ready,
            "모델준비_커버리지_pct": round(coverage, 2),
            "정오대_전체행": int(noon.sum()),
            "정오대_준비완료행": int((ready & noon).sum()),
            "정오대_결측률_pct": round(100.0 * (1 - (ready & noon).sum() / noon.sum()), 2) if noon.sum() else np.nan,
            "선택특성수": len(chosen),
            "선택특성": "|".join(chosen),
            "평가가능": not explicit_failure,
            "실패사유": ("시험행 30건 미만" if n_ready < MIN_TEST_ROWS else
                       "커버리지 80% 미만" if coverage < MIN_COVERAGE_PCT else ""),
            "추정타깃포함": False,
            "공식정책": "B_구간제외",
            "DIFSWRF처리": "NaN자체처리+결측표시" if tier == "초단기" else "기존규칙",
        })

        for hour in sorted(hours.unique()):
            mask = hours == hour
            hour_rows.append({
                "검증구성": scheme, "티어": tier, "수평_h": horizon, "폴드": fold, "대상시_hour": int(hour),
                "전체행": int(mask.sum()), "완전가용행": int((full & mask).sum()),
                "모델준비완료행": int((ready & mask).sum()),
                "모델준비_커버리지_pct": round(100.0 * (ready & mask).sum() / mask.sum(), 2),
            })

        for col in chosen:
            missing_rows.append({
                "검증구성": scheme, "티어": tier, "수평_h": horizon, "폴드": fold, "특성": col,
                "시험결측률_pct": round(100.0 * test[col].isna().mean(), 2) if n else np.nan,
                "native_missing허용": col in native_ok,
            })
    return fold_rows, hour_rows, missing_rows


def main() -> None:
    dpc.require_v5_outputs()
    OUT.mkdir(parents=True, exist_ok=True)
    folds, hours, missing = [], [], []

    for scheme, windows in (("감사용_연속6폴드", AUDIT_WINDOWS),
                            ("공식B_정상구간5폴드", OFFICIAL_B_WINDOWS)):
        for h in (1, 2, 3, 4):
            a, b, c = audit_one("초단기", h, dpc.load_ultra_frame(h), scheme, windows)
            folds += a; hours += b; missing += c
            print(f"[{scheme}] 초단기 +{h}h 사전검증 완료")
        for h in (1, 24, 48):
            a, b, c = audit_one("단기", h, dpc.load_short_frame(h), scheme, windows)
            folds += a; hours += b; missing += c
            print(f"[{scheme}] 단기 +{h}h 사전검증 완료")

    fold_df = pd.DataFrame(folds)
    hour_df = pd.DataFrame(hours)
    miss_df = pd.DataFrame(missing)

    imbalance = []
    for (scheme, tier, h), g in fold_df.groupby(["검증구성", "티어", "수평_h"]):
        positive = g.loc[g["모델준비완료행"] > 0, "모델준비완료행"]
        imbalance.append({
            "검증구성": scheme, "티어": tier, "수평_h": h,
            "최소_양수_시험행": int(positive.min()) if len(positive) else 0,
            "최대_시험행": int(positive.max()) if len(positive) else 0,
            "최대최소_표본비": round(float(positive.max() / positive.min()), 2) if len(positive) else np.nan,
            "평가불가_폴드수": int((~g["평가가능"]).sum()),
        })
    imbalance_df = pd.DataFrame(imbalance)

    fold_df.to_csv(OUT / "폴드별_사전검증.csv", index=False, encoding="utf-8-sig")
    hour_df.to_csv(OUT / "시간대별_사전검증.csv", index=False, encoding="utf-8-sig")
    miss_df.to_csv(OUT / "선택특성_결측률.csv", index=False, encoding="utf-8-sig")
    imbalance_df.to_csv(OUT / "폴드표본_불균형.csv", index=False, encoding="utf-8-sig")

    failed = fold_df[~fold_df["평가가능"]]
    official = fold_df[fold_df["검증구성"] == "공식B_정상구간5폴드"]
    official_failed = official[~official["평가가능"]]
    summary = [
        "폴드별·시간대별 사전검증(v5, 공식 B_구간제외)",
        f"전체 점검 조합: {len(fold_df)}개(감사용 6폴드 42 + 공식B 정상구간 5폴드 35)",
        f"감사용 포함 평가불가 조합: {len(failed)}개",
        f"공식B 정상구간 평가불가 조합: {len(official_failed)}개",
        "추정타깃 포함: 0건",
        "",
        "[평가불가 목록]",
        failed[["검증구성", "티어", "수평_h", "폴드", "모델준비완료행", "모델준비_커버리지_pct", "실패사유"]].to_string(index=False),
        "",
        "[폴드 불균형]",
        imbalance_df.to_string(index=False),
    ]
    (OUT / "요약.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n" + "\n".join(summary))
    print(f"\n산출물: {OUT}")


if __name__ == "__main__":
    main()
