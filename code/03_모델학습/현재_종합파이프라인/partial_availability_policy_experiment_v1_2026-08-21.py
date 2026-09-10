# -*- coding: utf-8 -*-
"""부분가용(인버터 <5대) 구간 처리방침 실험 (사용자 요청, v5 복구 데이터 전제).

## 배경
v5 복구로 부분가용 5분버킷이 전체의 6.57%나 있다는 게 드러났다(인버터5
94일 침묵 등). 아무 처리 없이 그대로 학습에 넣으면, 같은 일사량에서도
총출력이 구조적으로 낮은 구간을 모델이 "이 시기엔 원래 덜 나온다"로
잘못 배울 위험이 있다. v5 스크립트 docstring에서 예고한 대로, 처리
방침은 **①제외 ②가용대수 비례 정규화(추정) ③특성으로 투입** 중
무엇을 쓸지 실험으로 정한다.

## 평가 설계(공정 비교의 핵심)
네 정책 모두 **정확히 같은 시험행**에서 채점해야 비교가 성립한다.
그런데 부분가용 시각의 실측값 자체가 "무엇을 맞혀야 하는지"부터
모호하다(전체 5대 출력인지 4대 부분합인지). 그래서:
- **시험(test)은 항상 대상시각(미래) 가용인버터수=5인 행만** 쓴다 —
  유일하게 모호함이 없는 정답이다.
- **학습(train) 데이터 처리만 정책별로 다르게** 한다. 이러면 "같은
  질문(완전가용 미래를 맞히기)에 학습 방법만 바꿔서" 비교가 된다.

## 네 정책
- **A_그대로(기준)**: 부분가용 행도 무처리로 그대로 포함(지금까지의
  암묵적 상태 — 이게 얼마나 나쁜지의 기준선).
- **B_제외**: 대상시각 가용인버터수<5인 학습행을 뺀다.
- **C_비례정규화(추정)**: 부분가용 행의 타깃을 `실제_kW × 5/가용인버터수`로
  스케일업해 포함한다. ★이건 "정답을 만드는 게" 아니라 **학습용 추정
  라벨 실험**이다 — recover 스크립트가 데이터 자체에는 절대 하지 않기로
  한 스케일업을, 여기서는 모델링 실험으로 한정해 시도한다(정답 데이터
  파일은 건드리지 않음, 이 스크립트 안에서만 임시로 씀).
- **D_특성투입**: 타깃은 원본 그대로 두고, **발행시점(과거) 가용인버터수**
  를 입력 후보 특성에 추가해 다른 후보와 동일하게 폴드 내부 선택을
  거치게 한다(미래 가용대수는 절대 특성으로 안 씀 — 발행 이후 정보라
  누출이다).

## 데이터 출처(★v5로 교체★)
- 발전량 계열(15분): `outputs/v5_복구_2026-08-21/집계_15분_자료_v5.parquet`
  (v3·v4 아님 — 둘 다 폐기됨, AGENTS.md 참고)
- 기상·NWP(시간단위): 기존 `harness.DATASETS["v3_고정tm"]` 그대로(인버터
  문제와 무관해 영향 없음)
- 특성 파이프라인은 `train_ultra_short_official_v1_2026-08-21.py`의
  `build_ultra_short_frame`을 그대로 재사용(재구현 안 함), 5분 소스만
  v5로 바꿔치기한다.

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\03_모델학습\\현재_종합파이프라인"
python partial_availability_policy_experiment_v1_2026-08-21.py
```
로컬 재학습만(API 없음). 5계절 폴드 × 4정책 × 2수평(+1h·+4h)이라
수 분~수십 분 소요될 수 있다.

## 산출물 (`outputs/부분가용_정책실험_v1_2026-08-21/`)
- `판정표.csv` — 정책×수평별 MAE/RMSE/nRMSE(완전가용 시험행 기준)
- `폴드별상세.csv`
- `요약.txt`
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
OUT = ROOT / "outputs" / "부분가용_정책실험_v1_2026-08-21"
V5_15MIN = ROOT / "outputs" / "v5_복구_2026-08-21" / "집계_15분_자료_v5.parquet"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("papol_harness", "backtest_harness_v1_2026-08-20밤.py")
ultra = _load("papol_ultra", "train_ultra_short_official_v1_2026-08-21.py")

HORIZONS = [1, 4]  # 가장 짧은/긴 초단기 수평으로 대표 확인(전수는 아님 — 신호 확인용)
N_INVERTERS = 5
POLICIES = ["A_그대로(기준)", "B_제외", "C_비례정규화(추정)", "D_특성투입"]


def load_v5_quarter() -> pd.DataFrame:
    """v5 15분 자료를 ultra.load_15min_base()와 동일한 컬럼명으로 맞춘다."""
    if not V5_15MIN.exists():
        raise FileNotFoundError(
            f"v5 복구 산출물이 없다: {V5_15MIN}\n"
            "먼저 02_전처리/rebuild_plant_v5_recovered_2026-08-21.py를 실행할 것."
        )
    df = pd.read_parquet(V5_15MIN)
    df = df.rename(columns=ultra.RENAME_15MIN)
    return df


def build_frame_with_availability(quarter: pd.DataFrame, hourly: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """기존 build_ultra_short_frame 그대로 재사용 + 가용인버터수 두 종류 추가.
    (재구현 금지 원칙 — 특성·타깃 프레임 자체는 공식 함수를 그대로 씀)"""
    frame = ultra.build_ultra_short_frame(quarter, hourly, horizon)
    # 발행시점(과거) 가용대수 — 정책D 후보특성. shift(1)=직전 완결 15분(다른
    # 장비특성과 동일 규약, EQUIP_STARTLABEL과 정렬 일치).
    frame["가용인버터수_발행시점"] = quarter["가용인버터수"].shift(1)
    # 대상시점(미래) 가용대수 — ★특성으로는 안 씀★, 정책별 학습행 처리와
    # 평가용 "완전가용 시험행" 필터링에만 쓴다.
    frame["_목표_가용인버터수"] = quarter["가용인버터수"].shift(-horizon * 4)
    return frame


def make_policy_train(daylight: pd.DataFrame, policy: str, capacity_kw: float) -> tuple[pd.DataFrame, str]:
    """정책별로 학습 프레임(및 그 정책에서 쓸 타깃 컬럼명)을 만든다."""
    df = daylight.copy()
    if policy == "A_그대로(기준)":
        df["_학습타깃_kW"] = df["목표_발전출력_kW"]
        return df, "무처리(부분가용 행도 원본 그대로 포함)"
    if policy == "B_제외":
        df = df[df["_목표_가용인버터수"] >= N_INVERTERS].copy()
        df["_학습타깃_kW"] = df["목표_발전출력_kW"]
        return df, "대상시각 가용<5인 학습행 제거"
    if policy == "C_비례정규화(추정)":
        df = df.copy()
        ratio = N_INVERTERS / df["_목표_가용인버터수"].clip(lower=1)
        scaled = (df["목표_발전출력_kW"] * ratio).clip(upper=capacity_kw)
        df["_학습타깃_kW"] = np.where(df["_목표_가용인버터수"] >= N_INVERTERS,
                                    df["목표_발전출력_kW"], scaled)
        return df, "부분가용 타깃을 5/가용대수 배로 스케일업(추정 라벨, 정답데이터 비변경)"
    if policy == "D_특성투입":
        df["_학습타깃_kW"] = df["목표_발전출력_kW"]
        return df, "타깃 원본 유지 + 발행시점 가용인버터수를 후보특성에 추가"
    raise ValueError(policy)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    quarter = load_v5_quarter()
    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                         parse_dates=["time"], low_memory=False).set_index("time").sort_index()

    print(f"v5 15분 자료 로드: {len(quarter):,}행, 부분가용 5분버킷 비율 "
          f"{(quarter['가용인버터수'] < N_INVERTERS).mean()*100:.2f}%")

    candidate_cols = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS

    rows = []
    for horizon in HORIZONS:
        frame = build_frame_with_availability(quarter, hourly, horizon)
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간", "가용인버터수_발행시점")]
        daylight = frame[frame["목표_낮시간"] > 0]

        for i, window in enumerate(config["cross_validation_windows"], start=1):
            start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
            fold = f"{i}_{window.get('_계절', '')}"
            train_all = daylight[daylight.index < start]
            test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
            # ★평가는 항상 완전가용 미래시점만★(정책과 무관하게 고정)
            test_full = test_all[test_all["_목표_가용인버터수"] >= N_INVERTERS]
            if len(train_all) < 500 or len(test_full) < 50:
                print(f"  [+{horizon}h {fold}] 표본부족 — 건너뜀")
                continue

            for policy in POLICIES:
                train_variant, policy_desc = make_policy_train(train_all, policy, capacity_kw)
                use_extra_feature = policy == "D_특성투입"
                # 정책D만 '가용인버터수_발행시점'을 후보에 넣는다 — 상관계수·
                # 다중공선성·배포필터를 그대로 통과해야만 실제로 쓰인다
                # (무조건 채택하지 않음, 다른 후보와 동일 대우).
                feature_pool = candidate_cols + (["가용인버터수_발행시점"] if use_extra_feature else [])

                chosen = harness.select_features_in_fold(
                    train_variant.dropna(subset=["_학습타깃_kW"]), feature_pool, 0.3, True, True)
                features = base_cols + chosen

                required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + ["_학습타깃_kW"]
                train = train_variant.dropna(subset=required)
                test = test_full.dropna(subset=[c for c in features if c not in harness.NATIVE_MISSING_OK]
                                        + ["목표_발전출력_kW"])
                if len(train) < 500 or len(test) < 50:
                    print(f"  [+{horizon}h {fold} {policy}] 결측제거 후 표본부족 — 건너뜀")
                    continue

                model = ultra.make_model("LightGBM", seed)
                model.fit(train[features], train["_학습타깃_kW"])
                pred = np.clip(model.predict(test[features]), 0, capacity_kw)
                y = test["목표_발전출력_kW"].to_numpy()
                mae = float(np.mean(np.abs(y - pred)))
                rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
                rows.append({
                    "수평_h": horizon, "폴드": fold, "정책": policy, "정책설명": policy_desc,
                    "학습행수": len(train), "학습행수_부분가용포함": int((train_variant["_목표_가용인버터수"] < N_INVERTERS).sum()),
                    "시험행수(완전가용만)": len(test),
                    "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
                    "nMAE_pct": round(mae / capacity_kw * 100, 3),
                    "nRMSE_pct": round(rmse / capacity_kw * 100, 3),
                    "가용인버터수_선택됨(정책D)": bool(use_extra_feature and "가용인버터수_발행시점" in chosen),
                })
            print(f"  [+{horizon}h {fold}] 4개 정책 완료 (시험행 {len(test_full):,})")

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "폴드별상세.csv", index=False, encoding="utf-8-sig")

    agg = detail.groupby(["수평_h", "정책", "정책설명"], as_index=False).agg(
        폴드수=("폴드", "nunique"),
        전체시험행수=("시험행수(완전가용만)", "sum"),
        MAE_kW_가중평균=("MAE_kW", lambda s: np.average(s, weights=detail.loc[s.index, "시험행수(완전가용만)"])),
        RMSE_kW_가중평균=("RMSE_kW", lambda s: np.average(s, weights=detail.loc[s.index, "시험행수(완전가용만)"])),
    )
    agg["nMAE_pct"] = (agg["MAE_kW_가중평균"] / 240.0 * 100).round(3)
    agg["nRMSE_pct"] = (agg["RMSE_kW_가중평균"] / 240.0 * 100).round(3)
    agg = agg.sort_values(["수평_h", "RMSE_kW_가중평균"])
    agg.to_csv(OUT / "판정표.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 정책별 판정표(5계절 가중평균, 완전가용 시험행 기준) ===")
    print(agg.to_string(index=False))

    lines = ["부분가용 구간 처리방침 실험 — 요약", "",
             "평가는 항상 대상시각(미래) 가용인버터수=5인 행만(정책과 무관, 고정).",
             "학습 데이터 처리만 4가지로 비교: A_그대로(기준)/B_제외/C_비례정규화(추정)/D_특성투입.",
             "", agg.to_string(index=False)]
    (OUT / "요약.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
