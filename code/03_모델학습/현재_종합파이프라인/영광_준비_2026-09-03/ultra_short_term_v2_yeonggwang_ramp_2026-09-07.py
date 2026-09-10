# -*- coding: utf-8 -*-
"""영광 초단기(+1h~+4h) v2(09-07) - 변화율(ramp) 특성 추가.

## 배경(★학술 근거 확인 후 착수★)
v1(`ultra_short_term_v1_yeonggwang_2026-09-07.py`) 사후분석에서 영광은
lag·태양기하 상관이 김제보다 전 구간 약하고 구름량 상관은 더 강함을
실측 확인(09-07). 이는 Chu, Li, Coimbra, Feng & Wang(2021, *iScience*
24(10), 103136) 리뷰의 "지속성 모델은 구름으로 변동성이 커지는 구간에서
정확도가 크게 떨어진다"는 결론과 일치 - **레벨값(lag) 대신 변화율(ramp)
특성을 추가하면 구름에 의한 급변을 더 빨리 잡을 수 있다**는 게 이
버전의 가설.

## v1과의 차이(이것 외 전부 동일 - 순수 특성 추가 실험)
1. `power_ramp_{15,30,60}min` - 최근 kW 변화율(kW/분). 태양고도 변화와
   섞인 원값 대신 "방향"을 직접 준다.
2. `kt_ramp_{30,60}min` - 청천지수(kt, 흐림 정도를 태양고도와 분리)의
   변화율. McCandless, Haupt & Young(2016, *J. Appl. Meteor. Climatol.*
   55(7), DOI:10.1175/JAMC-D-15-0354.1)가 쓰는 "청천지수 기반 상태
   구분" 아이디어를 회귀 특성으로 단순화해 반영(별도 regime 분류 모델은
   아직 안 만듦 - 그건 다음 단계).

데이터·폴드·구조튜닝·타깃(raw kW)·capacity(634kW)는 v1과 전부 동일 -
개선 효과를 특성 추가 하나로만 격리해서 보기 위함.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V1_SPEC = importlib.util.spec_from_file_location(
    "ultra_short_yg_v1", HERE / "ultra_short_term_v1_yeonggwang_2026-09-07.py")
V1 = importlib.util.module_from_spec(V1_SPEC)
V1_SPEC.loader.exec_module(V1)  # type: ignore

OUT_DIR = HERE / "outputs" / "영광_초단기_v2_ramp_2026-09-07"

RAMP_FEATURES = [
    "power_ramp_15min", "power_ramp_30min", "power_ramp_60min",
    "kt_ramp_30min", "kt_ramp_60min",
]
BASE_FEATURES_V2 = V1.BASE_FEATURES + RAMP_FEATURES


def add_ramp_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # 1) 원값(kW) 변화율 - kW/분
    d["power_ramp_15min"] = (d["power_lag_0min"] - d["power_lag_15min"]) / 15.0
    d["power_ramp_30min"] = (d["power_lag_0min"] - d["power_lag_30min"]) / 30.0
    d["power_ramp_60min"] = (d["power_lag_0min"] - d["power_lag_60min"]) / 60.0

    # 2) 청천지수(kt) 변화율 - 태양고도(시간대) 효과를 분리한 "구름만의" 추세
    for m in (30, 60):
        slots = m // 5
        elev_lag = d["solar_elevation_deg"].shift(slots)
        clearsky_lag = V1.CAPACITY_KW * V1.haurwitz_clearsky_ghi_wm2(elev_lag.to_numpy()) / 1000.0
        clearsky_lag = pd.Series(clearsky_lag, index=d.index)
        kt_lag_valid = clearsky_lag > 1.0
        kt_lag = np.where(kt_lag_valid, d[f"power_lag_{m}min"] / clearsky_lag, np.nan)
        kt_lag = np.clip(kt_lag, 0, 1.5)
        d[f"kt_ramp_{m}min"] = d["kt_now"] - kt_lag
    return d


def build_horizon_frame_v2(base: pd.DataFrame, lead_hours: int):
    d, target_col, _old_features = V1.build_horizon_frame(base, lead_hours)
    d = add_ramp_features(d)
    return d, target_col, BASE_FEATURES_V2


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, meta = V1.load_base()

    def improve(model_mae, base_mae):
        return round((1 - model_mae / base_mae) * 100, 1) if base_mae else None

    horizon_results = {}
    for h in V1.LEAD_HOURS:
        full_features = [f"power_lag_{m}min" for m in V1.LAG_MINUTES] + BASE_FEATURES_V2
        d, target_col, _ = build_horizon_frame_v2(base, h)
        days = pd.DatetimeIndex(np.sort(d["day"].unique()))
        folds = V1.expanding_folds_full_coverage(days, V1.INITIAL_TRAIN_DAYS, V1.TEST_BLOCK_DAYS)
        perf = V1.run_walkforward_tuned(d, target_col, full_features, folds)
        entry = {"리드타임": f"+{h}h", "일수_total": int(len(days)), "성능": perf}
        if perf.get("폴드수", 0) > 0:
            entry["성능"]["MAE_개선율_vs단순지속성_pct"] = improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_단순지속성"]["MAE_kW"])
            entry["성능"]["MAE_개선율_vs스마트지속성_pct"] = improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_스마트지속성"]["MAE_kW"])
            entry["nMAE_pct"] = round(perf["pooled_모델"]["MAE_kW"] / V1.CAPACITY_KW * 100, 3)
        horizon_results[f"+{h}h"] = entry
        print(f"[+{h}h] v2(ramp) 완료 - pooled MAE(모델)={perf.get('pooled_모델',{}).get('MAE_kW')}kW, "
              f"nMAE={entry.get('nMAE_pct')}%, 구조분포={perf.get('구조선택_투표분포')}")

    result = {
        **meta, "capacity_kw_사용값": V1.CAPACITY_KW,
        "특징": "v1 대비 ramp/kt변화율 특성 5개 추가(그 외 전부 v1과 동일) - 특성 추가 효과만 격리 측정",
        "추가특성목록": RAMP_FEATURES,
        "평가대상": "각 리드타임의 대상시각 태양고도>0인 행만",
        "리드타임별_결과": horizon_results,
        "_방법론출처": "ramp 특성: Chu et al.(2021, iScience) 리뷰의 지속성-구름변동성 한계 지적에 근거. "
                     "kt_ramp: McCandless, Haupt & Young(2016, JAMC) regime-dependent 아이디어를 "
                     "회귀특성으로 단순화(정식 regime 분류는 별도 단계).",
        "_판정": "잠정치 - v1과의 nMAE 비교로 특성 추가 효과 검증 목적. promote_to_official 대상 아님.",
    }
    (OUT_DIR / "영광_초단기_v2_ramp_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "리드타임별_결과"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
