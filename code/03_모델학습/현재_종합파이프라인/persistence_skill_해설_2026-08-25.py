# -*- coding: utf-8 -*-
"""전 티어 "단순지속성 대비 얼마나 나은가"(Skill Score) 산출 — 성능표 해설용.

## 왜 필요한가
지금까지 나온 표는 MAE/RMSE/nMAE% 숫자만 있고 "이게 좋은 건지 나쁜 건지"에
대한 기준점이 없다. 여기서는 각 티어·수평에서 "그냥 마지막 실측값을 그대로
쓴다"는 가장 단순한 지속성 예측과 공식모델을 **정확히 같은 시험행**에서
비교해 Skill = 1 − RMSE_모델/RMSE_지속성 을 낸다(KPX 재현에서 이미 쓴 정의와
동일). 새로 학습하지 않고 이미 나온 공식 예측·저장된 지속성 특성만 쓴다.

## 재사용
- 초단기·단기 공식 예측: `E2E_v5_공식B_v2_⑥반영_2026-08-24/행단위_ac_power_
  예측정답.csv`(이미 최종).
- 지속성 실측: `defect_policy_comparison_v1_2026-08-21.load_ultra_frame/
  load_short_frame`가 이미 계산해둔 `_지속성_직전출력_kW`(발행시각 시점의
  마지막 실측값)를 그대로 조인만 한다.
- 일간 공식 예측: `일간_직접모델_최종감사_v1_2026-08-25/최종_직접모델_
  OOF.csv`. 지속성은 프로젝트 표준 일간 기준선인 "7일전 실적"
  (`daily_direct_final_audit_v1_2026-08-25`가 이미 만든 피처 프레임의
  `7일전_일간발전량_kWh`)을 그대로 쓴다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
AC_CSV = ROOT / "outputs" / "E2E_v5_공식B_v2_⑥반영_2026-08-24" / "행단위_ac_power_예측정답.csv"
DAILY_OOF = ROOT / "outputs" / "일간_직접모델_최종감사_v1_2026-08-25" / "최종_직접모델_OOF.csv"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("skill_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
dpc = e2e.dpc


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def ultra_short_skill() -> pd.DataFrame:
    ac = pd.read_csv(AC_CSV, encoding="utf-8-sig", parse_dates=["발행시각", "대상시각"])
    rows = []
    for tier, loader, horizons in [("초단기", dpc.load_ultra_frame, (1, 2, 3, 4)),
                                    ("단기", dpc.load_short_frame, (1, 24, 48))]:
        for h in horizons:
            frame = loader(h)
            pers = frame["_지속성_직전출력_kW"]
            sub = ac[(ac["티어"] == tier) & (ac["수평_h"] == h)].copy()
            sub["지속성_kW"] = pers.reindex(sub["발행시각"]).to_numpy()
            sub = sub.dropna(subset=["지속성_kW"])
            e_model = (sub["실제_kW"] - sub["예측_kW"]).to_numpy()
            e_pers = (sub["실제_kW"] - sub["지속성_kW"]).to_numpy()
            rows.append({
                "티어": tier, "수평_h": h, "n": len(sub),
                "모델_MAE": round(mae(e_model), 3), "모델_RMSE": round(rmse(e_model), 3),
                "지속성_MAE": round(mae(e_pers), 3), "지속성_RMSE": round(rmse(e_pers), 3),
                "Skill_MAE": round(1 - mae(e_model) / mae(e_pers), 4),
                "Skill_RMSE": round(1 - rmse(e_model) / rmse(e_pers), 4),
            })
    return pd.DataFrame(rows)


def daily_skill(capacity_kw: float) -> pd.DataFrame:
    oof = pd.read_csv(DAILY_OOF, encoding="utf-8-sig", parse_dates=["날짜"])
    daily_mod = _load("skill_daily", "daily_direct_final_audit_v1_2026-08-25.py")
    data, _features, _n_partial = daily_mod.corrected_dataset(capacity_kw)
    pers = data["7일전_일간발전량_kWh"]
    oof["지속성_kWh"] = pers.reindex(oof["날짜"]).to_numpy()
    before = len(oof)
    oof = oof.dropna(subset=["지속성_kWh"])
    e_model = (oof["실제_kWh"] - oof["예측_kWh"]).to_numpy()
    e_pers = (oof["실제_kWh"] - oof["지속성_kWh"]).to_numpy()
    return pd.DataFrame([{
        "티어": "일간", "수평_h": "D+1", "n": len(oof), "제외(지속성값없음)": before - len(oof),
        "모델_MAE": round(mae(e_model), 2), "모델_RMSE": round(rmse(e_model), 2),
        "지속성_MAE": round(mae(e_pers), 2), "지속성_RMSE": round(rmse(e_pers), 2),
        "Skill_MAE": round(1 - mae(e_model) / mae(e_pers), 4),
        "Skill_RMSE": round(1 - rmse(e_model) / rmse(e_pers), 4),
    }])


def main() -> None:
    import json
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    us = ultra_short_skill()
    dl = daily_skill(capacity_kw)
    print("=== 초단기·단기: 모델 vs 단순지속성 ===")
    print(us.to_string(index=False))
    print("\n=== 일간: 모델 vs 7일전 지속성 ===")
    print(dl.to_string(index=False))
    out = ROOT / "outputs" / "지속성대비_스킬점수_v5공식_2026-08-25"
    out.mkdir(parents=True, exist_ok=True)
    us.to_csv(out / "초단기_단기_스킬.csv", index=False, encoding="utf-8-sig")
    dl.to_csv(out / "일간_스킬.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
