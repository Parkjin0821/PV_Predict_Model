# -*- coding: utf-8 -*-
"""영광 중장기 - 월간 총발전량(kWh) 예측 후보(09-10, Claude).

광주 월간 개선후보(train_gwangju_monthly_improvement_candidate_v2_2026-09-10.py)와
동일 방법론 그대로 재사용 - GradientBoosting(huber) + expanding-window
walk-forward, 모든 특성은 shift(1) 이후 값만(발행시점 미래누출 없음).

영광은 "가용인버터수" 컬럼이 없어(일간 CSV엔 daylight_valid_ratio만
있음), 그 자리를 daylight_valid_ratio_lag1(전월 낮시간 유효비율)로
대체한다 - 같은 역할(설비 가동상태 프록시)이되 리크 없는 값.

기존 공식모델·스케줄러는 없음(영광 월간은 이번이 최초 시도) - 그대로
공식_후보로 저장, 운영 연결은 보류.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "영광_월간_후보_v1_2026-09-10"
DAILY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\시간집계_v1_2026-09-01"
    r"\영광_발전소_일간_공식후보.csv"
)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return {
        "n": len(y),
        "WAPE_pct": round(abs(e).sum() / abs(y).sum() * 100, 2),
        "MAE_kWh": round(abs(e).mean(), 1),
        "RMSE_kWh": round(np.sqrt((e ** 2).mean()), 1),
    }


def model_factories():
    return {
        "LightGBM_동일조건": lambda: LGBMRegressor(
            objective="regression_l1", n_estimators=80, learning_rate=.05,
            num_leaves=7, max_depth=3, min_child_samples=3, reg_lambda=1,
            verbosity=-1, random_state=20260819, n_jobs=2,
        ),
        "GradientBoostingHuber": lambda: GradientBoostingRegressor(
            loss="huber", n_estimators=50, max_depth=2, learning_rate=.04,
            min_samples_leaf=3, random_state=20260819,
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date_kst"]).set_index("date_kst")
    # 09-10 안전조건(광주와 동일): quality_status가 유효(valid_daylight_ge90pct)한
    # 날짜만 실제값으로 쓰고, 나머지는 결측 유지(보간 없음).
    daily.loc[daily["quality_status"] != "valid_daylight_ge90pct", "daily_energy_kwh"] = np.nan

    ycol = "daily_energy_kwh"
    ratio_col = "daylight_valid_ratio"
    m = daily[ycol].resample("MS").agg(total="sum", n="count")
    m["days"] = m.index.days_in_month
    m["y"] = m.total.where(m.n >= m.days)  # 완결월(하루도 안 빠진 달)만 유효
    m["ratio"] = daily[ratio_col].resample("MS").mean()

    idx = pd.date_range(m.index.min(), m.index.max(), freq="MS")
    m = m.reindex(idx)
    f = pd.DataFrame(index=idx)
    for k in [1, 2, 3]:
        f[f"lag{k}"] = m.y.shift(k)
    f["mean3"] = m.y.shift(1).rolling(3, min_periods=3).mean()
    f["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    f["days"] = idx.days_in_month
    f["daylight_ratio_lag1"] = m.ratio.shift(1)  # 광주의 availability_lag1과 동일 역할
    f["y"] = m.y

    features = ["lag1", "lag2", "lag3", "mean3", "month_sin", "month_cos", "days", "daylight_ratio_lag1"]
    v = f.dropna(subset=features + ["y"])
    print(f"완결월 수: {int(m.y.notna().sum())}, 특성 결측 없는 학습가능 월: {len(v)}, "
          f"기간: {v.index.min().date()}~{v.index.max().date()}")

    if len(v) < 10:
        print("표본이 너무 적어(10개월 미만) walk-forward를 시작할 수 없음 - 중단")
        return

    initial = max(6, len(v) // 3)
    rows = []
    factories = model_factories()
    for i in range(initial, len(v)):
        tr, te = v.iloc[:i], v.iloc[i:i + 1]
        r = {"month": str(te.index[0].date()), "actual_kWh": float(te.y.iloc[0]),
             "전월지속성_kWh": float(te.lag1.iloc[0])}
        for name, mk in factories.items():
            mdl = mk()
            mdl.fit(tr[features], tr.y)
            r[name + "_kWh"] = max(0.0, float(mdl.predict(te[features])[0]))
        rows.append(r)

    o = pd.DataFrame(rows)
    o.to_csv(OUT / "월간_OOF_상세.csv", index=False, encoding="utf-8-sig")

    result = {"전월지속성": metrics(o.actual_kWh, o.전월지속성_kWh)}
    for name in factories:
        result[name] = metrics(o.actual_kWh, o[name + "_kWh"])
    pd.DataFrame([{"구성": k, **vv} for k, vv in result.items()]).to_csv(
        OUT / "월간_성능요약.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    best_name = min(
        (n for n in factories), key=lambda n: result[n]["MAE_kWh"]
    )
    final = factories[best_name]()
    final.fit(v[features], v.y)
    joblib.dump({"model": final, "features": features, "model_name": best_name}, OUT / "model.joblib")

    manifest = {
        "status": "공식_후보",
        "region": "영광", "tier": "중장기", "horizon": "월간",
        "기존모델_스케줄러_변경": False,
        "source": str(DAILY_CSV),
        "complete_months": int(m.y.notna().sum()),
        "latest_complete_month": str(m.y.dropna().index.max().date()),
        "walk_forward_common_n": len(o),
        "issue_safe": "모든 lag·이동평균·낮시간유효비율은 전월 이하 값만 사용 - 미래누출 없음",
        "metrics": result,
        "selected_model": best_name,
        "운영_연결": "보류; 기존 공식모델 없음(영광 월간 최초 시도)",
        "created_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
