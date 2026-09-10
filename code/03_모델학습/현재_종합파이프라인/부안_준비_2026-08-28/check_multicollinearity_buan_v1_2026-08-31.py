# -*- coding: utf-8 -*-
"""부안 총출력모델 특성 다중공선성+상관분석(08-31, 사용자 지시: "다중공선성,
상관분석 이후 각 요인들에 따라서 교차검증 진행 후 모델 선정").

광주 check_multicollinearity_v3_2026-08-20밤.py와 동일 방법론(VIF + 쌍상관
|r|>=0.8 & VIF>=10 동시충족 시 타깃상관 약한 쪽 제거)을 부안 자체 데이터
(GRID결합 v1_2026-08-31.parquet)로 독립 재현한다 - 광주 결과를 그대로
가져다 쓰지 않고 부안 데이터로 직접 상관을 구한다(이게 이번 지시의
핵심 - "부안 자체" 상관분석은 지금까지 한 번도 안 했었다).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
DATA = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\과거발전_기상결합_v1_2026-08-28"
    r"\부안_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet"
)
OUT_DIR = HERE / "outputs" / "부안_다중공선성_상관분석_2026-08-31"
VIF_SEVERE, PAIR_CORR_HIGH = 10.0, 0.8

CANDIDATE_FEATURES = [
    "forecast_DSWRF", "forecast_TCDC", "forecast_LCDC", "forecast_MCDC", "forecast_HCDC",
    "forecast_REH", "forecast_POP", "forecast_SKY",
    "issue_asos_기온_C", "issue_asos_풍속_m_s", "issue_asos_상대습도_pct", "issue_asos_전운량_pct",
    "solar_elevation_deg",
]
TARGET = "plant_ac_power_kw"


def compute_vif(x: pd.DataFrame) -> pd.Series:
    x = (x - x.mean()) / x.std(ddof=0)
    vifs = {}
    for col in x.columns:
        y = x[col].to_numpy()
        others = x.drop(columns=[col]).to_numpy()
        r2 = LinearRegression().fit(others, y).score(others, y)
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    df = df[df["quality_status"] == "valid_ge9of12"]
    df = df[df["physical_daylight"].astype(bool)]  # 광주와 동일하게 낮시간만 상관/VIF 계산
    x = df[CANDIDATE_FEATURES].dropna()
    y = df.loc[x.index, TARGET]
    print(f"낮시간·완전표본: {len(x)}행")

    target_corr = {c: float(x[c].corr(y)) for c in CANDIDATE_FEATURES}
    target_corr_sorted = dict(sorted(target_corr.items(), key=lambda kv: -abs(kv[1])))

    vif = compute_vif(x)

    corr_matrix = x.corr(method="pearson")
    high_pairs = []
    cols = corr_matrix.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr_matrix.loc[a, b]
            if abs(r) >= PAIR_CORR_HIGH:
                weaker = a if abs(target_corr[a]) < abs(target_corr[b]) else b
                high_pairs.append({"변수A": a, "변수B": b, "쌍상관계수": round(float(r), 3),
                                    "A_타깃상관": round(target_corr[a], 3),
                                    "B_타깃상관": round(target_corr[b], 3),
                                    "제거후보": weaker})

    drop_candidates = sorted({p["제거후보"] for p in high_pairs})
    severe_vif = set(vif[vif >= VIF_SEVERE].index)
    final_drop = [c for c in drop_candidates if c in severe_vif]
    pruned_features = [c for c in CANDIDATE_FEATURES if c not in final_drop]

    result = {
        "표본": {"낮시간_완전표본_행수": len(x)},
        "타깃상관계수_부안자체(내림차순)": {k: round(v, 3) for k, v in target_corr_sorted.items()},
        "VIF": {k: (None if np.isinf(v) else round(float(v), 2)) for k, v in vif.items()},
        "쌍상관_0.8이상": high_pairs,
        "제거후보(쌍상관>=0.8_AND_VIF>=10_동시충족)": final_drop,
        "최종_특성목록": pruned_features,
        "_비교": "광주 08-20 상관분석(select_features_by_correlation_threshold_v3)과 순위가 "
               "같은 방향인지 참고용으로 대조할 것 - 광주: DSWRF>REH>TCDC~LCDC>POP>SKY>MCDC>HCDC 순.",
    }

    (OUT_DIR / "부안_다중공선성_상관분석_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
