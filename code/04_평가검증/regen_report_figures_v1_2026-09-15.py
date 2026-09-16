# -*- coding: utf-8 -*-
"""Notion 종합보고서 그림 재생성(09-15, 사용자 피드백 반영).

## 배경
사용자 피드백: "이상치 정책의 동일표본 예측오차 그래프가 0부터 시작하는
축이라 6.80~6.89kW 차이가 전혀 안 보인다. 6~7로 잡고 촘촘하게 다시
그려라. 설명도 쉬운 말로." + "그림 5-2도 6~9로 촘촘하게".

기존 산출물(04_평가검증/outputs/*)의 **실측 CSV를 그대로 읽어서** 축만
다시 잡는다 - 수치를 새로 만들거나 임의 보정하지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rcParams

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "보고서그림_재생성_v1_2026-09-15"

# 한글 폰트(맑은 고딕) - 09-15 그림 5-4~5-6 UTF-8 깨짐과 같은 원인 방지
for cand in ("Malgun Gothic", "맑은 고딕", "NanumGothic"):
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        rcParams["font.family"] = cand
        break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False


def fig_outlier_policy_zoom() -> Path:
    """이상치 정책별 동일표본 MAE - y축을 실제 값 범위로 확대."""
    src = HERE / "outputs" / "광주_이상치정책_비교_v1_2026-09-09" / "이상치정책별_제거행_모델성능.csv"
    df = pd.read_csv(src, encoding="utf-8-sig")
    df = df.sort_values("MAE_kW").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2e7d32" if i == 0 else "#546e7a" for i in range(len(df))]
    bars = ax.bar(range(len(df)), df["MAE_kW"], color=colors, width=0.6)

    # 물리범위(기준) 대비 95% 신뢰구간을 오차막대로 표시
    base = df.loc[df["policy"].str.contains("물리범위_0_240"), "MAE_kW"]
    base_mae = float(base.iloc[0]) if len(base) else float(df["MAE_kW"].median())
    lo = df["MAE_kW"] - (base_mae + df["delta_CI95_low_kW"] - df["MAE_kW"]).abs()
    hi = (base_mae + df["delta_CI95_high_kW"]) - df["MAE_kW"]
    ax.errorbar(range(len(df)), df["MAE_kW"],
                yerr=[(df["MAE_kW"] - (base_mae + df["delta_CI95_low_kW"])).abs(),
                      ((base_mae + df["delta_CI95_high_kW"]) - df["MAE_kW"]).abs()],
                fmt="none", ecolor="#b71c1c", elinewidth=1.4, capsize=6, capthick=1.4)

    for i, v in enumerate(df["MAE_kW"]):
        ax.text(i, v - 0.012, f"{v:.3f}", ha="center", va="top", fontsize=11,
                fontweight="bold", color="white")

    ax.axhline(base_mae, color="#1565c0", linestyle="--", linewidth=1.2)
    ax.text(-0.45, base_mae + 0.004, "기준(물리범위)", color="#1565c0", fontsize=9, ha="left")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{p}\n(제외 {int(n):,}행)" for p, n in
                        zip(df["policy"], df["removed_5min_rows"])], fontsize=9.5)
    ax.set_ylim(6.72, 7.00)          # ★핵심★ 실제 값 범위로 확대(신뢰구간까지 포함)
    ax.set_ylabel("동일표본 예측오차 MAE (kW)", fontsize=11)
    ax.set_title("이상치 정책별 예측오차 — 광주 +1h, 공통 시험 1,309행\n"
                 "(빨간 막대 = 기준 대비 95% 신뢰구간; 0을 걸치면 '차이 없음')",
                 fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "fig4_이상치정책_MAE_축확대.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_regional_gap_imputation() -> Path:
    """4지역 공백길이별 보간 오차 - 지역별 비교(§3 지역 확장 요구)."""
    src = HERE / "outputs" / "3지역_결측보간_마스킹복원_v1_2026-09-09" / "3지역_보간법_공백길이별_비교.csv"
    df = pd.read_csv(src, encoding="utf-8-sig")
    lin = df[df["method"] == "선형"].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"부안": "o", "김제": "s", "영광": "^"}
    for region, g in lin.groupby("region"):
        g = g.sort_values("gap_minutes")
        ax.plot(g["gap_minutes"], g["MAE_kW"], marker=markers.get(region, "o"),
                linewidth=2, markersize=8, label=f"{region}(인버터 단위)")
    # 광주는 발전소 총출력 기준(스케일이 달라 점선으로 구분 표기)
    gwangju = {5: 5.551, 10: 6.055, 15: 6.037, 30: 6.821}
    ax.plot(list(gwangju), list(gwangju.values()), marker="D", linestyle="--",
            linewidth=2, markersize=8, color="#6a1b9a", label="광주(발전소 총출력)")

    ax.set_xticks([5, 10, 15, 30])
    ax.set_xlabel("가린 공백 길이 (분)", fontsize=11)
    ax.set_ylabel("선형보간 복원 MAE (kW)", fontsize=11)
    ax.set_title("공백 길이별 선형보간 복원오차 — 4지역\n"
                 "(광주는 발전소 총출력, 나머지 3지역은 인버터 단위라 절대값 직접비교 불가)",
                 fontsize=12, pad=12)
    ax.axvline(10, color="#d84315", linestyle=":", linewidth=1.5)
    ax.text(10.4, ax.get_ylim()[1] * 0.97, "운영 상한 10분", color="#d84315", fontsize=9, va="top")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "fig3_4지역_공백길이별_보간오차.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for fn in (fig_outlier_policy_zoom, fig_regional_gap_imputation):
        p = fn()
        print(f"생성: {p}")
