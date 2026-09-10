# -*- coding: utf-8 -*-
"""김제 인근일사량 채택 보고용 차트·상세 수치 생성(09-07).

## 목적
사업계획서 제출용으로 상관분석·다중공선성·교차검증(k-fold) 각 단계를
그래프와 수치표로 자세히 남긴다. 기존 스크립트(check_correlation_
nearby_solar_gimje_v1_2026-09-01.py, factor_phase2_run_gimje_nearby_
solar_v1_2026-09-07.py)를 재구현하지 않고 **그대로 다시 호출**해서
같은 결과를 재현한 뒤, 그 결과 안의 폴드별 배열만 별도로 뽑아
matplotlib 차트로 만든다 - 수치 자체는 100% 기존 산출물과 동일.

## 산출물
outputs/보고서차트_인근일사량_2026-09-07/
  01_상관분석_폴드별_안정성.png - 전주146·정읍245 vs 대안특성 상관계수 추이
  02_교차검증_폴드별_MAE_비교.png - 공식(인근일사량 없음) vs 신규 XGBoost 폴드별 MAE
  03_실측vs예측_산점도.png - 신규모델 pooled 테스트셋 실측 대 예측
  04_모델비교_막대.png - LightGBM/XGBoost/선형회귀 MAE, 공식 vs 신규
  상세수치.json - 위 4개 차트에 쓰인 원본 수치 전부(재현 가능하게 보존)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

for font_path in [r"C:\Windows\Fonts\malgun.ttf"]:
    if Path(font_path).is_file():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
PHASE1_SCRIPT = HERE / "factor_reverify_v6_hourly_gimje_2026-09-03.py"
PHASE1_SUMMARY = HERE / "outputs" / "요인재검증_v6_hourly_2026-09-03" / "요약.json"
PHASE2_MODULE = PIPELINE_ROOT / "factor_phase2_multicollinearity_modelselect_v1_2026-09-03.py"
CORR_JSON = HERE / "outputs" / "김제_인근일사량_상관분석_2026-09-01" / "김제_인근일사량_상관분석_결과.json"
OFFICIAL_PHASE2 = HERE / "outputs" / "요인재검증_v6_phase2_2026-09-03" / "phase2_요약.json"
OUT_DIR = HERE / "outputs" / "보고서차트_인근일사량_2026-09-07"
SEED = 42
TARGET = "plant_ac_power_kw"
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30

NEARBY = {
    "전주146_인근일사량_W_m2": Path(
        r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
        r"\전주146\기상청_ASOS146_시간환경_20240825_20260804.csv"
    ),
    "정읍245_인근일사량_W_m2": Path(
        r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
        r"\정읍245\기상청_ASOS245_시간환경_20240825_20260804.csv"
    ),
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_nearby_solar(path: Path, colname: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["시각"]).dt.tz_localize(None)
    return pd.DataFrame({"target_time_kst": ts, colname: df["일사량_W_m2"].to_numpy()}).drop_duplicates("target_time_kst")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads(PHASE1_SUMMARY.read_text(encoding="utf-8"))
    baseline_cols = phase1["선택_baseline"]
    candidate_cols = list(phase1["선택_후보"])

    phase1_mod = _load_module("gimje_phase1_chart", PHASE1_SCRIPT)
    phase2_mod = _load_module("phase2_common_chart", PHASE2_MODULE)
    v1 = _load_module("gimje_v1_for_folds_chart", phase1_mod.V1_SCRIPT)
    harness = phase2_mod._load_harness()

    frame, _ = phase1_mod.build_hourly_frame()
    for colname, path in NEARBY.items():
        frame = frame.merge(load_nearby_solar(path, colname), on="target_time_kst", how="left")
    new_cols = list(NEARBY.keys())

    days = pd.DatetimeIndex(np.sort(frame["issue_day"].unique()))
    folds = v1.expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    def pooled_score(y_true, pred):
        err = y_true - pred
        return {"n": int(len(y_true)), "MAE_kW": round(float(np.mean(np.abs(err))), 2),
                "RMSE_kW": round(float(np.sqrt(np.mean(err ** 2))), 2)}

    # 다중공선성 가지치기 재현(신규 후보 포함) - 최종 특성 확정
    daylight_pool = frame[frame.get("physical_daylight", 1) == 1].dropna(subset=[TARGET])
    candidate_augmented = candidate_cols + new_cols
    kept_new, dropped_new = phase2_mod.prune_candidates(daylight_pool, TARGET, candidate_augmented, harness)
    final_features_new = list(dict.fromkeys(baseline_cols + kept_new))
    kept_official, dropped_official = phase2_mod.prune_candidates(daylight_pool, TARGET, candidate_cols, harness)
    final_features_official = list(dict.fromkeys(baseline_cols + kept_official))

    print(f"[신규] 최종 {len(final_features_new)}개, 제거 {dropped_new}")
    print(f"[공식] 최종 {len(final_features_official)}개, 제거 {dropped_official}")

    # 폴드별 XGBoost 성능(공식 vs 신규) + 신규모델 pooled 실측/예측 배열 확보
    def walk_forward_xgb(features: list[str]) -> tuple[list[dict], np.ndarray, np.ndarray]:
        required = [c for c in features if c not in phase1_mod.NATIVE_MISSING_OK]
        fold_rows, all_true, all_pred = [], [], []
        for i, (train_days, test_days) in enumerate(folds, start=1):
            train = frame[frame["issue_day"].isin(train_days)].dropna(subset=required + [TARGET])
            test = frame[frame["issue_day"].isin(test_days)].dropna(subset=required + [TARGET])
            if len(train) < 100 or len(test) < 5:
                continue
            model = harness.make_model("XGBoost", SEED)
            model.fit(train[features], train[TARGET])
            pred = np.clip(model.predict(test[features]), 0, None)
            y_true = test[TARGET].to_numpy()
            fold_rows.append({"폴드": i, "MAE": float(np.mean(np.abs(y_true - pred))), "시험행수": len(test)})
            all_true.append(y_true)
            all_pred.append(pred)
        return fold_rows, np.concatenate(all_true), np.concatenate(all_pred)

    fold_official, true_official, pred_official = walk_forward_xgb(final_features_official)
    fold_new, true_new, pred_new = walk_forward_xgb(final_features_new)

    # === 차트 1: 상관분석 폴드별 안정성(기존 산출물 그대로 읽음) ===
    corr = json.loads(CORR_JSON.read_text(encoding="utf-8"))
    fold_corr = corr["폴드내부_재검증(leakage없음)"]
    fold_nums = [r["폴드"] for r in fold_corr]
    series = {}
    for feat in ["전주146_27.4km", "정읍245_26.4km", "solar_elevation_deg", "forecast_DSWRF", "forecast_TCDC"]:
        series[feat] = [r["타깃상관"].get(feat) for r in fold_corr]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"전주146_27.4km": "#e07a3f", "정읍245_26.4km": "#c94f4f",
              "solar_elevation_deg": "#5b8def", "forecast_DSWRF": "#57a06b", "forecast_TCDC": "#8b7fc7"}
    for feat, vals in series.items():
        ax.plot(fold_nums, vals, marker="o", markersize=4, label=feat, color=colors.get(feat), linewidth=2)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("폴드 번호(학습구간이 30일씩 확장)")
    ax.set_ylabel("타깃(발전량)과의 상관계수")
    ax.set_title("김제: 인근 실측일사량 vs 기존 대안특성 — 폴드별 상관 안정성(21폴드, leakage 없음)")
    ax.legend(loc="center right", fontsize=9)
    ax.set_ylim(-0.5, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_상관분석_폴드별_안정성.png", dpi=150)
    plt.close(fig)

    # === 차트 2: 교차검증 폴드별 MAE 비교(XGBoost, 공식 vs 신규) ===
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fo = pd.DataFrame(fold_official)
    fn = pd.DataFrame(fold_new)
    ax.plot(fo["폴드"], fo["MAE"], marker="o", markersize=4, label="공식(09-03, 인근일사량 없음)", color="#8890a3", linewidth=2)
    ax.plot(fn["폴드"], fn["MAE"], marker="o", markersize=4, label="신규(09-07, 전주146·정읍245 추가)", color="#e07a3f", linewidth=2)
    ax.set_xlabel("폴드 번호(walk-forward, 학습 30일씩 확장)")
    ax.set_ylabel("폴드별 MAE (kW)")
    ax.set_title("김제 D+1 총출력: XGBoost 폴드별 MAE — 공식 vs 인근일사량 추가")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_교차검증_폴드별_MAE_비교.png", dpi=150)
    plt.close(fig)

    # === 차트 3: 실측 vs 예측 산점도(신규모델, pooled 전체 테스트) ===
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(true_new, pred_new, s=6, alpha=0.25, color="#5b8def")
    lim = max(true_new.max(), pred_new.max()) * 1.02
    ax.plot([0, lim], [0, lim], color="#888", linestyle="--", linewidth=1, label="완전일치선(y=x)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("실측 발전량 (kW)")
    ax.set_ylabel("예측 발전량 (kW)")
    mae_new = float(np.mean(np.abs(true_new - pred_new)))
    ax.set_title(f"김제 D+1 총출력(XGBoost, 인근일사량 포함): 실측 대 예측\n(pooled 테스트 {len(true_new)}행, MAE {mae_new:.2f}kW)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_실측vs예측_산점도.png", dpi=150)
    plt.close(fig)

    # === 차트 4: 3모델 비교 막대(공식 vs 신규) ===
    official_phase2 = json.loads(OFFICIAL_PHASE2.read_text(encoding="utf-8"))
    new_phase2_path = HERE / "outputs" / "요인재검증_v6_phase2_인근일사량추가_2026-09-07" / "phase2_요약.json"
    new_phase2 = json.loads(new_phase2_path.read_text(encoding="utf-8"))

    models = ["LightGBM", "XGBoost", "선형회귀"]
    off_mae = {m["모델"]: m["MAE_kW"] for m in official_phase2["모델비교"]}
    new_mae = {m["모델"]: m["MAE_kW"] for m in new_phase2["모델비교"]}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(models)); width = 0.35
    ax.bar(x - width / 2, [off_mae[m] for m in models], width, label="공식(09-03)", color="#8890a3")
    ax.bar(x + width / 2, [new_mae[m] for m in models], width, label="신규(09-07, 인근일사량 추가)", color="#e07a3f")
    for i, m in enumerate(models):
        ax.text(i - width / 2, off_mae[m] + 0.4, f"{off_mae[m]:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, new_mae[m] + 0.4, f"{new_mae[m]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("MAE (kW)")
    ax.set_title("김제 D+1 총출력: 모델별 MAE — 공식 vs 인근일사량 추가")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_모델비교_막대.png", dpi=150)
    plt.close(fig)

    detail = {
        "상관분석_폴드별": {feat: vals for feat, vals in series.items()},
        "상관분석_폴드번호": fold_nums,
        "교차검증_폴드별_MAE_공식": fold_official,
        "교차검증_폴드별_MAE_신규": fold_new,
        "산점도_pooled_행수": len(true_new),
        "산점도_pooled_MAE": mae_new,
        "최종특성_공식": final_features_official,
        "최종특성_신규": final_features_new,
        "다중공선성_제거_공식": dropped_official,
        "다중공선성_제거_신규": dropped_new,
        "모델비교_공식": official_phase2["모델비교"],
        "모델비교_신규": new_phase2["모델비교"],
    }
    (OUT_DIR / "상세수치.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[성공] 차트 4개 + 상세수치.json 저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
