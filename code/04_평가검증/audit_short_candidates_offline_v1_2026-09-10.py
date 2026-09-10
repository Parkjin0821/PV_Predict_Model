# -*- coding: utf-8 -*-
"""4지역 +24h 및 3지역 +48h 공식후보 번들의 오프라인 무결성 감사."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
ROOT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")
OUT = ROOT / "outputs" / "단기후보_오프라인감사_2026-09-10"

BUNDLES = {
    ("광주", "+24h"): ROOT / "광주_준비_2026-09-08" / "outputs" / "단기_재학습_후보_2026-09-09" / "+24h",
    ("부안", "+24h"): ROOT / "부안_준비_2026-08-28" / "outputs" / "단기_재학습_후보_2026-09-09" / "+24h",
    ("김제", "+24h"): ROOT / "김제_준비_2026-09-01" / "outputs" / "단기_재학습_후보_2026-09-09" / "+24h",
    ("영광", "+24h"): ROOT / "영광_준비_2026-09-03" / "outputs" / "단기_재학습_후보_2026-09-09" / "+24h",
    ("부안", "+48h"): ROOT / "부안_준비_2026-08-28" / "outputs" / "단기_재학습_후보_2026-09-09" / "+48h",
    ("김제", "+48h"): ROOT / "김제_준비_2026-09-01" / "outputs" / "단기_재학습_후보_2026-09-09" / "+48h",
    ("영광", "+48h"): ROOT / "영광_준비_2026-09-03" / "outputs" / "단기_재학습_후보_2026-09-09" / "+48h",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def walk_forward_selected(manifest: dict) -> dict:
    rows = manifest.get("walk_forward") or []
    selected = manifest.get("selected_model")
    for row in rows:
        if row.get("model") == selected:
            return row
    return {}


def audit_one(region: str, horizon: str, folder: Path) -> dict:
    manifest_path = folder / "manifest.json"
    model_path = folder / "model.joblib"
    result = {"region": region, "horizon": horizon, "folder": str(folder), "checks": {}}
    checks = result["checks"]
    checks["manifest_exists"] = manifest_path.exists()
    checks["model_exists"] = model_path.exists()
    if not all((checks["manifest_exists"], checks["model_exists"])):
        result["verdict"] = "FAIL"
        return result

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = joblib.load(model_path)
    features = list(manifest.get("features") or manifest.get("feature_cols") or [])
    bundle_features = list(bundle.get("features") or bundle.get("feature_cols") or [])
    model = bundle.get("model")
    digest = sha256(model_path)
    selected_metrics = walk_forward_selected(manifest)
    if not selected_metrics:
        selected_metrics = ((manifest.get("검증") or {}).get("seg1_홀드아웃_최근30일_평가") or {})
    leak = manifest.get("leakage_audit") or ((manifest.get("검증") or {}).get("리크감사") or {})
    selected_model = manifest.get("selected_model") or bundle.get("model_name") or type(model).__name__

    checks.update({
        "status_is_candidate": manifest.get("status") == "공식_후보",
        "region_matches": manifest.get("region") == region,
        "horizon_matches": manifest.get("horizon") == horizon,
        "sha256_matches": digest == manifest.get("model_sha256"),
        "features_nonempty": bool(features),
        "manifest_bundle_features_equal": features == bundle_features,
        "model_feature_count_matches": getattr(model, "n_features_in_", len(features)) == len(features),
        "leakage_max_target_lag_zero": leak.get("max_target_relative_lag_used") == 0,
        "operation_is_held": "보류" in str(manifest.get("운영_연결", "")),
    })

    smoke_ok = False
    smoke_error = None
    try:
        sample = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        pred = np.asarray(model.predict(sample), dtype=float)
        smoke_ok = pred.shape == (1,) and np.isfinite(pred[0])
    except Exception as exc:  # noqa: BLE001
        smoke_error = f"{type(exc).__name__}: {exc}"
    checks["deserialized_predict_smoke"] = smoke_ok
    result["checks"] = {key: bool(value) for key, value in checks.items()}
    checks = result["checks"]

    test_rows = selected_metrics.get("test_rows", selected_metrics.get("n"))
    warnings = []
    if test_rows is not None and int(test_rows) < 30:
        warnings.append(f"검증 표본 {test_rows}건으로 공식 승격 근거 부족")
    if region == "영광" and "미실시" not in str(manifest.get("defect_audit", "")):
        warnings.append("영광 결함감사 미실시 표기가 누락됨")

    result.update({
        "selected_model": selected_model,
        "feature_count": len(features),
        "training_rows": manifest.get("training_rows", manifest.get("train_rows")),
        "test_rows": test_rows,
        "mae_kw": selected_metrics.get("MAE_kW"),
        "rmse_kw": selected_metrics.get("RMSE_kW"),
        "sha256": digest,
        "smoke_error": smoke_error,
        "warnings": warnings,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    })
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [audit_one(region, horizon, folder) for (region, horizon), folder in BUNDLES.items()]
    overall = "PASS_WITH_WARNINGS" if all(r["verdict"] == "PASS" for r in rows) else "FAIL"
    payload = {
        "created_at_kst": datetime.now(KST).isoformat(),
        "scope": "4지역 +24h, 부안·김제·영광 +48h 후보 번들",
        "overall": overall,
        "promotion_decision": "보류 유지",
        "results": rows,
    }
    atomic_text(OUT / "감사결과.json", json.dumps(payload, ensure_ascii=False, indent=2))

    lines = [
        "# 단기 후보 오프라인 감사 (2026-09-10)", "",
        f"- 종합판정: **{overall}**", "- 운영 연결: **보류 유지**", "",
        "| 지역 | 수평 | 모델 | 특성 | 학습 | 검증 | MAE(kW) | RMSE(kW) | 판정 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['region']} | {r['horizon']} | {r.get('selected_model')} | "
            f"{r.get('feature_count')} | {r.get('training_rows')} | {r.get('test_rows')} | "
            f"{r.get('mae_kw')} | {r.get('rmse_kw')} | {r['verdict']} |"
        )
    lines += ["", "## 경고"]
    warning_count = 0
    for r in rows:
        for warning in r.get("warnings", []):
            warning_count += 1
            lines.append(f"- {r['region']} {r['horizon']}: {warning}")
    if not warning_count:
        lines.append("- 없음")
    lines += ["", "## 해석", "- PASS는 파일·스키마·누출표시·직렬화 무결성 통과를 뜻하며 공식 승격을 뜻하지 않는다.", "- API 장애기간의 라이브 생성률은 모델 성능 판정에서 제외한다."]
    atomic_text(OUT / "감사보고서.md", "\n".join(lines) + "\n")
    print(json.dumps({"overall": overall, "output": str(OUT), "results": [{"region": r["region"], "horizon": r["horizon"], "verdict": r["verdict"], "warnings": r.get("warnings", [])} for r in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
