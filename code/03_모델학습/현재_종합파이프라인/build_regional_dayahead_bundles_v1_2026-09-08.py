"""사용 금지: 기존 phase2 D+1 특성의 시간누출을 발견한 감사 이력 파일.

API를 호출하지 않는다. 이미 검증된 frame 생성기와 phase2 모델 팩토리를
재사용하며, 마지막 7일 replay에서 유한값·물리범위·특성순서를 확인한다.
09-08 실행 후 target-relative power lag와 김제 미래 일사관측 누출을 발견했다.
생성된 번들은 manifest에서 invalidated 처리했다. 다시 실행하지 않는다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PHASE2 = ROOT / "factor_phase2_multicollinearity_modelselect_v1_2026-09-03.py"
SEED = 42
TARGET = "plant_ac_power_kw"

REGIONS = {
    "부안": {
        "dir": ROOT / "부안_준비_2026-08-28",
        "builder": "factor_reverify_v6_hourly_buan_2026-09-03.py",
        "summary": "outputs/요인재검증_v6_phase2_2026-09-03/phase2_요약.json",
        "capacity": 1000.0,
    },
    "김제": {
        "dir": ROOT / "김제_준비_2026-09-01",
        "builder": "factor_reverify_v6_hourly_gimje_2026-09-03.py",
        "summary": "outputs/요인재검증_v6_phase2_인근일사량추가_2026-09-07/phase2_요약.json",
        "capacity": 1100.0,
        "nearby": {
            "전주146_인근일사량_W_m2": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01\전주146\기상청_ASOS146_시간환경_20240825_20260804.csv"),
            "정읍245_인근일사량_W_m2": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01\정읍245\기상청_ASOS245_시간환경_20240825_20260804.csv"),
        },
    },
    "영광": {
        "dir": ROOT / "영광_준비_2026-09-03",
        "builder": "factor_build_v6_hourly_yeonggwang_2026-09-03.py",
        "summary": "outputs/요인구축_v6_phase2_2026-09-03/phase2_요약.json",
        "capacity": 634.0,
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈 로드 실패: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add_nearby(frame: pd.DataFrame, mapping: dict[str, Path]) -> pd.DataFrame:
    out = frame
    for col, path in mapping.items():
        src = pd.read_csv(path, usecols=["시각", "일사량_W_m2"])
        src["target_time_kst"] = pd.to_datetime(src.pop("시각"), errors="raise").dt.tz_localize(None)
        src = src.rename(columns={"일사량_W_m2": col}).drop_duplicates("target_time_kst")
        out = out.merge(src, on="target_time_kst", how="left")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(region: str, cfg: dict, harness) -> dict:
    module = load_module(f"bundle_builder_{region}", cfg["dir"] / cfg["builder"])
    built = module.build_hourly_frame() if region != "부안" else module.build_hourly_frame(
        module._load_module("buan_v5_bundle", module.V5_SCRIPT)
    )
    frame = built[0] if isinstance(built, tuple) else built
    if cfg.get("nearby"):
        frame = add_nearby(frame, cfg["nearby"])

    summary_path = cfg["dir"] / cfg["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    features = list(summary["최종특성"])
    model_name = summary["선정모델(MAE기준)"]
    missing_columns = [c for c in features if c not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"{region}: 특성 컬럼 부재 {missing_columns}")

    native_missing_ok = set(getattr(module, "NATIVE_MISSING_OK", []))
    required = [c for c in features if c not in native_missing_ok]
    clean = frame.dropna(subset=required + [TARGET]).sort_values("target_time_kst").copy()
    if len(clean) < 100:
        raise RuntimeError(f"{region}: 학습 가능 행 부족 {len(clean)}")

    model = harness.make_model(model_name, SEED)
    model.fit(clean[features], clean[TARGET])
    replay_start = clean["target_time_kst"].max() - pd.Timedelta(days=7)
    replay = clean[clean["target_time_kst"] >= replay_start]
    pred = np.asarray(model.predict(replay[features]), dtype=float)
    clipped = np.clip(pred, 0.0, cfg["capacity"])
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{region}: replay 비유한 예측")
    if list(replay[features].columns) != features:
        raise RuntimeError(f"{region}: 특성 순서 불일치")

    out_dir = cfg["dir"] / "outputs" / "공식_D1_추론번들_v1_2026-09-08"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "model.joblib"
    temp_path = out_dir / "model.joblib.tmp"
    payload = {
        "model": model,
        "features": features,
        "region": region,
        "horizon": "D+1",
        "capacity_kw": cfg["capacity"],
        "model_name": model_name,
    }
    joblib.dump(payload, temp_path)
    loaded = joblib.load(temp_path)
    check = np.asarray(loaded["model"].predict(replay[features]), dtype=float)
    if not np.allclose(pred, check, rtol=0, atol=1e-10):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"{region}: 저장 왕복 예측 불일치")
    os.replace(temp_path, final_path)

    manifest = {
        "status": "ready",
        "region": region,
        "horizon": "D+1_only",
        "model_name": model_name,
        "features": features,
        "feature_count": len(features),
        "capacity_kw": cfg["capacity"],
        "training_rows": len(clean),
        "training_start_kst": str(clean["target_time_kst"].min()),
        "training_end_kst": str(clean["target_time_kst"].max()),
        "replay_rows": len(replay),
        "replay_raw_min_kw": float(pred.min()),
        "replay_raw_max_kw": float(pred.max()),
        "replay_clipped_rows": int(np.count_nonzero(pred != clipped)),
        "future_leak_check": "training frame uses established issue/target construction; live inference must separately enforce received_at<=issue_at",
        "source_summary": str(summary_path),
        "model_sha256": sha256(final_path),
        "created_at_kst": datetime.now().astimezone().isoformat(),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    raise RuntimeError("사용 금지: D+1 target-relative lag 시간누출 발견. manifest가 invalidated 처리됨")
    # 아래 코드는 재현·감사 이력으로만 보존한다.
    phase2 = load_module("regional_bundle_phase2", PHASE2)
    harness = phase2._load_harness()
    results = [build(region, cfg, harness) for region, cfg in REGIONS.items()]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
