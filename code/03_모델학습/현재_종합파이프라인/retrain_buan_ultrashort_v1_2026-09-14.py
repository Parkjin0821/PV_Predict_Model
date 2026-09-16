# -*- coding: utf-8 -*-
"""부안 초단기(+1~4h) 재학습 후보 생성 - 주기적 재학습 파이프라인의 1단계.

09-14 사용자 승인으로 착수: 4지역 초단기 shadow nMAE가 "날이 지나도
줄어들지 않는다"는 질문에 답하는 과정에서, 지금 구조(고정 번들+48h
롤링평가)는 재학습 없이는 성능이 개선될 수 없다는 게 확인돼 주기적
재학습(드리프트 감지+월1회 백스톱, 게이트 통과시 자동 승격)을 검토·
착수하기로 함. 4지역 중 부안부터 시범.

## 이 스크립트가 하는 일 (1단계: 후보 생성만)
`build_regional_ultrashort_mediumterm_issue_safe_v1_2026-09-08.py`의
`build_buan_ultra()`를 그대로 재사용(재구현 안 함, 09-14에 `bundle_
version` 인자만 추가하고 왕복검증으로 기존과 SHA-256 완전 동일함을
확인했음)해서, **운영 중인 번들(`공식_초단기_issue_safe_v1_2026-09-08`)은
절대 건드리지 않고** 날짜스탬프가 붙은 별도 후보 폴더에 새로 학습한다.

## 아직 안 하는 일 (다음 단계)
- 게이트 평가(기존 dashboard 게이트 로직 재사용 예정) 자동 실행
- 게이트 통과 시 `active_bundle.json` 원자적 교체(자동 승격)
- 드리프트 감지 트리거 + 월 1회 백스톱 스케줄러
이 스크립트는 후보를 만들어 결과만 남긴다 - 승격은 별도 검증 후 수동/
자동화 예정.

## 전제조건
`부안_인버터별_5분_야간0포함.parquet`이 최신 데이터로 갱신돼 있어야
"재학습"이 의미가 있다(09-14 확인 시점 09-01 10:55 이후 미갱신 -
코덱스에게 전처리 체인 재실행 요청해둠, AGENTS.md 09-14 항목 참고).
이 스크립트는 그 parquet을 그대로 읽으므로, 갱신 전에 돌리면 기존과
동일한 모델이 나올 뿐이다(이미 09-14 검증에서 SHA-256 동일 확인됨).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
ROOT = Path(__file__).resolve().parent
BUAN_DIR = ROOT / "부안_준비_2026-08-28"
BUILD_SCRIPT = ROOT / "build_regional_ultrashort_mediumterm_issue_safe_v1_2026-09-08.py"
RV_SCRIPT = ROOT / "revalidate_all_horizons_issue_safe_v1_2026-09-08.py"
SOURCE_PARQUET = (
    Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28")
    / "부안_인버터별_5분_야간0포함.parquet"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없음: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    now = datetime.now(tz=KST)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    bundle_version = f"공식_초단기_재학습후보_{stamp}"

    parquet_age_hours = None
    if SOURCE_PARQUET.exists():
        mtime = datetime.fromtimestamp(SOURCE_PARQUET.stat().st_mtime, tz=KST)
        parquet_age_hours = round((now - mtime).total_seconds() / 3600, 1)

    build_mod = load_module("build_ultra_for_retrain", BUILD_SCRIPT)
    rv = load_module("revalidate_all_horizons_for_retrain", RV_SCRIPT)

    result = build_mod.build_buan_ultra(rv, bundle_version=bundle_version)

    summary = {
        "region": "부안", "bundle_version": bundle_version,
        "created_at_kst": now.isoformat(timespec="seconds"),
        "source_parquet_age_hours": parquet_age_hours,
        "source_parquet_stale_warning": (
            "48시간 이상 갱신 안 된 데이터로 재학습함 - 새 데이터가 거의/전혀 "
            "반영 안 됐을 수 있음. 코덱스 전처리 체인 갱신 여부 확인 필요."
            if parquet_age_hours is not None and parquet_age_hours > 48
            else None
        ),
        "horizons": {
            h: {"model_sha256": r["model_sha256"], "selected_model": r["selected_model"],
                "nmae_pct_selected": r["nmae_pct_selected"], "training_rows": r["training_rows"]}
            for h, r in result.items()
        },
    }
    out_dir = BUAN_DIR / "outputs" / bundle_version
    (out_dir / "재학습_요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if summary["source_parquet_stale_warning"]:
        print(f"\n[경고] {summary['source_parquet_stale_warning']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
