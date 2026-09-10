# -*- coding: utf-8 -*-
"""★09-01 저녁부로 대체(SUPERSEDED)★ - 더 이상 실행되지 않음, 이력 보존용으로만 남김.

`collect_kma_nwp_live_v1_2026-08-25.py`에 `--vars` CLI 옵션을 정식
추가(09-01, "지역 공통 파이프라인" 논의 결과 - 격자예보·NWP 라이브
수집기 2개는 사이트별 래퍼 대신 CLI 파라미터로 통일)했으므로 이 래퍼는
더 이상 필요 없다. `UCUBE_NWP_D1_gimje_daily` 스케줄 작업도 원본
스크립트를 직접 가리키도록 이미 변경됨:

    python collect_kma_nwp_live_v1_2026-08-25.py --vars DSWRF TCDC LCDC MCDC HCDC \
      --latitude 35.80026670423991 --longitude 126.851859588009 \
      --output-dir "...\김제\kma_live_inputs_gimje_v1_2026-09-01"

아래 원본 코드는 09-01 오전~낮 동안 실제로 이 방식(모듈 override)으로
동작했던 이력 보존 목적으로만 유지한다. 새로 수정·재사용하지 말 것.

--- 이하 원본 docstring(참고용) ---

김제 운영용 KMA 고정-tm NWP(D+1) 라이브 수집기 - 광주 09시런 래퍼 재사용.

`collect_kma_nwp_live_v1_2026-08-25.py`(광주용)를 그대로 재사용하되,
Gimje 09-01 확정 route(`김제_기상수집경로_v1_2026-08-31.json`)의 두
가지를 덮어쓴다:

1. `RUN_VARS`에서 DSWRFLX·DIFSWRF 제외 - 08-28 기상청 공식회신으로
   KIMR 상류결측(2026-06-01~ 영구중단)이 확인돼 광주·부안·김제 전부
   완결성 필수변수에서 제외하기로 확정된 결정을 그대로 승계. 원본
   광주 라이브 스크립트는 `--vars` CLI 옵션이 없어(과거 710일 백필
   스크립트와 달리) 이 두 변수를 계속 호출하게 되므로, 여기서
   모듈 임포트 후 `src.RUN_VARS`를 직접 덮어써서 막는다(원본 파일은
   수정하지 않음 - 광주 자체 라이브 스케줄에 영향 없음).
2. 기본 위경도·출력경로를 김제로(광주 값 계승 방지).

CLI는 원본과 동일(`--issue-date`, `--output-dir`, `--latitude`,
`--longitude`, `--max-retries`, `--timeout-seconds`,
`--request-delay-seconds`, `--force`) - 기본값만 김제로 교체.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "collect_kma_nwp_live_v1_2026-08-25.py"

GIMJE_LATITUDE = 35.80026670423991
GIMJE_LONGITUDE = 126.851859588009
GIMJE_OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_live_inputs_gimje_v1_2026-09-01"
)
GIMJE_RUN_VARS = ["DSWRF", "TCDC", "LCDC", "MCDC", "HCDC"]


def load_source():
    spec = importlib.util.spec_from_file_location("ucube_nwp_live_gimje_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"기존 광주 NWP 라이브 수집기를 불러올 수 없습니다: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    live = load_source()

    # DSWRFLX·DIFSWRF 제외(08-28 기상청 공식회신 - 김제 route와 동일 결정 승계).
    live.src.RUN_VARS = GIMJE_RUN_VARS

    # CLI 기본값을 김제로 교체(사용자가 --latitude 등으로 여전히 오버라이드 가능).
    live.DEFAULT_OUT = GIMJE_OUTPUT_DIR
    live.src.LATITUDE = GIMJE_LATITUDE
    live.src.LONGITUDE = GIMJE_LONGITUDE

    return live.main()


if __name__ == "__main__":
    raise SystemExit(main())
