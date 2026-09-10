# -*- coding: utf-8 -*-
"""★09-01 저녁부로 대체(SUPERSEDED)★ - 더 이상 실행되지 않음, 이력 보존용으로만 남김.

`collect_kma_nwp_live_v1_2026-08-25.py`에 `--vars` CLI 옵션을 정식
추가했으므로 이 래퍼는 더 이상 필요 없다. `UCUBE_NWP_D1_buan_daily`
스케줄 작업도 원본 스크립트를 직접 가리키도록 이미 변경됨:

    python collect_kma_nwp_live_v1_2026-08-25.py --vars DSWRF TCDC LCDC MCDC HCDC \
      --latitude 35.7874617462152 --longitude 126.73000042548799 \
      --output-dir "...\부안\kma_live_inputs_v1_2026-08-28"

--- 이하 원본 docstring(참고용, 새로 수정·재사용하지 말 것) ---

부안 운영용 KMA 고정-tm NWP(D+1) 라이브 수집기 - 광주 09시런 래퍼 재사용.

`collect_kma_nwp_live_gimje_v1_2026-09-01.py`와 동일 패턴(원본 광주
라이브 스크립트를 그대로 재사용, 원본 파일은 안 건드림) - 위경도·
출력경로·RUN_VARS만 부안으로 교체.

DSWRFLX·DIFSWRF 제외 이유는 김제 래퍼와 동일: 08-28 기상청 공식회신으로
KIMR 상류결측(2026-06-01~ 영구중단) 확인, 부안 과거 백필도 이미 이
두 변수를 제외해왔음("광주·부안과 동일 판단 그대로 적용" - 김제
`김제_기상수집경로_v1_2026-08-31.json` 기록).

09-01 발견: 부안은 코덱스 자체 예약(토큰 소진 등으로 조용히 중단
가능)에 의존하고 있었고 실제로 NWP는 08-28 이후 중단돼 있었다 -
Windows 작업 스케줄러 등록으로 이 의존성 자체를 없앤다(이 스크립트는
단순 결정론적 파이썬이라 코덱스 실행 없이도 동작).

CLI는 원본과 동일 - 기본값만 부안으로 교체.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "collect_kma_nwp_live_v1_2026-08-25.py"

BUAN_LATITUDE = 35.7874617462152
BUAN_LONGITUDE = 126.73000042548799
BUAN_OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_live_inputs_v1_2026-08-28"
)
BUAN_RUN_VARS = ["DSWRF", "TCDC", "LCDC", "MCDC", "HCDC"]


def load_source():
    spec = importlib.util.spec_from_file_location("ucube_nwp_live_buan_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"기존 광주 NWP 라이브 수집기를 불러올 수 없습니다: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    live = load_source()

    live.src.RUN_VARS = BUAN_RUN_VARS
    live.DEFAULT_OUT = BUAN_OUTPUT_DIR
    live.src.LATITUDE = BUAN_LATITUDE
    live.src.LONGITUDE = BUAN_LONGITUDE

    return live.main()


if __name__ == "__main__":
    raise SystemExit(main())
