# -*- coding: utf-8 -*-
"""부안 중장기(일간 D+1) 배포용 번들 생성 - 기존 build_medium() 그대로 재사용.

`build_regional_ultrashort_mediumterm_issue_safe_v1_2026-09-08.py`의
`build_medium()`은 지역 무관하게 동작하도록 이미 만들어져 있었다(김제·
영광이 이미 이 함수를 그대로 씀). 부안만 그때 `medium_term_daily_v1_
buan_*.py`가 없어서 호출 안 됐을 뿐이다(09-14 신규 작성 완료). 이
스크립트는 build_medium()을 재구현하지 않고 그대로 불러와 부안에
적용한다.

가을 walk-forward 미검증 상태 그대로 저장(manifest에 자동 기록됨 -
build_medium 자체가 walk_forward_kWh를 남기므로 이 결과를 보면 가을
데이터가 언제 들어왔는지 재확인 가능).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUAN_DIR = ROOT / "부안_준비_2026-08-28"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없음: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    build_mod = load_module("build_ultra_for_buan_medium",
                            ROOT / "build_regional_ultrashort_mediumterm_issue_safe_v1_2026-09-08.py")
    rv = load_module("revalidate_for_buan_medium",
                     ROOT / "revalidate_all_horizons_issue_safe_v1_2026-09-08.py")

    result = build_mod.build_medium(
        "부안", BUAN_DIR, BUAN_DIR / "medium_term_daily_v1_buan_2026-09-14.py", rv)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
