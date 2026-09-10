# -*- coding: utf-8 -*-
"""기존 shadow 성공행 중 입력결측 예측을 감사하고 무효표시한다.

원본 DB를 같은 폴더에 타임스탬프 백업한 뒤, 예측값은 보존하면서 상태만
``invalid_input``으로 바꾼다. API 호출은 하지 않는다.
"""
from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DB = ROOT / "outputs" / "shadow_predictions" / "shadow_predictions.sqlite3"


def main() -> None:
    if not DB.is_file():
        raise FileNotFoundError(DB)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DB.with_name(f"{DB.stem}_before_readiness_audit_{stamp}{DB.suffix}")
    shutil.copy2(DB, backup)
    with sqlite3.connect(DB) as conn:
        rows = conn.execute(
            "SELECT id, horizon_h, issue_time, n_missing_features FROM shadow_predictions "
            "WHERE status='success' AND COALESCE(n_missing_features, 0) > 0"
        ).fetchall()
        conn.execute(
            "UPDATE shadow_predictions SET status='invalid_input', "
            "reason='운영 입력결측 상태에서 예측되어 사후 readiness 감사로 무효화' "
            "WHERE status='success' AND COALESCE(n_missing_features, 0) > 0"
        )
        conn.commit()
    print(f"백업: {backup}")
    print(f"무효표시: {len(rows)}건")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
