"""기상청 ASOS 환경자료 수집 실행파일.

예시:
  python 기상청_환경자료_수집실행.py --mode daily
"""

from __future__ import annotations

from download_kma_public_asos import main


if __name__ == "__main__":
    raise SystemExit(main())
