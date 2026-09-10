"""기상청 단기예보 조회서비스에서 발표시각별 익일 예보를 수집한다.

기본 실행:
  python 기상청_단기예보_수집.py

기본 조건:
  - 광주 치평동 격자: 가로 58, 세로 74
  - 매일 오전 8시 발표분
  - 발표 다음 날 00~23시 예보만 모델용으로 저장
  - 기간: 2024-08-25 ~ 2026-08-04

주의:
  공공데이터포털의 일반 단기예보 서비스가 오래된 발표분을 보관하지 않는 경우
  과거 날짜에는 '자료 없음'이 반환될 수 있다. 이 경우 현재 운영 수집에는 이
  서비스를 사용하고, 과거 학습자료는 기상청 API허브 과거이력으로 받아야 한다.
"""

from __future__ import annotations

import argparse
import getpass
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode
from urllib.request import urlopen

import pandas as pd


서비스_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
기본_시작일 = "2024-08-25"
기본_종료일 = "2026-08-04"
기본_발표시각 = "0800"
격자_가로 = 58
격자_세로 = 74

요소명 = {
    "TMP": "기온_C",
    "TMN": "일최저기온_C",
    "TMX": "일최고기온_C",
    "REH": "상대습도_pct",
    "SKY": "하늘상태",
    "POP": "강수확률_pct",
    "PCP": "1시간강수량_원문",
    "PTY": "강수형태",
    "SNO": "1시간신적설_원문",
    "WSD": "풍속_m_s",
    "VEC": "풍향_deg",
    "UUU": "동서바람성분_m_s",
    "VVV": "남북바람성분_m_s",
    "WAV": "파고_m",
}


def 인수읽기() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="기상청 단기예보 익일자료 수집")
    parser.add_argument("--start", default=기본_시작일, help="발표 시작일 YYYY-MM-DD")
    parser.add_argument("--end", default=기본_종료일, help="발표 종료일 YYYY-MM-DD")
    parser.add_argument("--base-time", default=기본_발표시각, help="발표시각 HHMM, 기본 0800")
    parser.add_argument("--sleep", type=float, default=0.35, help="정상 호출 사이 대기초")
    parser.add_argument("--output-dir", default=None, help="저장폴더")
    return parser.parse_args()


def 날짜목록(시작일: str, 종료일: str) -> list[str]:
    시작 = datetime.strptime(시작일, "%Y-%m-%d")
    종료 = datetime.strptime(종료일, "%Y-%m-%d")
    if 종료 < 시작:
        raise ValueError("종료일이 시작일보다 빠릅니다.")
    결과 = []
    현재 = 시작
    while 현재 <= 종료:
        결과.append(현재.strftime("%Y%m%d"))
        현재 += timedelta(days=1)
    return 결과


def 인증키읽기() -> str:
    값 = getpass.getpass("공공데이터포털 일반 인증키 입력(화면에 표시되지 않음): ").strip()
    if not 값:
        raise ValueError("인증키가 비어 있습니다.")
    # 포털에서 복사한 인코딩 키와 디코딩 키를 모두 받을 수 있게 한 번만 해제한다.
    return unquote(값)


def 완료목록읽기(경로: Path) -> set[str]:
    if not 경로.exists():
        return set()
    자료 = json.loads(경로.read_text(encoding="utf-8"))
    return set(자료.get("완료발표", []))


def 완료목록저장(경로: Path, 완료: set[str]) -> None:
    임시 = 경로.with_suffix(".tmp")
    임시.write_text(
        json.dumps({"완료발표": sorted(완료)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    임시.replace(경로)


def 요청(인증키: str, 발표일: str, 발표시각: str) -> list[dict]:
    매개변수 = {
        "serviceKey": 인증키,
        "pageNo": 1,
        "numOfRows": 2000,
        "dataType": "JSON",
        "base_date": 발표일,
        "base_time": 발표시각,
        "nx": 격자_가로,
        "ny": 격자_세로,
    }
    마지막오류: Exception | None = None
    for 시도 in range(1, 4):
        try:
            주소 = 서비스_URL + "?" + urlencode(매개변수)
            with urlopen(주소, timeout=60) as 응답:
                본문 = json.loads(응답.read().decode("utf-8")).get("response", {})
            머리 = 본문.get("header", {})
            코드 = str(머리.get("resultCode", ""))
            설명 = str(머리.get("resultMsg", ""))
            if 코드 == "03":
                return []
            if 코드 not in {"00", "0"}:
                raise RuntimeError(f"기상청 응답 오류 {코드}: {설명}")
            항목 = 본문.get("body", {}).get("items", {}).get("item", [])
            return 항목 if isinstance(항목, list) else []
        except HTTPError as 오류:
            if 오류.code == 403:
                오류 = RuntimeError("HTTP 403: 인증키 승인상태·일일한도·서비스 활용신청을 확인하세요.")
            elif 오류.code == 429:
                오류 = RuntimeError("HTTP 429: 호출한도를 초과했습니다.")
            마지막오류 = 오류
            if 시도 == 3 or "403" in str(오류):
                break
            대기 = 5 * 시도
            print(f"    일시 오류: {대기}초 후 재시도 ({시도}/3) - {오류}")
            time.sleep(대기)
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError, RuntimeError) as 오류:
            마지막오류 = 오류
            if 시도 == 3 or "403" in str(오류) or "응답 오류" in str(오류):
                break
            대기 = 5 * 시도
            print(f"    일시 오류: {대기}초 후 재시도 ({시도}/3) - {오류}")
            time.sleep(대기)
    raise RuntimeError(f"{발표일} {발표시각} 수집 실패: {마지막오류}")


def 표준화(항목: list[dict], 발표일: str, 발표시각: str) -> pd.DataFrame:
    발표 = pd.to_datetime(발표일 + 발표시각, format="%Y%m%d%H%M")
    익일 = 발표.normalize() + pd.Timedelta(days=1)
    행 = []
    for 값 in 항목:
        대상시각 = pd.to_datetime(
            str(값.get("fcstDate", "")) + str(값.get("fcstTime", "")).zfill(4),
            format="%Y%m%d%H%M",
            errors="coerce",
        )
        if pd.isna(대상시각) or 대상시각.normalize() != 익일:
            continue
        코드 = str(값.get("category", ""))
        행.append(
            {
                "예보발표시각": 발표,
                "예보대상시각": 대상시각,
                "요소코드": 코드,
                "요소명": 요소명.get(코드, 코드),
                "예보값_원문": 값.get("fcstValue"),
                "격자가로": 값.get("nx", 격자_가로),
                "격자세로": 값.get("ny", 격자_세로),
            }
        )
    return pd.DataFrame(행)


def 통합저장(일별폴더: Path, 원본경로: Path, 모델경로: Path) -> tuple[int, int]:
    파일들 = sorted(일별폴더.glob("*.csv"))
    if not 파일들:
        return 0, 0
    원본 = pd.concat([pd.read_csv(경로, low_memory=False) for 경로 in 파일들], ignore_index=True)
    원본 = 원본.drop_duplicates(["예보발표시각", "예보대상시각", "요소코드"], keep="last")
    원본 = 원본.sort_values(["예보발표시각", "예보대상시각", "요소코드"])
    원본.to_csv(원본경로, index=False, encoding="utf-8-sig")

    모델 = 원본.pivot_table(
        index=["예보발표시각", "예보대상시각"],
        columns="요소명",
        values="예보값_원문",
        aggfunc="last",
    ).reset_index()
    모델.columns.name = None
    모델.to_csv(모델경로, index=False, encoding="utf-8-sig")
    return len(원본), len(모델)


def main() -> None:
    인수 = 인수읽기()
    if len(인수.base_time) != 4 or not 인수.base_time.isdigit():
        raise ValueError("발표시각은 0800처럼 네 자리로 입력하세요.")
    저장폴더 = Path(인수.output_dir) if 인수.output_dir else Path(__file__).resolve().parent / "processed" / "기상청_단기예보"
    일별폴더 = 저장폴더 / "일별원본"
    일별폴더.mkdir(parents=True, exist_ok=True)
    진행경로 = 저장폴더 / "수집진행.json"
    원본경로 = 저장폴더 / "기상청_단기예보_익일_원본.csv"
    모델경로 = 저장폴더 / "기상청_단기예보_익일_모델용.csv"

    인증키 = 인증키읽기()
    전체 = 날짜목록(인수.start, 인수.end)
    완료 = 완료목록읽기(진행경로)
    대상 = [날짜 for 날짜 in 전체 if 날짜 + 인수.base_time not in 완료]
    print(f"전체 {len(전체):,}일 중 {len(대상):,}일 수집 예정 (완료 {len(완료):,}일)")
    print(f"격자: ({격자_가로}, {격자_세로}), 발표시각: {인수.base_time[:2]}:{인수.base_time[2:]}")

    자료없음연속 = 0
    for 번호, 발표일 in enumerate(대상, 1):
        항목 = 요청(인증키, 발표일, 인수.base_time)
        표 = 표준화(항목, 발표일, 인수.base_time)
        if 표.empty:
            자료없음연속 += 1
            print(f"  자료 없음: {발표일} {인수.base_time}")
            # 오래된 첫 날짜부터 계속 비어 있으면 불필요한 710회 호출을 막는다.
            if 번호 <= 3 and 자료없음연속 >= 3:
                raise RuntimeError(
                    "과거 발표분 3일 연속 자료 없음입니다. 이 서비스의 과거 보관기간 밖일 "
                    "가능성이 큽니다. 과거 학습자료는 기상청 API허브에서 수집해야 합니다."
                )
            continue

        자료없음연속 = 0
        키 = 발표일 + 인수.base_time
        표.to_csv(일별폴더 / f"{키}.csv", index=False, encoding="utf-8-sig")
        완료.add(키)
        완료목록저장(진행경로, 완료)
        if 번호 % 10 == 0 or 번호 == len(대상):
            print(f"  진행 {번호:,}/{len(대상):,} ({발표일})")
        time.sleep(max(0, 인수.sleep))

    원본행, 모델행 = 통합저장(일별폴더, 원본경로, 모델경로)
    print(f"원본 저장: {원본경로} ({원본행:,}행)")
    print(f"모델용 저장: {모델경로} ({모델행:,}시간)")


if __name__ == "__main__":
    main()
