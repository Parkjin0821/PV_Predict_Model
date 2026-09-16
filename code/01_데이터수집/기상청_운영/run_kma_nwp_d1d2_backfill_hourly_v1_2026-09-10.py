# -*- coding: utf-8 -*-
"""4지역(광주·부안·김제·영광) D+1+D+2(+48h) 구엔드포인트 710일 백필을
매시간 자동 재개한다.

09-10 사용자 요청: "00:00 되면 알아서 돌리고, 안되면 1시간 단위로
체킹해서 자동 진행". 별도 재시도 로직을 새로 만들지 않고, 기존
`collect_kma_nwp_d1d2_extended_v1_2026-09-08.py`가 이미 가진
"already_complete 스킵" 재개형 설계를 그대로 활용한다 - 이 스크립트를
매시간(00:00 자정 리셋 직후 슬롯 포함) 실행하면:
  - 아직 안 끝난 지역/날짜는 이어서 채워짐
  - 그날 일일한도(typ01, 20,000회 공유)가 아직 안 풀렸으면 이번 시간엔
    호출이 실패/0건일 뿐, 다음 시간 슬롯에 자동 재시도됨
  - 4지역 전부 완료되면 이후 호출은 전부 already_complete로 즉시 스킵
    (API 낭비 없음, 굳이 끄지 않아도 무해)

일일 예산은 4지역 합쳐 4000회로 제한(typ01 공유한도 20,000회 중
ASOS/GRID/D+1 등 다른 일일 수집에도 여유를 남기기 위함).
"""

from __future__ import annotations

import os
import subprocess
import sys
import socket
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
HERE = Path(__file__).resolve().parent
PYTHON = r"C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
# ★09-15 변경(2키 병행)★: 구경로(nph_sun_nwp_txt) 대신 NC 확장 수집기를
# 쓴다. 이유: (1) 새로 발급한 NC 전용키는 NC 엔드포인트 2종만 승인돼 있어
# 구경로를 못 쓴다 (2) 오늘 성공한 백필은 전부 NC 경로였고 status도
# complete_nc로 통일돼 있다. NC 확장 수집기는 이 파일(구경로 스크립트)을
# monkeypatch로 재사용하므로 --region/--start-date/--hours-step 등 인자는
# 완전히 동일하다.
COLLECTOR = HERE / "collect_kma_nwp_nc_d1d2_extended_v1_2026-09-09.py"

# ★★09-16 변경(날짜별 이중 경로)★★
# 09-15에 전 구간을 NC 경로로 통일했더니, 백필이 최신→과거로 내려가다
# **2026-04-14 벽**에 부딪혀 영구 정지했다(매 실행이 2호출 만에
# "연속 날짜 실패 2회" 조기중단 → 다음 실행도 동일 → 무한 무진전 루프).
# 실측 원인: **KIM NC 아카이브 보존기간이 약 5개월**이라 그 이전 날짜는
# 영구적으로 `file is not exist`다(2025-02~2026-03 요청 1,696건 전부 실패,
# 성공 0일). 반면 구경로(nph_sun_nwp_txt)는 2025-06-15 프로브에서
# HTTP 200 + 실제값 16개를 돌려줬다 - **과거분은 구경로로만 받을 수 있다.**
#
# 그래서 "수집기를 구경로로 되돌리는" 게 아니라, 러너가 **날짜에 따라
# 수집기를 갈라 호출**한다(최신 구간은 계속 NC로 받아 경로·품질이 안 섞이게).
NC_CUTOVER_DATE = "2026-04-16"          # 이 날짜 포함 이후 = KIM NC 경로
LEGACY_END_DATE = "2026-04-15"          # 이 날짜 포함 이전 = 구경로
LEGACY_COLLECTOR = HERE / "collect_kma_nwp_d1d2_extended_v1_2026-09-08.py"

# ★중요 제약★: NC 전용키는 NC 엔드포인트 2종만 승인돼 있어 **구경로를 못 쓴다**.
# 따라서 구경로 leg은 지역과 무관하게 항상 구키(default)로 돌린다.
# 그 결과 구경로 leg 4지역이 전부 구키 하나에 몰리므로 예산을 보수적으로 잡는다.
#   구키 하루 사용량(최악) = (구경로 150×4지역 + NC 50×2지역) × 24시간
#                          = (600 + 100) × 24 = 16,800회 + 실시간 약 1,000회
#                          = 약 17,800회 < 20,000회 한도
# NC leg은 이미 2026-04-16~현재가 4지역 모두 사실상 완결이라 대부분
# already_done 스킵(0회 호출)이므로 실제 소비는 이보다 훨씬 낮다.
# 사용자 지시대로 **1일 관찰 뒤 증액 판단**한다.
LEG_BUDGET = {"nc": 50, "legacy": 150}
# 구경로 leg은 아카이브에 원래 없는 날이 드문드문 있어도 런 전체가 죽으면
# 안 된다 - NC leg(2)보다 크게 잡아 몇 건 건너뛰고 계속 전진하게 한다.
LEG_MAX_CONSEC_FAIL = {"nc": 2, "legacy": 10}

# ★09-15 추가(Claude)★: D1D2 백필 전용 별도 authKey. 처음엔 환경변수를
# 영구 등록(setx)하는 방식으로 설계했는데, 그러면 이 값이 "KIM NC +24h/
# +48h 확장 수집"(collect_kma_nwp_nc_d1d2_extended_v1_2026-09-09.py)에도
# 새어 들어간다는 걸 발견했다 - 그 스크립트가 이 파일(COLLECTOR)을 그대로
# 재사용(monkeypatch)하기 때문. 그래서 영구 환경변수 대신, 이 파일 하나가
# 읽는 로컬 전용 키파일로 바꿔 subprocess 호출 시에만 한정 주입한다 -
# NC 확장 수집·다른 어떤 스크립트도 이 파일을 안 읽으므로 절대 안 새어감.
# 키 값은 코드에 절대 넣지 않는다 - 사용자가 이 파일에 직접 붙여넣는다.
BACKFILL_AUTH_KEY_FILE = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\d1d2_backfill_authkey_local.txt"
)


# ★09-15 추가(2키 병행)★: NC 전용키 파일. 지역별로 어느 키를 쓸지
# REGION_KEY에서 정한다. 키 값 자체는 코드에 절대 안 넣고 파일에서만 읽는다.
NC_AUTH_KEY_FILE = Path(r"C:\Users\u-cube\JIN\태양광 발전\nc_d1d2_authkey_local.txt")

# 지역별 키 배정 - 두 키의 하루 한도(각 20,000회)를 동시에 활용해 백필
# 처리량을 2배로 올린다. "default"는 환경변수를 안 넣어 구키(기본
# AUTH_KEY)를 쓰는 경우.
#   - 구키: 실시간(ASOS·GRID·NWP D+1) 약 1,000회/일만 쓰므로(09-15 GRID
#     중복호출 제거 후) 나머지가 통째로 남는다. NC 지점조회 응답은 1KB
#     수준이라 5GB 용량 한도에도 사실상 영향 없음(하루 14,400회≈14MB).
#   - NC키: 실시간 NC 수집이 960회/일만 쓴다.
# ★09-16 변경(사용자 지시)★: 부안·김제도 NC leg를 NC 전용키로 이동.
# 실측(09-16): 백필 legacy leg(4지역 전부 구키, 750회×4=3,005회)+
# 실시간(GRID 336+ASOS 224+D+1NWP 232=792회) = 구키 약 3,797/20,000회
# (19%)만 사용 중 - 실시간에 지장 없음을 이미 확인했지만, 부안·김제
# NC leg(구키 소비분, 오늘은 이미 따라잡아 0~5회뿐이라 영향 미미)까지
# 구키에서 완전히 빼서 "백필이 실시간 대시보드 수집을 방해할 가능성"
# 자체를 원천 차단한다. NC전용키는 오늘 4,717/20,000(23.6%)로도 여유
# 충분(4지역 NC leg 전부 몰아도 최대 추가 2,400/일 수준, 20,000엔
# 안전하게 못 미침).
REGION_KEY = {"광주": "nc", "영광": "nc", "부안": "nc", "김제": "nc"}


def _load_backfill_env(region: str, leg: str = "nc") -> dict[str, str]:
    """지역별 지정 키를 subprocess 환경에만 한정 주입한다.

    ★09-16★: leg="legacy"(구경로)일 때는 지역 배정을 무시하고 항상 구키를
    쓴다 - NC 전용키가 구경로 엔드포인트에 승인돼 있지 않기 때문(쓰면 403).
    """
    env = dict(os.environ)
    env.pop("KMA_AUTH_KEY_BACKFILL", None)  # 상속 오염 방지
    which = "default" if leg == "legacy" else REGION_KEY.get(region, "default")
    if which == "nc":
        try:
            key = NC_AUTH_KEY_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            key = ""
        if key:
            env["KMA_AUTH_KEY_BACKFILL"] = key
        return env
    # default: 백필 전용키 파일이 채워져 있으면 그걸, 아니면 구키(무설정).
    try:
        key = BACKFILL_AUTH_KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    if key:
        env["KMA_AUTH_KEY_BACKFILL"] = key
    return env
LOG_DIR = HERE / "logs" / "d1d2_backfill_hourly"

REGIONS = ["광주", "부안", "김제", "영광"]
START_DATE = "2024-08-25"
# ★09-15 변경★: 기존엔 2026-08-04로 고정돼 있어 그 이후 과거 결측일
# (예: 09-02, 09-10 등)은 이 백필이 아예 닿지 못했다(실제로 그 구간은
# 그때그때 수동 실행으로 메워왔음). 어제까지로 확장해 과거 전 구간을
# 한 경로로 처리한다. 이미 완결된 날짜는 already_done으로 0회 스킵되므로
# 범위를 넓혀도 호출 낭비가 없다(오늘 실측 검증 완료).
END_DATE = (datetime.now(tz=KST).date() - timedelta(days=1)).isoformat()
# ★09-14 추가★: 광주는 09-09에 늦게 합류해 710일 목표 대비 완료일수가
# 나머지 3지역보다 크게 뒤처져 있다(09-14 13시 실측: 광주 95일(13.4%)
# vs 부안 234일·김제 233일·영광 235일(약 33%) - 약 139일 결손, 09-09
# 합류 시점 이후 지금까지 동일 예산으로 나란히 전진만 해서 격차가
# 전혀 안 좁혀지고 있었음). 4지역 API 할당량은 지역별로 별도가 아니라
# 이 상품(NWP D1D2) 하나에 계정(authKey) 단위로 공유되는 것으로 보여
# (typ01 산하 다른 상품은 별개 - 09-13 항목 참고), 4지역 합계 예산
# 400/시간(09-11 확정치, 일 최대 9,600회)은 그대로 두고 지역별
# 배분만 광주에 몰아준다 - 병렬 실행은 유지해 부안·김제·영광도
# 매시간 계속 진행되게 하되(완전 순차로 바꾸면 그 3곳이 4시간에
# 한 번만 갱신되어 오히려 손해), 광주 몫만 키운다.
# 광주가 다른 지역과 완료일수가 비슷해지면(약 700일 근처) 이 재배분을
# 원래대로(4지역 동일 100씩)로 되돌릴 것 - AGENTS.md 09-14 항목 참고.
# ★09-15 재조정(2키 병행 + 3시간 해상도)★: 사용자 목표 "약 5일 내 완주".
# 남은 1,884 지역-일 × 80회(3시간 해상도) = 150,720회. 두 키를 동시에 쓰고
# 지역당 300회/시간(4지역 합 1,200회/시간 = 28,800회/일)이면 약 5.2일.
#   키별 하루 사용량: 구키 300×2지역×24=14,400회(+실시간 약1,000) = 15,400회
#                    NC키 300×2지역×24=14,400회(+실시간 960)   = 15,360회
#   → 두 키 모두 20,000회 한도 대비 약 4,600회 여유를 항상 남긴다.
# ★실시간 수집 보호가 최우선★ - 여유가 부족해지면 이 숫자부터 낮출 것.
# ★09-16 폐기★: 날짜별 이중 경로로 바뀌면서 예산은 leg 단위(LEG_BUDGET)로
# 잡는다. 이 상수는 더 이상 쓰이지 않음(기록용으로만 남김).
PER_REGION_BUDGET = {"광주": 300, "부안": 300, "김제": 300, "영광": 300}
# ★09-14 수정(2차)★: 최초 제안 220/60/60/60은 나머지 3지역을 -40%까지
# 깎아 사용자가 "이러면 나머지가 아예 안 되잖아"라고 지적 - 15콜/일
# 기준 환산하면 100/시간=약 6.7일/시간, 60/시간=약 4.0일/시간(-40%),
# 220/시간=약 14.7일/시간. 160/80/80/80(광주 +60%, 나머지 -20%,
# 각 약 5.3일/시간)으로 덜 공격적으로 재조정 - 나머지 3지역도 여전히
# 눈에 띄게 진행되면서 광주 캐치업 속도도 확보.
# - 위 사연 이전 09-11 기록: 350(1400/슬롯, 일 최대 33,600회)이던 걸
#   100씩 균등으로 낮춤. ASOS/GRID·NC라이브 등 "매일 최소 신선도 유지"가
#   필요한 경량 수집기(일 합계 ~1,000회)가 NC-D1D2-live(별도로
#   400→100/슬롯 조정, 일 최대 4,800회)와 합쳐도 굶지 않도록 캐치업
#   최댓값에 안전마진(최소 4,000회/일)을 남겨두기 위함 - 09-11 공유
#   authKey 할당량 소진 의심 사고 이후. 이번 재배분도 합계 400을 그대로
#   지켜 이 안전마진 원칙은 유지한다.
CONNECT_TIMEOUT_SECONDS = 5
REGION_PROCESS_TIMEOUT_SECONDS = 20 * 60
SHARED_LOCK_DIR = LOG_DIR.parent / "kim_nwp_d1d2_db_write.lockdir"
SHARED_LOCK_STALE_SECONDS = 25 * 60


def acquire_shared_lock(now: datetime) -> tuple[bool, str]:
    """NC 라이브와 백필이 같은 SQLite에 동시에 쓰지 않도록 원자 잠금."""
    try:
        SHARED_LOCK_DIR.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        try:
            age = now.timestamp() - SHARED_LOCK_DIR.stat().st_mtime
            if age > SHARED_LOCK_STALE_SECONDS:
                # 잠금 전용 marker만 지운 뒤 잠금 디렉터리를 재생성한다.
                marker = SHARED_LOCK_DIR / "owner.json"
                if marker.exists():
                    marker.unlink()
                SHARED_LOCK_DIR.rmdir()
                SHARED_LOCK_DIR.mkdir(parents=False, exist_ok=False)
            else:
                return False, f"공유 DB 잠금 사용 중(age={age:.0f}s)"
        except (OSError, FileExistsError) as exc:
            return False, f"공유 DB 잠금 획득 실패: {type(exc).__name__}: {exc}"
    marker = SHARED_LOCK_DIR / "owner.json"
    marker.write_text(json.dumps({"owner": "backfill", "started_at": now.isoformat()},
                                 ensure_ascii=False), encoding="utf-8")
    return True, "백필 공유 잠금 획득"


def release_shared_lock() -> None:
    try:
        marker = SHARED_LOCK_DIR / "owner.json"
        if marker.exists():
            marker.unlink()
        SHARED_LOCK_DIR.rmdir()
    except OSError:
        pass


def endpoint_reachable() -> tuple[bool, str]:
    """인증키를 전송하지 않고 API 호스트의 TCP 443 연결만 짧게 확인한다."""
    host = urlparse(src_url()).hostname
    if not host:
        return False, "collector URL hostname 확인 실패"
    errors = []
    for attempt in range(1, 3):
        try:
            with socket.create_connection((host, 443), timeout=CONNECT_TIMEOUT_SECONDS):
                return True, f"{host}:443 연결 성공({attempt}회차)"
        except OSError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return False, f"{host}:443 연결 2회 실패: {' | '.join(errors)}"


def src_url() -> str:
    """하위 수집기와 같은 모듈에서 실제 엔드포인트를 읽는다.

    ★09-15★ COLLECTOR를 NC 확장 수집기로 바꾸면서, 구경로 전용이던
    `module.src.URL` 참조가 AttributeError를 내지 않도록 방어한다.
    NC 확장본은 `nc.NC_URL`(NC 엔드포인트)을 갖고, 구경로본은 `src.URL`을
    갖는다. 둘 다 없으면 호스트만 확인하면 되므로 apihub 기본 URL로 폴백.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("d1d2_collector", COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    nc_mod = getattr(module, "nc", None)
    if nc_mod is not None and getattr(nc_mod, "NC_URL", None):
        return nc_mod.NC_URL
    src_mod = getattr(module, "src", None)
    if src_mod is not None and getattr(src_mod, "URL", None):
        return src_mod.URL
    return "https://apihub.kma.go.kr/"


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=KST)
    log_path = LOG_DIR / f"{now.strftime('%Y%m%d')}.log"
    lines = [f"=== {now.isoformat(timespec='seconds')} ==="]
    any_fail = False

    locked, lock_detail = acquire_shared_lock(now)
    if not locked:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {now.isoformat(timespec='seconds')} ===\n")
            fh.write(f"[shared_lock] {lock_detail}; 다음 정시에 재시도\n")
        print(lock_detail)
        return 0

    fh = log_path.open("a", encoding="utf-8")
    try:
        fh.write("\n".join(lines) + "\n")
        reachable, detail = endpoint_reachable()
        fh.write(f"[endpoint_preflight] reachable={reachable} detail={detail}\n")
        if not reachable:
            fh.write("API 연결 장애로 이번 슬롯을 즉시 종료; 다음 정시에 재시도\n\n")
            print(f"API 연결 사전점검 실패, 즉시 종료: {detail}")
            return 0
        def run_leg(region: str, leg: str) -> tuple[str, int | None, str, str, bool]:
            # ★09-16★: leg에 따라 수집기·날짜범위·예산·키가 전부 달라진다.
            # 완료된 날짜는 양쪽 다 run_status 기준 already_done 스킵(0회 호출)이라
            # 중복 수집은 없다.
            if leg == "legacy":
                collector, lo, hi = LEGACY_COLLECTOR, START_DATE, LEGACY_END_DATE
            else:
                collector, lo, hi = COLLECTOR, NC_CUTOVER_DATE, END_DATE
            cmd = [
                PYTHON, str(collector), "--region", region,
                "--start-date", lo, "--end-date", hi,
                "--live", "--max-api-calls", str(LEG_BUDGET[leg]),
                "--max-retries", "1", "--timeout-seconds", "10",
                "--max-consecutive-day-failures", str(LEG_MAX_CONSEC_FAIL[leg]),
                "--newest-first",
                # ★09-15 추가★: 과거 백필은 3시간 간격(하루 80개)으로 받는다.
                # 09-15 낮에 전역 해상도를 1시간(240개/일)으로 올렸더니 하루치
                # 비용이 3배가 되어 "날짜를 많이 채운다"는 백필 본래 목적의
                # 처리량이 1/3(20,000회 기준 250일분→83일분)로 떨어졌다.
                # 실시간 수집(30분 주기)은 인자 없이 기본값 1시간을 그대로 써서
                # 운영 유연성을 유지하고, 백필만 여기서 3시간으로 되돌린다.
                "--hours-step", "3",
            ]
            try:
                # ★09-15 수정★: text=True만 쓰면 자식 stdout을 시스템 기본
                # 인코딩(한국어 Windows=cp949)으로 디코딩하다가
                # UnicodeDecodeError가 나서 **수집기 출력이 통째로 유실**된다
                # (로그에 "--- 지역 (exit=0) ---"만 남고 내용이 비어 있던 원인).
                # 종료코드는 정상 수집되지만 진단 정보가 사라져 검증이 불가능해지므로
                # utf-8 + errors=replace로 고정한다.
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=REGION_PROCESS_TIMEOUT_SECONDS,
                    env=_load_backfill_env(region, leg),
                )
                label = f"{region}/{leg}"
                return label, proc.returncode, proc.stdout or "", proc.stderr or "", False
            except subprocess.TimeoutExpired as exc:
                return f"{region}/{leg}", None, exc.stdout or "", exc.stderr or "", True
            except Exception as exc:  # noqa: BLE001
                return f"{region}/{leg}", -1, "", f"{type(exc).__name__}: {exc}", False

        # 지역별 DB가 서로 다르므로 4개 지역을 병렬 처리한다. 전역 공유 잠금은
        # NC 라이브와의 동시 쓰기만 막고, 이 네 작업 사이의 병렬성은 허용한다.
        # ★09-16★: 같은 지역의 두 leg은 **같은 SQLite 파일**에 쓴다. 날짜
        # 범위가 안 겹쳐 행 충돌은 없지만 SQLite는 DB 단위로 쓰기 잠금을 걸어
        # "database is locked"가 날 수 있다 - 지역 내에서는 nc → legacy 순차,
        # 지역 사이는 기존대로 병렬(지역별 DB가 달라 안전)로 간다.
        def run_region(region: str) -> list[tuple[str, int | None, str, str, bool]]:
            return [run_leg(region, "nc"), run_leg(region, "legacy")]

        with ThreadPoolExecutor(max_workers=len(REGIONS)) as pool:
            futures = [pool.submit(run_region, region) for region in REGIONS]
            for future in as_completed(futures):
                # ★09-16★: run_region이 이제 leg 2건을 리스트로 돌려준다.
                for label, returncode, stdout, stderr, timed_out in future.result():
                    if timed_out:
                        fh.write(
                            f"--- {label} {REGION_PROCESS_TIMEOUT_SECONDS}초 초과로 중단; "
                            "다음 정시에 재시도 ---\n"
                        )
                        any_fail = True
                        continue
                    fh.write(f"--- {label} (exit={returncode}) ---\n")
                    fh.write(stdout[-4000:] + "\n")
                    if stderr:
                        fh.write("[stderr] " + stderr[-2000:] + "\n")
                    if returncode != 0:
                        any_fail = True
        fh.write("\n")
    finally:
        fh.close()
        release_shared_lock()

    print(f"완료(로그: {log_path}), any_fail={any_fail}")
    return 0  # 부분 실패도 다음 시간에 자동 재시도되므로 태스크 자체는 항상 성공 처리


if __name__ == "__main__":
    sys.exit(main())
