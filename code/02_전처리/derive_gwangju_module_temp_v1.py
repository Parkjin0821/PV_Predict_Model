"""미래 모듈 표면온도·출력온도(셀 온도) 추정 — NOCT 계열 공식 기반.

AGENTS.md 6번 작업, 그리고 2026-08-19 "출력온도" 정의 확인 결과에 따른 산출.

## 왜 두 값으로 나누는가 (사용자 확인, 08-19)
사용자가 확인한 바, "설비요인" 목록의 모듈 표면온도와 출력온도는 별개 항목으로
요청된 것이며, 출력온도는 실측/직접계산이 불가능하므로 추정값으로 채우기로
했다. 이 스크립트는 PV 열모델에서 흔히 구분하는 두 단계를 그대로 대응시켰다:

1. **모듈(배면) 표면온도** `추정_모듈표면온도_C`
   Faiman(2008) 모델: T_module = T_air + G / (U0 + U1 * WS)
   - G: 전천일사량(W/m^2, 여기서는 수치예보 DSWRF), T_air: 기온(°C),
     WS: 풍속(m/s)
   - U0=25.0 W/m^2K, U1=6.84 W*s/m^3K: 개방형 거치(open-rack) 유리/실리콘
     셀/폴리머 시트 모듈의 문헌 표준계수(Faiman 2008; Sandia PVPMC 기본값).
     이 발전소의 실측 계수가 아니다 — 제조사 사양서·설치각도 등이 미확보라
     대체할 수 없어 문헌 표준값을 그대로 쓴다.

2. **출력온도(셀/운전온도)** `추정_출력온도_C`
   Sandia PV Array Performance Model(King et al. 2004)의 ΔT 보정:
   T_cell = T_module + (G/1000) * ΔT_cnd
   - ΔT_cnd=3.0°C: 개방형 거치 유리/셀/폴리머 시트 모듈의 문헌 표준값.
   - 셀은 모듈 배면보다 항상 약간 더 뜨거우며, 발전 출력에 실제로 영향을
     주는 온도는 배면온도가 아니라 이 셀온도이므로 "출력온도"에 대응시켰다.

## 명시적 한계 (AGENTS.md 원칙 4·5에 따른 공개)
- 이 계산은 보고서-강태형.hwp에 없는 자체 판단 방법론이다(NOCT/Faiman/Sandia
  문헌 공식 사용, 회사 제공 사양 아님).
- 실제 패널 설치각도·방위각·제조사 NOCT 실측값·정격 열손실계수가 없어
  지역·설비 보정 없이 문헌 표준계수를 그대로 적용했다. 정밀도가 필요하면
  설비대장 확보 후 계수를 교체해야 한다.
- 입력 일사량(DSWRF)은 실측이 아니라 수치예보모델 값이므로, 이 온도값도
  "미래 예보 기반 추정치"이며 실측 모듈온도가 아니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


GRID_DIR_TMP = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\grid_forecast_710d_collection_v3_2026-08-19"
)
GRID_DIR_WSD = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\grid_forecast_wsd_pop_710d_v4_2026-08-19"
)
NWP_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_nwp_solar_cloud_710d_v1_2026-08-19"
)

TMP_CSV = GRID_DIR_TMP / "광주_격자예보_3시간단위_TMP_SKY_REH_710일.csv"
WSD_CSV = GRID_DIR_WSD / "광주_격자예보_WSD_POP_710일.csv"
NWP_CSV = NWP_DIR / "광주_수치예보_DSWRF_DSWRFLX_DIFSWRF_TCDC_710일.csv"

OUT_CSV = NWP_DIR / "광주_추정_모듈온도_710일.csv"

FCST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

# Faiman(2008) 개방형 거치 유리/셀/폴리머 시트 기본계수
U0 = 25.0  # W/m^2K
U1 = 6.84  # W*s/m^3K

# Sandia PVPMC 개방형 거치 기본계수
DELTA_T_CND = 3.0  # °C, 1000 W/m^2 기준 셀-모듈 온도차


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"입력 파일이 없습니다 ({label}): {path}\n"
            "수집이 끝난 뒤 다시 실행하세요."
        )


def main() -> int:
    require(TMP_CSV, "격자예보 TMP")
    require(WSD_CSV, "격자예보 WSD")
    require(NWP_CSV, "수치예보 DSWRF")

    tmp = pd.read_csv(TMP_CSV, dtype={"발표일": str})
    wsd = pd.read_csv(WSD_CSV, dtype={"발표일": str})
    nwp = pd.read_csv(NWP_CSV, dtype={"발표일": str})

    merged = tmp[["발표일"]].merge(wsd[["발표일"]], on="발표일").merge(nwp[["발표일"]], on="발표일")
    if len(merged) < 710:
        print(f"경고: 세 파일의 공통 발표일이 710일 미만입니다 ({len(merged)}일).")

    out = pd.DataFrame({"발표일": merged["발표일"]})
    tmp_idx = tmp.set_index("발표일")
    wsd_idx = wsd.set_index("발표일")
    nwp_idx = nwp.set_index("발표일")

    for hour in FCST_HOURS:
        t_air = tmp_idx.loc[merged["발표일"], f"TMP_{hour:02d}h"].to_numpy()
        ws = wsd_idx.loc[merged["발표일"], f"WSD_{hour:02d}h"].to_numpy()
        g = nwp_idx.loc[merged["발표일"], f"DSWRF_{hour:02d}h"].to_numpy()

        t_module = t_air + g / (U0 + U1 * ws)
        t_cell = t_module + (g / 1000.0) * DELTA_T_CND

        out[f"추정_모듈표면온도_{hour:02d}h_C"] = t_module
        out[f"추정_출력온도_{hour:02d}h_C"] = t_cell

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUT_CSV} ({len(out)}행)")
    print("계수 출처: Faiman(2008) U0=25.0/U1=6.84, Sandia PVPMC ΔT_cnd=3.0 (개방형 거치 표준값, 실측 아님)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
