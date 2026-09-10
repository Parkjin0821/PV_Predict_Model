"""핵심 성능기준과 계층 합계 일치를 자동 검증한다."""

from __future__ import annotations

import json

import pandas as pd

from pv_pipeline import ROOT


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    outputs = ROOT / "outputs"
    ultra = pd.read_csv(outputs / "초단기_앙상블" / "최종시험_공통표본_성능표.csv")
    hourly = pd.read_csv(outputs / "1시간_앙상블" / "공식24_48시간_공통표본_성능표.csv")
    daily = pd.read_csv(outputs / "일간_계층조정" / "최종시험_일간성능표.csv")
    daily_prediction = pd.read_parquet(outputs / "일간_계층조정" / "최종시험_일간예측.parquet")
    hourly_adjusted = pd.read_parquet(outputs / "일간_계층조정" / "최종시험_계층조정_1시간상세.parquet")

    ultra = ultra[ultra["모델"].eq("자동가중앙상블")]
    hourly = hourly[hourly["모델"].eq("자동가중앙상블")]
    daily = daily[daily["모델"].eq("계층조정_일간")]
    require(len(ultra) == 13, "초단기 예측수평이 13개가 아닙니다.")
    require(len(hourly) == 25, "1시간 공식 예측수평이 25개가 아닙니다.")
    require(len(daily) == 1, "일간 계층조정 성능행이 없습니다.")
    for name, frame in [("초단기", ultra), ("1시간", hourly), ("일간", daily)]:
        require(bool((frame["정확도_pct"] >= 70).all()), f"{name} 정확도 기준 미달")
        require(bool((frame["결정계수"] >= 0.70).all()), f"{name} 결정계수 기준 미달")

    sums = hourly_adjusted.groupby("예측대상일", as_index=False)["계층조정_1시간예측_kW"].sum()
    sums = sums.rename(columns={"계층조정_1시간예측_kW": "시간별합계_kWh"})
    compare = daily_prediction.merge(sums, on="예측대상일", how="inner", validate="one_to_one")
    difference = (compare["계층조정_일간예측_kWh"] - compare["시간별합계_kWh"]).abs()
    require(len(compare) == len(daily_prediction), "일간 예측 일부가 시간별 상세와 연결되지 않습니다.")
    require(float(difference.max()) <= 1e-6, "시간별 합계와 일간 총량이 일치하지 않습니다.")

    result = {
        "초단기_통과수평": len(ultra),
        "초단기_최저정확도_pct": float(ultra["정확도_pct"].min()),
        "초단기_최저결정계수": float(ultra["결정계수"].min()),
        "1시간_통과수평": len(hourly),
        "1시간_최저정확도_pct": float(hourly["정확도_pct"].min()),
        "1시간_최저결정계수": float(hourly["결정계수"].min()),
        "일간_시험일수": len(daily_prediction),
        "계층합계_최대차이_kWh": float(difference.max()),
        "검증결과": "통과",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
