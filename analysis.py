
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf


# --------------------------------------------------
# 1. 프로젝트 설정
# --------------------------------------------------
TICKER = "005930.KS"
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"  # 종료일은 미포함
COMPANY = "삼성전자"

DATA_DIR = Path("data")
IMAGE_DIR = Path("images")
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "samsung_stock_2023_2024.csv"

plt.rcParams["axes.unicode_minus"] = False
# 한글 폰트는 운영체제에 따라 자동 선택되지 않을 수 있습니다.
import platform

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"


# --------------------------------------------------
# 2. 데이터 수집
# --------------------------------------------------
def download_data():
    """Yahoo Finance에서 삼성전자 일별 주가를 수집합니다."""
    ticker = yf.Ticker(TICKER)
    raw = ticker.history(
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if raw.empty:
        raise RuntimeError(
            "데이터를 가져오지 못했습니다. 인터넷 연결과 "
            "Yahoo Finance 데이터 접근 상태를 확인하세요."
        )

    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "Date"
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [c for c in required if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")

    # 배당을 고려한 수익률 분석에는 Adj Close를 우선 사용합니다.
    # 조정 종가가 없으면 Close를 사용하되 보고서에 기록합니다.
    if "Adj Close" in raw.columns:
        raw["AnalysisPrice"] = raw["Adj Close"]
        price_label = "조정 종가(Adj Close)"
    else:
        raw["AnalysisPrice"] = raw["Close"]
        price_label = "종가(Close, 조정 종가 미제공)"

    raw = raw.loc[
        (raw.index >= START_DATE) & (raw.index < END_DATE)
    ].copy()

    # 필수 가격 또는 거래량이 결측인 행은 분석에서 제외합니다.
    before = len(raw)
    raw = raw.dropna(subset=["Close", "AnalysisPrice", "Volume"])
    dropped_rows = before - len(raw)

    if len(raw) < 100:
        raise ValueError(
            f"유효 데이터가 {len(raw)}개뿐입니다. "
            "최소 100개가 필요합니다."
        )

    return raw, price_label, dropped_rows


# --------------------------------------------------
# 3. 전처리 및 파생 변수
# --------------------------------------------------
def prepare_data(raw):
    df = raw.copy()

    df["MA20"] = df["AnalysisPrice"].rolling(
        window=20, min_periods=20
    ).mean()
    df["MA60"] = df["AnalysisPrice"].rolling(
        window=60, min_periods=60
    ).mean()

    # 전 거래일 대비 수익률. 첫 행은 비교 대상이 없어 결측입니다.
    df["DailyReturn"] = df["AnalysisPrice"].pct_change() * 100

    # 연환산하지 않은 20거래일 이동 표준편차(% 단위)
    df["Volatility20"] = (
        df["DailyReturn"].rolling(20, min_periods=20).std()
    )

    # 월별 수익률은 각 달의 마지막 관측 가격으로 계산합니다.
    monthly_price = df["AnalysisPrice"].resample("ME").last()
    monthly_return = monthly_price.pct_change() * 100

    monthly = pd.DataFrame({
        "Month": monthly_price.index.strftime("%Y-%m"),
        "MonthEndPrice": monthly_price.values,
        "MonthlyReturnPct": monthly_return.values,
    }).dropna(subset=["MonthlyReturnPct"])

    # 이상치 자동 삭제 대신, 평균에서 3표준편차를 벗어난
    # 일별 수익률을 '추가 검토 대상'으로 표시합니다.
    returns = df["DailyReturn"].dropna()
    mean_return = returns.mean()
    std_return = returns.std()

    if std_return > 0:
        df["ReviewFlag"] = (
            (df["DailyReturn"] - mean_return).abs() > 3 * std_return
        )
    else:
        df["ReviewFlag"] = False

    return df, monthly


# --------------------------------------------------
# 4. 시각화 3개
# --------------------------------------------------
def create_charts(df, monthly):
    # 그래프 1: 전체 주가 추이
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["AnalysisPrice"], linewidth=1.4)
    ax.set_title("삼성전자 주가 추이 (2023~2024)")
    ax.set_xlabel("날짜")
    ax.set_ylabel("가격 (KRW)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "01_price_trend.png", dpi=160)
    plt.close(fig)

    # 그래프 2: 20일 및 60일 이동평균
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df.index, df["AnalysisPrice"],
        label="조정 종가", alpha=0.45, linewidth=1
    )
    ax.plot(df.index, df["MA20"], label="20일 이동평균", linewidth=1.5)
    ax.plot(df.index, df["MA60"], label="60일 이동평균", linewidth=1.5)
    ax.set_title("삼성전자 주가와 이동평균")
    ax.set_xlabel("날짜")
    ax.set_ylabel("가격 (KRW)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "02_moving_average.png", dpi=160)
    plt.close(fig)

    # 그래프 3: 월별 수익률
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [
        "#25855A" if value >= 0 else "#C94C4C"
        for value in monthly["MonthlyReturnPct"]
    ]
    ax.bar(monthly["Month"], monthly["MonthlyReturnPct"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("삼성전자 월별 수익률")
    ax.set_xlabel("월")
    ax.set_ylabel("수익률 (%)")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "03_monthly_return.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------
# 5. 실제 계산 결과로 리포트 작성
# --------------------------------------------------
def write_report(df, monthly, price_label, dropped_rows):
    first_price = df["AnalysisPrice"].iloc[0]
    last_price = df["AnalysisPrice"].iloc[-1]
    total_change = (last_price / first_price - 1) * 100

    returns = df["DailyReturn"].dropna()
    best_date = returns.idxmax()
    worst_date = returns.idxmin()
    best_return = returns.loc[best_date]
    worst_return = returns.loc[worst_date]

    monthly_best = monthly.loc[monthly["MonthlyReturnPct"].idxmax()]
    monthly_worst = monthly.loc[monthly["MonthlyReturnPct"].idxmin()]

    overall_volatility = returns.std()
    avg_volume = df["Volume"].mean()
    flagged = df.loc[df["ReviewFlag"], ["DailyReturn"]]

    missing_before = int(
        df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum()
    )
    duplicate_dates = int(df.index.duplicated().sum())

    # 관찰(Fact)과 해석(Hypothesis)을 구분합니다.
    report = f"""# 삼성전자 2023~2024년 주가 시계열 분석 리포트

## 1. 분석 주제 및 선정 이유

삼성전자 일별 주가를 분석하여 전체 추세, 단기·장기 이동평균,
월별 수익률과 변동성을 확인한다. 주가 시계열 분석의 기초 기법을
적용하고, 그래프에서 관찰한 사실과 가능한 해석을 구분하는 것이 목적이다.

- 종목: 삼성전자 ({TICKER})
- 분석 기간: {df.index.min().date()} ~ {df.index.max().date()}
- 가격 기준: {price_label}
- 데이터 출처: [Yahoo Finance 과거 주가](https://finance.yahoo.com/quote/005930.KS/history/)
- 수집 라이브러리: yfinance

## 2. 분석 질문

1. 분석 기간의 주가는 전체적으로 상승했는가, 하락했는가?
2. 20일 이동평균과 60일 이동평균은 추세 변화에 어떤 정보를 주는가?
3. 월별 수익률은 얼마나 달랐으며, 변동성이 컸던 구간은 언제인가?
4. 가장 큰 일별 상승·하락은 언제였으며, 추가 조사가 필요한가?

## 3. 데이터 설명 및 정제

- 유효 데이터 수: {len(df)}개 거래일
- 실제 분석 기간: {df.index.min().date()} ~ {df.index.max().date()}
- 주요 컬럼: Open, High, Low, Close, Adj Close(제공 시), Volume
- 파생 컬럼: AnalysisPrice, MA20, MA60, DailyReturn, Volatility20, ReviewFlag
- 정제: 날짜 중복 제거, 날짜 오름차순 정렬, 필수 가격·거래량 결측 행 제외
- 결측 처리로 제외한 행: {dropped_rows}개
- 분석 후 필수 컬럼 결측 합계: {missing_before}개
- 중복 날짜 수(정제 후): {duplicate_dates}개
- 이상치: 일별 수익률이 평균에서 3표준편차를 벗어나면 검토 대상으로 표시한다.
  자동 삭제하지 않는다. 실제 급등락일 수 있기 때문이다.
- 거래일 사이의 주말·공휴일은 결측치로 간주하지 않는다.

## 4. 시각화 결과

### 4.1 전체 주가 추이

![삼성전자 주가 추이](images/01_price_trend.png)

가격의 전체적인 흐름과 큰 변동 구간을 확인한다.

### 4.2 20일·60일 이동평균

![이동평균 비교](images/02_moving_average.png)

20일 이동평균은 상대적으로 단기 흐름, 60일 이동평균은
상대적으로 완만한 흐름을 보여준다. 교차는 후행 신호이며 단독 매매 신호로
해석하지 않는다.

### 4.3 월별 수익률

![월별 수익률](images/03_monthly_return.png)

월별 수익률의 부호와 크기를 비교한다. 월별 관측 수가 제한적이므로
반복되는 계절성이 입증되었다고 단정하지 않는다.

## 5. 인사이트

### 인사이트 1. 분석 기간의 전체 변화

- **관찰(Fact):** 첫 관측 가격은 {first_price:,.2f}원, 마지막 관측 가격은
  {last_price:,.2f}원이며, 두 관측값 사이의 변화율은 {total_change:.2f}%다.
- **해석(Hypothesis):** 이 수치는 선택한 기간의 시작과 끝을 비교한 결과다.
  기간 중의 경로와 위험을 모두 설명하지는 못한다.
- **행동(Action):** 가격 추이 그래프 및 월별 수익률과 함께 읽고,
  다른 기간에서도 결과가 유지되는지 비교한다.

### 인사이트 2. 일별 수익률의 극단값

- **관찰(Fact):** 가장 큰 일별 상승은 {best_date.date()}에 {best_return:.2f}%,
  가장 큰 일별 하락은 {worst_date.date()}에 {worst_return:.2f}%였다.
- **해석(Hypothesis):** 해당 날짜는 시장·기업 관련 뉴스와 대조할 가치가 있다.
  가격 데이터만으로 변동 원인을 확정할 수 없다.
- **행동(Action):** 해당 날짜의 공시, 실적 발표, 산업 뉴스 및 시장 전체 움직임을
  추가 확인한다.

### 인사이트 3. 월별 수익률 차이

- **관찰(Fact):** 가장 높은 월별 수익률은 {monthly_best['Month']}의
  {monthly_best['MonthlyReturnPct']:.2f}%이고, 가장 낮은 월별 수익률은
  {monthly_worst['Month']}의 {monthly_worst['MonthlyReturnPct']:.2f}%였다.
- **해석(Hypothesis):** 월별 성과 차이는 특정 시점의 시장 환경이나 개별 이슈와
  관련될 수 있지만, 이 결과만으로 계절성을 주장할 수 없다.
- **행동(Action):** 월별 수익률과 변동성을 함께 비교하고, 더 긴 기간의 자료로
  재검증한다.

### 인사이트 4. 수익률 변동성

- **관찰(Fact):** 일별 수익률의 표준편차는 {overall_volatility:.3f}%였다.
  평균 거래량은 {avg_volume:,.0f}주다.
- **해석(Hypothesis):** 표준편차는 수익률의 변동 정도를 요약하지만,
  손실 가능성이나 미래 변동성을 직접 보장하지 않는다.
- **행동(Action):** 20거래일 이동 변동성 그래프를 추가하거나,
  상승·하락 구간을 나누어 변동성을 비교한다.

### 추가 점검 대상

평균에서 3표준편차 이상 벗어난 일별 수익률은 {len(flagged)}건이다.
이 값들은 오류로 단정하지 않고 원자료와 해당 날짜의 뉴스를 확인해야 한다.

## 6. 결론 및 한계점

### 결론

이번 분석은 삼성전자 주가의 시작·종료 가격 변화, 이동평균, 일별 수익률,
월별 수익률 및 변동성을 이용해 시계열의 추세와 변동을 요약한다.
실제 결론은 위의 계산 결과와 그래프를 함께 검토해 작성한다.

### 한계점

1. 주가 데이터만으로 가격 변화의 원인을 인과적으로 증명할 수 없다.
2. 배당 조정 가격을 사용하더라도 투자자의 세금·거래비용은 반영하지 않는다.
3. 이동평균은 과거 데이터를 요약하는 후행 지표다.
4. 월별 관측치는 적어 계절성을 확정하기 어렵다.
5. Yahoo Finance 데이터는 수정되거나 일시적으로 제공되지 않을 수 있다.
6. 이 분석은 교육 목적이며 투자 추천이 아니다.

## 7. AI 사용 로그

- **사용 작업:** 분석 질문 구성, Python 데이터 처리·시각화 코드 설계,
  인사이트의 관찰/해석/행동 구조 및 리포트 초안 작성
- **사용 이유:** 반복 작업을 줄이고 분석 절차를 체계화하기 위해
- **검증 방법:** 스크립트를 직접 실행하고 데이터 수, 날짜 범위, 결측치,
  수익률 계산식과 그래프를 확인한다. 극단값은 원자료와 대조하며,
  AI가 제안한 해석을 사실로 간주하지 않는다.
- **투명성:** 수치와 인사이트의 정량적 근거는 스크립트 실행 결과에서 생성한다.
  AI가 실제 실행하지 않은 계산을 검증했다고 주장하지 않는다.

## 8. 재현 방법

프로젝트 루트에서 다음 명령을 실행한다.

```bash
python -m pip install -r requirements.txt
python analysis.py
```

성공하면 `data/`, `images/`, `REPORT.md`가 생성된다.
네트워크 접속과 데이터 제공 상태에 따라 수집이 실패할 수 있다.

## 9. 출처 및 라이선스 주의

- Yahoo Finance: https://finance.yahoo.com/quote/005930.KS/history/
- yfinance 문서: https://ranaroussi.github.io/yfinance/
- 데이터는 출처의 이용약관 및 사용 조건을 확인하고 사용한다.
  데이터 제공자의 허가 없이 원자료를 재배포하지 않는다.
"""
    Path("REPORT.md").write_text(report, encoding="utf-8")


# --------------------------------------------------
# 6. 실행
# --------------------------------------------------
def main():
    raw, price_label, dropped_rows = download_data()
    df, monthly = prepare_data(raw)

    # 원자료에 파생 변수를 더한 분석용 CSV 저장
    export_df = df.copy()
    export_df.index.name = "Date"
    export_df.to_csv(CSV_PATH, encoding="utf-8-sig")

    create_charts(df, monthly)
    write_report(df, monthly, price_label, dropped_rows)

    print("분석 완료")
    print(f"데이터 포인트: {len(df)}")
    print(f"기간: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"CSV: {CSV_PATH}")
    print("그래프: images/01_price_trend.png")
    print("        images/02_moving_average.png")
    print("        images/03_monthly_return.png")
    print("리포트: REPORT.md")


if __name__ == "__main__":
    main()
