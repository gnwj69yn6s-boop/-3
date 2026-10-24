# 삼성전자 주가 시계열 분석

## 프로젝트 소개

2023~2024년 삼성전자 일별 주가 데이터를 활용해 가격 추세,
이동평균, 수익률, 월별 성과와 변동성을 분석하는 프로젝트입니다.

## 분석 목표

- 전체 주가 추세 확인
- 20일 및 60일 이동평균 비교
- 월별 수익률과 일별 변동성 분석
- 극단적인 수익률이 나타난 날짜 탐색
- 관찰한 사실과 가능한 해석 구분

## 기술 스택

- Python 3.10 이상
- pandas
- NumPy
- matplotlib
- yfinance

## 실행 방법

### 1. 저장소 복제

```bash
git clone <본인의-GitHub-저장소-URL>
cd samsung-stock-analysis
```

### 2. 가상환경 생성 및 활성화

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 라이브러리 설치

```bash
python -m pip install -r requirements.txt
```

### 4. 분석 실행

```bash
python analysis.py
```

실행 후 CSV 데이터, PNG 그래프 3개,
분석 결과를 반영한 REPORT.md가 생성됩니다.

## 데이터 출처

Yahoo Finance - Samsung Electronics (005930.KS)

https://finance.yahoo.com/quote/005930.KS/history/

수집 기간: 2023-01-01 ~ 2024-12-31

## 데이터 사용 주의

데이터 제공자의 이용약관 및 라이선스를 확인하세요.
원자료의 재배포 권한을 별도로 확인하지 않았다면
데이터 파일 대신 수집 코드를 공개하는 방식을 고려하세요.

## 한계점

- 주가만으로 가격 변화의 원인을 확정할 수 없습니다.
- 이동평균은 과거 가격을 요약하는 지표입니다.
- 분석 결과는 투자 추천이 아닙니다.
