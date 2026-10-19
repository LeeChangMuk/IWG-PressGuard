# 프레스가드 라이트 v1.2 결과 요약

선정 모델: **hist_gb** (히스토그램 그래디언트 부스팅). 선정·보정·임계값은 개발 구간 블록 교차검증 OOF만 사용했다.

## 개발 OOF 후보 비교 (선정 근거)
|후보|구분|OOF F1|OOF 정밀도|OOF 재현율|OOF AP|블록 F1 최소~최대|시험 F1(보고 전용)|
|---|---|---:|---:|---:|---:|---:|---:|
|hist_gb ★|부스팅 앙상블|0.864|0.896|0.834|0.957|0.781~0.928|0.700|
|soft_vote|이종 앙상블|0.857|0.897|0.822|0.938|0.660~0.916|0.678|
|random_forest|배깅 앙상블|0.840|0.884|0.800|0.940|0.718~0.905|0.669|
|extra_trees|배깅 앙상블|0.823|0.918|0.746|0.918|0.659~0.874|0.679|
|logistic|선형|0.729|0.849|0.639|0.798|0.467~0.785|0.601|
|decision_tree|트리|0.682|0.768|0.613|0.761|0.625~0.719|0.521|
|rule_std|기준선|0.659|0.996|0.492|0.574|0.464~0.790|0.548|
|isolation_forest|기준선|0.567|0.789|0.443|0.601|0.405~0.697|0.402|

## 시험 1회 평가 (표준 모드)
- 표본 판정: F1 0.7004, 정밀도 0.6831, 재현율 0.7185, AP 0.8469, ROC-AUC 0.9813, Brier 0.01191
- TP 97, FP 45, FN 38, TN 4045 (판정 대상 3,976행 / 전체 4,225행; 준비 행은 무경보로 계산)
- 지속성 경보(3표본 중 2): F1 0.7674, 정밀도 0.8049, 재현율 0.7333, FP 24, 이상 구간 5/5 포착, 정상 409초 중 경보 진입 10건
- 구간 부트스트랩 95%: F1 0.537~0.796, AP 0.705~0.943 (이상 구간 5개 조건부)

## 운영 모드 (개발 OOF 규칙)
|모드|임계값|시험 정밀도|시험 재현율|시험 F1|지속경보 FP|이상 구간 포착|
|---|---:|---:|---:|---:|---:|---:|
|무인 감시|0.6908|0.840|0.659|0.739|7|5/5|
|표준 운전|0.3585|0.683|0.719|0.700|24|5/5|
|점검 선별|0.0639|0.451|0.815|0.580|81|5/5|

## v1.1 모델(ExtraTrees) 참고 비교 (보고 전용)
- 같은 판정 규칙: F1 0.6234 (정밀도 0.750, 재현율 0.533); v1.1 원래 규칙(준비 행 포함) F1 0.6276

## 한계
- Only two source recordings: label = file = date; cross-equipment / cross-date validation impossible.
- No fault onset or maintenance history: early-warning lead time cannot be measured.
- Five anomalous test segments from one 2.8-minute recording; intervals are conditional on these files.
- Units, product mix, speed/load and maintenance records absent.
- Calibrated probabilities reflect the in-dataset anomaly share (2.9%); re-calibrate for field prevalence.

## 재현
`python reproduce.py` (원본 해시 확인 → 학습·보정·선정·시험·분석·그림 일괄 실행)
