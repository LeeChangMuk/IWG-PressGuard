# KAMP AI 플랫폼(dl.kamp-ai) 학습 실행 결과

AI-PaaS에서 이 소스의 학습 알고리즘(PressGuard_HGB_v1.2)을 실행환경 pressguard_py312(Python 3.12.15, t3qai_client 1.1.2)로 학습한 결과를 플랫폼에서 내려받은 그대로 보관했습니다.

- 학습 시간: 2026-10-07 10:41:16 ~ 10:42:39 (1분 23초), 상태 완료
- platform_check.txt: 시험 4,225행의 판정·경보가 제출 결과(results/predictions_test.csv)와 동일, 점수 최대 차이 2.2e-16
- model_platform.json: 플랫폼에서 학습된 모델. 제출 모델(pumpguard/model.json)과 구조가 같고 수치 최대 차이 4.4e-16
  (플랫폼 서버의 부동소수점 끝자리 차이로 파일 바이트만 다름 → platform_check.txt 두 번째 줄 'False'의 의미)
- metrics.json · MODEL_RESULTS.md · selection_protocol.json · predictions_test_platform.csv: 플랫폼 학습이 만든 지표·예측 결과
- train_platform.log: 플랫폼 학습 로그(오류·경고 없음)

이후 같은 모델을 추론 서비스로 배포해, 시험 이상 구간 12행 입력에 대해 WARMUP 2 → WATCH 1 → ALARM 9, 경보 진입 1건을 확인했습니다.

## v1.2.1 패키지와의 관계
- 이 폴더는 v1.2.0 알고리즘 패키지로 플랫폼에서 실행한 기록입니다. platform_check.txt 두 번째 줄의 문구는 당시 래퍼 형식이며, 현재 `train.py` 래퍼는 같은 비교를 '수치 동일(|diff| ≤ 1e-9) / 바이트 동일' 두 값으로 기록합니다.
- v1.2.1에서 바뀐 것은 경보 근거 표시(오프셋 불변 근거 신호, 하부 진동 정지 안내, 경보 진입 시점 SHAP), 분석 스크립트의 OOF 점수 재읽기 정밀도, 추론 서비스의 입력 오류 처리입니다. 학습 파이프라인(`pressguard_pipeline.py` = `train.py`, `protocol.py`)과 모델·임계값·점수·판정·경보는 그대로이므로 위 결과가 유효합니다. 예측 CSV의 `reason_standard` 열 문구만 v1.2.1에서 달라집니다.
