# 이관 전 로컬 검증 기록

플랫폼에 올리기 전에, 플랫폼이 쓰는 **실제 클라이언트(t3qai_client 1.2.3, PyPI)**로 학습·추론 모듈을 검증했습니다.

| 검증 | 방법 | 결과 |
|---|---|---|
| 학습 모듈 연동 | t3qai 플랫폼 API(train_load_path·train_start·train_load_param·train_save_result_metrics·train_finish)를 흉내 낸 로컬 서버에 실제 클라이언트로 접속, `python3 -m train` 실행 | 상태 코드 3(정상 종료), 지표 전달 확인 |
| 데이터 경로 | 데이터셋을 zip으로 둔 경우 자동 해제 + SHA-256 확인 | 통과 |
| 재현성 | 알고리즘 zip을 빈 폴더에 풀어 처음부터 학습 | model.json SHA-256 91fd124f…(제출본과 동일), 시험 F1 0.7004 |
| Plan B 버전 | numpy 2.2.6·pandas 2.2.3·scikit-learn 1.7.2 | 모델 바이트는 다르나 시험 4,225행 점수 차 0, 판정·경보 완전 일치 |
| 재학습 없는 배포 | train_mode=bundle | 제출 모델·결과로 정상 종료 |
| 추론 서비스 | 실제 t3qai_serving(uvicorn)으로 기동 → JSON(리스트·columns/values·모드 지정·문자열 숫자)과 파일 추론 요청 | 4가지 JSON 형식 모두 같은 결과, 현장 엔진과 점수 일치, 파일 추론은 CSV+요약 zip 반환 |

검증 환경: Python 3.12.3, Linux x64. 실제 플랫폼 버전·화면 구성은 매뉴얼(2022.11, v3.7.3)과 다를 수 있습니다.

## 추가 검증(t3qai_client 1.1.2)
- KAMP 운영팀 안내에 따라 1.1.2로 고정. 1.1.2는 플랫폼 API v2 경로를 사용(1.2.x는 v3).
- 학습 모듈: v2 API 흉내 서버로 전 과정 정상 종료(상태 3), 지표는 1.1.2 규격(accuracy·loss·step)과 분류 결과(predict_y·actual_y)로 전달, 모델 바이트 동일.
- 추론 모듈: 1.1.2 serving으로 리스트·columns/values(문자열)·작은따옴표 입력·파일 추론 모두 정상.
