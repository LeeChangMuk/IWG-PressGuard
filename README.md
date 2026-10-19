# IWG 프레스가드 라이트 v1.2.1 (모델 v1.2.0)

진동·전류 기반 프레스 유압펌프 이상 상태 분류와 오경보 분석을 위한 KAMP 제6회 일반부문 과제3 프로젝트입니다. 현장에서는 읽기 전용 CSV 시범운영 프로그램으로 사용합니다. 실제 고장 사전예측·수명 추정·자동 설비정지 기능은 구현하지 않았습니다.

## 제출 구성 (대회 양식 대응)

| 양식 항목 | 위치 | 내용 |
|---|---|---|
| 소스코드 (requirements.txt) | `*.py`, `pumpguard/`, `requirements.txt` | 학습·평가·분석 코드, 표준 라이브러리 추론 엔진, 고정 패키지 버전(Python 3.12) |
| 학습용 데이터 | `data/` | 대회 제공 원본 `press_data_normal.csv`(20,000행), `outlier_data.csv`(600행)와 SHA-256(`SOURCE_HASHES.json`) |
| README | `README.md` | 이 문서(구성·실행 방법·재현 절차) |
| 테스트데이터 예측결과 | `predictions_test.csv` | 고정 시험 구간 4,225행의 예측 결과(`results/predictions_test.csv`와 동일) |

`predictions_test.csv` 열 설명

| 열 | 의미 |
|---|---|
| source_file, original_csv_line | 원본 파일명과 원본 CSV 줄 번호(머리글 = 1행) |
| segment_id | 연속 관측구간 ID(0.15초 초과 공백에서 분할) |
| TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current | 원본 입력값 |
| label | 정답(0 정상, 1 이상). 예측에는 사용하지 않음 |
| n_history, judged | 구간 내 이력 표본 수, 판정 대상 여부(이력 3표본 이상 = 1) |
| score | Platt 보정 이상확률(데이터셋 이상 비율 기준) |
| decision_unmanned / decision_standard / decision_screening | 운영 모드별 표본 판정(1 = 기준 이상) |
| alarm_standard, status_standard, reason_standard | 표준 모드 지속성 경보(최근 판정 3개 중 2개), 표시 상태, 경보 근거 |

## v1.2 요약
- 최종 모델: 히스토그램 그래디언트 부스팅(hist_gb) + Platt 확률보정. 오프셋 불변 인과 특징 11개.
- 검증: 파일별 시간순 마지막 20% 구간 = 고정 시험(1회 평가). 앞 80%는 시간 블록 5-겹 교차검증(인접 1구간 퍼지)으로 후보 8개 비교·선정·보정·임계값 결정.
- 후보 8개: 단일센서 한계값(기준선), Isolation Forest(비지도 기준선), 로지스틱, 결정트리, 랜덤포레스트, ExtraTrees, 그래디언트 부스팅, 이종 소프트보팅.
- 시험(표준 모드): 표본 F1 0.700, AP 0.847, ROC-AUC 0.981; 지속성 경보 F1 0.767, 이상 구간 5/5 포착.
- 중첩 블록 교차검증(선정 절차 전체 7회 반복): 7회 모두 같은 모델 선택, 외부 폴드 F1 평균 0.814 (범위 0.66~0.90).
- 운영 모드 임계값(개발 OOF 규칙): 무인 0.691 / 표준 0.358 / 점검 선별 0.064.
- 판정 규칙: 이력 3표본 이상 행만 판정·투표(1표본은 특징이 모두 0, 2표본은 상관이 항상 ±1). 준비 행은 모든 지표에서 '무경보'로 계산.

## v1.2.1 변경 사항 (모델·임계값·점수·시험 판정은 변경 없음)
- **분석 정밀도 수정**: `analysis.py`가 개발 OOF 점수를 CSV에서 다시 읽을 때 기본 파서의 끝자리 오차로, 표준 임계값을 정의하는 경계 표본(이상 1건)이 미탐으로 집계되던 문제를 수정(`float_precision="round_trip"`, train.py 결과와 TP·FP 일치 검증 추가). 오류 분석의 미탐 합계 74 → 73건.
- **경보 근거 강화(표시 전용)**: 정상 99% 기준 초과 신호를 오프셋 불변 특징(진동·전류 표준편차, 전류 변화량)으로만 표시(레벨 특징인 전류 RMS 제외). 하부 진동 정지(AI1 진동 표준편차 ≤ 0.0264, 정확히 0.02641) 상태의 관찰·경보에 하부 센서 확인 안내 표시. 경보 진입 시점에 배포 모델의 SHAP 주요 기여 특징을 표준 라이브러리 TreeSHAP으로 계산해 표시(shap 라이브러리와 차이 1e-15 이하).
- **구간별 점검 우선순위**: 연속 구간을 현재 모드 기준 초과 비율 순으로 정렬해 화면과 결과 ZIP(`segment_priority.csv`)에 제공. 시험 구간에서 이상 구간 5개가 124개 중 1·2·3·4·6위(보고서 수치)로 재현됨을 자동 테스트로 확인.
- **현장 프로그램**: 새 파일 분석(폴더 감시 포함) 시 저장 전 작업자 기록이 지워지던 문제 수정(기록마다 대상 파일명·해시 저장). 화면 버전 표기 자동화, 사용자 안내를 README로 연결.
- **경제성 시나리오**: 경보 확인 빈도 = 현장 표준 모드 기준 확인 횟수 × 시험에서 측정한 모드별 상대 경보 빈도(무인 0.3·표준 1·선별 2.4)로 계산하고, 순편익이 0이 되는 확인 한계를 함께 출력.
- **재현성**: 보고서 1·3장 수치(데이터 품질 진단, 하부 진동 정지 비율, SHAP 조건부 효과)를 `signal_diagnostics.py`, `analysis.py`, `shap_analysis.py`가 직접 산출. `build_exe.ps1` 출력 경로를 `portable/`로 수정. 플랫폼 추론 서비스가 숫자가 아닌 입력을 '입력오류' 행으로 처리. 자동 테스트 29 → 37개.

## 파일
| 파일 | 역할 |
|---|---|
| protocol.py | 고정 규칙: 분할·블록 교차검증·후보 8개·보정·임계값·지표·JSON 내보내기 |
| train.py | 개발 OOF → 선정 → 최종 학습·내보내기 → 시험 1회 평가(배포 엔진과 일치 검증) |
| nested_cv.py | 선정 절차 전체의 중첩 블록 교차검증(불확실성 추정, 선정에 미사용) |
| ablations.py | 개발 구간 설계 점검: 특징군·불균형 가중·학습 행·하이퍼파라미터 민감도·보정 방법 |
| analysis.py | 확률보정 진단, 순열·묶음 제거 중요도, FN/FP 집중 조건, 하부 진동 정지 비율, 점검 우선순위, 비용 해석, 오류 사례 |
| shap_analysis.py | 배포 JSON 트리 SHAP·상호작용·조건부 효과·사례(sklearn과 교차검증) |
| signal_diagnostics.py | 10Hz 신호 성격 진단과 데이터 품질·출처 교란 진단(보고서 1장) |
| roi_scenario.py, report_figures.py | 가정 기반 경제성, 보고서 그림 |
| reproduce.py | 위 전 과정을 한 번에 실행하고 재현성 기록 |
| pumpguard/ | 표준 라이브러리 추론(features.py, inference.py: 판정·지속성 경보·경보 근거 TreeSHAP·구간 순위, model.json) |
| app.py, dashboard.html, build_exe.ps1 | 현장 프로그램과 EXE 빌드 |
| tests/ | 자동 테스트 37개 |
| portable/PressGuard.exe | 현장 실행 파일(Windows, 설치 불필요) |
| demo/ | 시험 구간 원본 행(정상·이상 파일 분리, 화면 체험·테스트용) |
| kamp_dl_platform/ | KAMP AI 플랫폼(dl.kamp-ai) 재현 패키지 |
| reference/model_v1.1_extratrees.json | v1.1 모델(보고서의 연속성 비교 전용) |
| results/ | 시험 예측, 지표, 운영 모드, 분석·설계 점검·SHAP·신호 진단 결과, 경제성 가정 |
| licenses/ | EXE에 포함된 Python·PyInstaller 라이선스 |

## 실행

현장용 **`portable/PressGuard.exe`(이 zip에 포함, 약 9MB)** 는 모델과 화면을 모두 담은 단일 실행 파일입니다. 공장 PC에 복사해 더블클릭하면 기본 브라우저가 내 PC의 `127.0.0.1` 주소를 열고 바로 동작합니다. 설치·인터넷·GPU·Python이 필요 없습니다.

권장 PC: Windows 10/11 64비트, CPU 2코어, 메모리 4GB, 여유 공간 200MB, 크롬 또는 엣지. 같은 엔진 기준 1코어로 10Hz 데이터 33분(20,000표본)을 약 2.9초에 분석하며 메모리는 약 30MB를 사용합니다(경보 진입 시점 SHAP 계산 포함). 처음 실행 시 Windows SmartScreen 경고가 나오면 '추가 정보 → 실행'을 선택합니다.

EXE 재빌드: `build_exe.ps1 -Python <Python 3.12 실행 파일>`(PyInstaller 6.22.3, requirements-build.txt). 결과는 `portable/PressGuard.exe`에 저장됩니다.

소스 실행(추론 자체는 Python 표준 라이브러리만 사용):

```powershell
python app.py
python app.py --analyze data/press_data_normal.csv --output result_normal --mode standard
```

종료는 화면 오른쪽 상단의 `프로그램 종료`를 누릅니다. 브라우저 탭만 닫으면 폴더 감시 프로세스가 남을 수 있습니다.

## 입력 계약

- 필수 컬럼: `TimeStamp`, `AI0_Vibration`, `AI1_Vibration`, `AI2_Current`
- 시간은 날짜를 포함한 ISO 형태입니다. 예: `2022-07-12 00:00:00.019`
- 기본 기대 간격 0.1초. 0.15초보다 큰 공백에서 이력 초기화. 0.15초 이하이면서 0.1초와 0.015초 넘게 차이 나는 간격은 입력오류로 처리합니다. 센서 단위·전처리가 학습 자료와 일치하는지 현장 확인 필요.
- CSV UTF-8(BOM 허용) 또는 CP949. 앱 제한은 파일당 16MB / 100,000행입니다.
- GUI에서 컬럼명은 매핑할 수 있습니다. 단위변환·재샘플링·PLC 연결을 자동 처리하지 않습니다.
- `Equipment_state`, 파일명, 원본 인덱스는 예측 입력에 사용하지 않습니다. 절대시각은 학습 변수로 사용하지 않고 순서·간격 확인에만 씁니다.
- 결측·비정상 시각·숫자 오류는 입력오류, 센서 고정·극단 범위 이탈은 판단보류입니다. 정상으로 집계하지 않습니다.

현장 방문 전 확인 항목: 센서 위치(AI0 상부·AI1 하부 가정)와 단위, 원시 신호인지 RMS·평균제거 값인지, 수집 주기, Windows 64비트·브라우저·USB 허용 여부, 가동·정지·금형변경 및 정비 기록, CSV 반출 허용 범위와 현장 담당자.

폴더 감시는 하위 폴더를 읽지 않습니다. 수정시각·크기가 두 번 연속 같은 CSV를 읽고, 파일마다 이력을 초기화합니다. 저장이 완료된 파일을 시간별로 내보내는 방식이 적합합니다. 별도로 지정한 결과 폴더에 각 분석 ZIP을 저장하고, 화면에는 최신 결과를 표시합니다. 작업자 확인 기록은 새 파일이 분석되어도 유지되며(기록마다 대상 파일명·해시 저장), `결과 ZIP 저장`으로 내려받을 때 함께 저장됩니다.

## 결과 파일

`PressGuard_results.zip`에는 전체 표본 예측 `predictions.csv`, 구간별 점검 우선순위 `segment_priority.csv`, 집계 `summary.json`, 원본·모델 SHA256과 설정 `audit.json`, 작업자 관찰 `operator_feedback.csv`가 들어갑니다. 작업자 관찰은 검증된 정답이 아니며 자동 재학습하지 않습니다.

경보 근거(`reason`)는 판정을 바꾸지 않는 표시 정보입니다.
- 정상 기준 대비 관측: AI0·AI1 진동 표준편차, 전류 표준편차, 전류 변화량이 정상 학습 자료의 99% 기준을 넘으면 큰 순서로 2개 표시
- 하부 진동 정지 안내: 관찰·경보 행의 AI1 진동 표준편차가 0.0264 이하이면 하부 센서 부착·배선·커넥터 확인을 먼저 안내(보고서 3장: 오경보의 70%가 이 조건, 이상 표본의 25.6%도 해당하므로 경보를 억제하지 않음)
- 주요 기여(SHAP): 경보에 진입한 행에서 이상 방향 기여가 큰 특징 2개(`contributions` 열에 상위 3개)

점수는 KAMP 내부 분할의 라벨을 기준으로 보정한 이상 점수입니다. 현장의 고장 확률·남은 수명으로 해석하지 않습니다. 실제 고장 시작시각과 정비 기록이 없어 고장 이전 조기경보 선행시간을 평가할 수 없습니다.

## 재현 환경

Python 3.12 / requirements.txt 고정 버전. 배포 모델(`pumpguard/model.json`, SHA-256 `91fd124f…`)은 Linux x64(Python 3.12.3)에서 만들었고 같은 환경에서 바이트 동일 재생성을 확인했습니다. Windows(Python 3.12.15)에서는 부동소수점 끝자리 차이로 모델 파일 바이트가 다르지만, 시험 4,225행의 점수 차이는 3.3e-16 이하이고 판정·경보는 동일합니다. EXE 빌드는 requirements-build.txt.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python reproduce.py
.venv\Scripts\python -m unittest discover -s tests
```

`reproduce.py`는 원본 CSV SHA-256 확인 → 학습·선정·보정·시험 → 분석·설계 점검·중첩 교차검증·SHAP·신호/데이터 품질 진단·경제성·그림을 한 번에 실행(일반 PC 약 8~15분)하고, 결과를 `../evidence/reproduced`에 저장합니다. 배포 모델은 덮어쓰지 않으며, 재생성 모델과의 바이트 동일 여부와 시험 판정·경보 일치 여부를 `REPRODUCIBILITY.json`에 기록합니다.

`results/` 출처: `metrics.json`, `MODEL_RESULTS.md`, `model_comparison.csv`, `selection_protocol.json`, `nested_cv.json`, `screening_segment_ranking_test.csv`는 배포 모델을 만든 실행(Linux x64) 결과입니다. `predictions_test.csv`의 점수·판정·경보 열도 같은 실행 결과이며, 근거(`reason_standard`) 열만 v1.2.1 엔진으로 다시 생성했습니다. `analysis.json`, `ablations.json`, `shap_summary.json`, `signal_diagnostics.json`, `roi_scenarios.json`과 CSV 분석표는 v1.2.1 코드로 Windows(Python 3.12.15)에서 재실행한 결과입니다.

## 데이터와 모델의 한계

정상·이상 파일은 각각 다른 날짜이고 진동 정밀도와 평균제거 특성이 다릅니다. 그래서 파일 단위 오프셋에 영향을 받는 특징은 설계 단계에서 제외했습니다. 그래도 내부 분류 성능은 현장 일반화 성능의 증거가 아닙니다. 보정 확률은 데이터셋 내 이상 비율(개발 구간 판정 대상 표본 기준 약 2.8%) 기준이므로 현장 비율에 맞춘 재보정이 필요합니다. 시험 구간의 정상 경보 진입 빈도(표준 모드 409초 중 10건)는 5초 이하 기록 조각에서 측정한 값이므로, 현장 정상 자료로 기준을 재산출한 뒤 실제 확인 빈도를 다시 측정해야 합니다. 현장 설치 후에는 센서 매핑, 다양한 부하·전환 상태의 정상 수집, 고장·정비 정답과 독립된 평가가 필요합니다.

## 출처

데이터: KAMP 소성가공 예지보전 AI 데이터셋(대회 제공). 원본 저작권·이용조건은 원 제공자에게 있습니다.
과제 안내: https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=28

## KAMP AI 플랫폼 재현

`kamp_dl_platform/`에 dl.kamp-ai(AI-PaaS)용 실행환경(`1_실행환경`, requirements·Dockerfile, t3qai_client 1.1.2), 학습·추론 알고리즘 패키지(`2_학습알고리즘`), 데이터셋 업로드용 zip(`3_데이터셋`), 추론 테스트 입력(`4_추론테스트`)과 이관 가이드가 있습니다. 플랫폼에서 학습(약 1분 23초)과 추론 서비스 배포까지 수행했고, 플랫폼 학습 결과의 시험 판정·경보가 이 소스의 결과와 동일함을 확인했습니다(`platform_run_result/`, v1.2.0 실행 기록).
