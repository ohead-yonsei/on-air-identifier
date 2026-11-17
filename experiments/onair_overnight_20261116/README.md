# ON-AIR 야간 연구 실행·산출물 안내

시작: 2026-11-16 밤(KST). 확인 예정: 2026-11-17 오전 9시(KST).

먼저 [오전 인계 보고서](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/MORNING_REPORT_KO.md)와 [핵심 이미지 모음](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/highlights/INDEX_KO.md)을 읽으면 된다.

## 위치·환경

- 맥: `/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116`
- 서버: `/home/ohead/ai-workspace/projects/onair_overnight_20261116`
- 접속: 맥 `/Users/ohead/Documents/ChatGPT/SSH연결/ai-desktop`, 기존 SSH 설정 사용
- 원본 영상: `/home/ohead/ai-workspace/projects/onair_diag_20261116/candidate_20261010_R10.mp4`
- 원본 SHA256: `9475c34a814961b4f24cc95c5812ac855ce251bed83db6ffab2bffe0443f0489`
- 서버 Python: `/home/ohead/ai-workspace/.venv/bin/python`, 3.12.3
- PyTorch 2.12.1+cu132, torchvision 0.27.1+cu132, RTX 3090 24,576MiB
- Ollama: `http://127.0.0.1:11435` 서버 내부 loopback. 인증 프록시·비밀 파일을 사용하지 않았다.
- 9B 및 27B 모델 ID·digest는 `config.json`, `FOLLOWUP_DESIGN.json`, 원시 요청 기록 참조
- 설치된 OpenCV, NumPy, SciPy, Pillow를 사용했다. 공유 가상환경에 패키지를 설치하거나 서비스 설정을 바꾸지 않았다.
- 일반 말·사람 검출 가중치는 `model_cache/checkpoints`에만 다운로드했다. 실제 URL·크기·SHA256은 `evidence/detector_environment.json`에 있다. 가중치 캐시는 서버에 두고 맥 동기화·검토용 묶음에서는 제외한다.

## 재실행

기존 결과를 보존하려면 새 폴더에 재현에 필요한 입력·코드를 복사한 뒤 실행한다. 원본 폴더에서 실행하는 것은 권하지 않는다. 생성 요청 코드는 완료한 item을 건너뛰며, 전역 요청 ledger와 오전 마감 시각을 확인한다. 재현 시에는 새 폴더의 `config.json`에서 **새 마감 시각과 요청 한도**를 명시해야 한다. 시간 측정값은 재실행 환경에 따라 달라질 수 있다.

```bash
cd /home/ohead/ai-workspace/projects
cp -a onair_overnight_20261116 onair_overnight_reproduce
cd onair_overnight_reproduce
# 아래는 코드 실행 예시다. 기존 raw/results가 있으면 생성 호출은 대부분 건너뛴다.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ohead/ai-workspace/.venv/bin/python code/survey_video.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ohead/ai-workspace/.venv/bin/python code/run_detector.py --first
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ohead/ai-workspace/.venv/bin/python code/run_vlm.py --task view --first
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ohead/ai-workspace/.venv/bin/python code/batch.py
```

완전히 새로 생성하려면 새 폴더에 `code`, `config.json`, `selection.json`, `geometry_annotations.json`, `manual_cap_probe.json`, `protocols`, `frames`, `evidence/annotation_lock.json`, `evidence/video_pts.json`을 복사한다. `raw`의 기존 결과를 지우는 대신 새 결과 폴더를 사용한다. 원본 영상 경로와 설치 모델이 접근 가능해야 한다. 설정 변경도 함께 기록한다.

추가 비교의 명령은 다음과 같다. 처음 50회 기본 실행 뒤 설계한 사후 탐색이며 독립 검증 결과로 합치지 않는다.

```bash
python code/modular_roi.py
python code/run_vlm.py --task view --core-only --front-first
python code/run_vlm.py --task view --core-only --model qwen3.8:27b-q4_K_M --digest 25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e
python code/run_vlm.py --task localize --core-only --model qwen3.8:27b-q4_K_M --digest 25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e
python code/manual_cap_probe.py
python code/flow_direction.py
python code/flow_direction.py --lookback 0.1 --motion-model homography --tag flow_01_homography
python code/flow_direction.py --lookback 0.5 --motion-model affine --tag flow_05_affine
python code/flow_direction.py --lookback 0.1 --motion-model affine --tag flow_01_affine
python code/flow_direction.py --lookback 0.1 --motion-model affine --tag flow_01_affine_mirror --mirror
python code/rank_proxy.py
python code/summarize_all.py
python code/build_highlights.py
python code/verify_artifacts.py
```

위 `python`은 가상환경을 활성화한 경우다. 비활성화 상태에서는 앞의 절대 Python 경로를 사용한다. `flow_direction.py`는 이미 결과 JSON이 있으면 덮어쓰지 않고 건너뛴다. `code/archive`는 실패 당시 코드의 보존본이며 기본 실행 진입점이 아니다.

## 실제 배치 실행과 한도

기본 배치는 `subprocess.Popen(..., start_new_session=True)`로 SSH 종료와 독립 실행했고 PID 2050329를 기록했다. 27B 비교는 PID 2051507로 별도 실행했다. 로그는 `batch.log`, `batch_27b.log`, 진행 이력은 `journal.jsonl`이다. 실행 완료 여부는 최종 상태·프로세스 확인 기록으로 판단한다.

상한은 VLM 160요청, 일반 검출 240프레임, 공개 가중치 700MB였다. 실제 VLM은 97요청, 일반 검출은 25프레임이었다. 이 숫자는 생성 호출 수이며 시간 측정·좌표 오차 진단 반복을 독립 영상 표본으로 세지 않는다. 모델 HTTP 실패·재시도는 없었고 CPU 구현 중단 2건과 수정 기록은 별도 보존했다.

## 주요 파일

| 파일/폴더 | 내용 |
|---|---|
| `PLAN_KO.md` | 실행 전 계획·범위·한도 |
| `FOLLOWUP_DESIGN.json`, `FLOW_GRID_DESIGN.json` | 실패 관찰 뒤 추가한 비교의 설계 |
| `MORNING_REPORT_KO.md` | 실제 결과·한계·다음 우선순위 |
| `highlights/INDEX_KO.md` | 주요 근거 7개와 사진 |
| `selection.json`, `frame_manifest.json` | 입력 선정, 원본 PTS·decoded index·해시 |
| `geometry_annotations.json` | 수동 좌표·방향·새 대응 초안과 제외 사유 |
| `evidence/annotation_lock.json` | 새 결과 확인 전 고정한 라벨 해시 |
| `raw/request_ledger.jsonl` | 요청 전에 저장한 전체 프롬프트·설정·이미지/요청 해시 |
| `raw/vlm.jsonl` | 뷰·위치 추정 75개 원시 응답 |
| `raw/modular_cap.jsonl`, `raw/manual_cap.jsonl` | 자동 영역 14개 고유 호출, 수동 영역 8개 호출 |
| `raw/detector.jsonl` | 일반 말·사람 검출 박스·점수·GPU forward 시간 |
| `raw/geometry.jsonl` | 좌표 연결 비용 행렬·예측·시간 원시값 |
| `results/SUMMARY.json` | 계산된 전체 수치 |
| `results/geometry_comparison.csv` | 22개 코끝×5규칙 = 110행 |
| `results/localization_comparison.csv` | 8프레임×3조건 = 24행 |
| `results/modular_pipeline.json` | 수동/9B 방향 각 8개 후보 선택·영역·분류 결과 |
| `results/*direction.json`, `flow*/` | 자동 움직임 추정, 과거 프레임 시각, 특징점·변환·시각화 |
| `review/USER_REVIEW.csv` | 확인 칸을 비워 둔 사용자 검수 목록 |
| `evidence/execution_failures.jsonl` | CPU 중단 원인·영향·수정 |
| `evidence/direction_reference_limitations.json` | 수동 방향을 정확도 정답으로 쓰지 않는 이유 |
| `SHA256SUMS` | 최종 공유 산출물 해시. 모델 캐시·pycache·요약 ZIP 제외 |

프레임·모델 요청·결과의 실제 대응은 `verify_artifacts.py`로 확인한다. 그림에 추가한 번호·화살표는 검토용이며 모델 입력에 넣지 않았다. 정답 확인 칸을 비워 두었으므로 새 초안이 사람 검수 라벨로 자동 승격되지 않는다.
