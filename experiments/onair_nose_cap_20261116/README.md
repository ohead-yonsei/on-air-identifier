# ON-AIR 코끝–모자 연결 진단

실행일: 2026-11-16. 기존 실험 및 영상은 변경하지 않았습니다.

이번 작업은 수동 코끝·모자 좌표를 실제 서버에서 연결해 보는 기하 진단입니다. 자동 코끝 검출, 자동 몸통 연결, 자동 모자색 분류, 카메라 방향 추정, 공식 순위 판정 모델을 실행한 실험이 아닙니다. 새 모델 생성 호출 0회, 학습 0회, GPU 사용 없음입니다.

- 맥 폴더: `/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_nose_cap_20261116`
- 서버 폴더: `/home/ohead/ai-workspace/projects/onair_nose_cap_20261116`
- 접속 수단: 맥 `/Users/ohead/Documents/ChatGPT/SSH연결/ai-desktop`
- 서버: `[server-address-hidden]`, 기존 SSH 설정 사용. 인증정보는 포함하지 않았습니다.
- Python: `/home/ohead/ai-workspace/.venv/bin/python`, 3.12.3
- 실제 실행 CPU: Intel Xeon Gold 6246R @ 3.40GHz, 16코어·32스레드
- NumPy 2.5.2, SciPy 1.18.0, Pillow 12.3.0. 설치나 환경 변경 없음.

## 실행

저장된 결과를 보존하려면 먼저 **새 폴더에 복사한 뒤** 실행합니다. 아래 명령은 복사본에서 실행하는 예입니다. 코드의 ROOT는 코드 파일의 상위 실험 폴더를 자동 사용합니다.

```bash
cd /home/ohead/ai-workspace/projects
cp -a onair_nose_cap_20261116 onair_nose_cap_reproduce
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ohead/ai-workspace/.venv/bin/python onair_nose_cap_reproduce/code/run_association.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ohead/ai-workspace/.venv/bin/python onair_nose_cap_reproduce/code/test_direction_gate.py
```

첫 번째는 고정한 세 가지 기본 규칙, 두 번째는 그 실패를 확인한 뒤 추가한 방향 제한 규칙입니다. 두 번째는 같은 표본에서 고안하고 재평가한 탐색이므로 독립 검증으로 해석하면 안 됩니다. 난수 seed는 20261116입니다. 시간 측정값은 재실행 환경에 따라 달라질 수 있습니다.

입력은 기존 `onair_cap_silks_20261116/source_frames`에서 복사한 1280×720 원본 두 장입니다. 원본 PTS 35.0초(index 1049), 38.5초(index 1154). 기존 영상 추출 근거와 마번 검수 범위를 `evidence/`에 함께 보존했습니다. 새 영상 추출이나 미래 프레임 사용은 없습니다. 두 장 사이를 추적하지 않습니다.

`source_frames`에는 기록 보존을 위한 방송 원본 그래픽이 있습니다. `inputs`와 비교 그림에는 좌상단 `(0,0)–(240,110)`, 하단 `y=610–720`을 마스킹했습니다. 알고리즘에는 이미지·음성·그래픽·마번·정답 대응을 전달하지 않고, 섞은 좌표 배열과 수동 방향 벡터만 전달했습니다.

## 파일

| 파일/폴더 | 내용 |
|---|---|
| `REPORT_KO.md` | 실제 결과, 각도 질문에 대한 답, 한계 |
| `annotations.json` | 수동 코끝·모자 좌표, 제외 항목, 육안 대응 초안, 라벨 출처 |
| `config.json` | 최초 세 가지 규칙과 고정 설정 |
| `code/run_association.py` | 거리·선·중복 금지 연결, 시간 측정, 좌표 흔들림 진단 |
| `code/test_direction_gate.py` | 후속 방향 제한 규칙, 각도 ±15/±30도 진단 |
| `results/raw.jsonl` | 기본 규칙의 좌표 배열·비용 행렬·예측·1,000회 시간 원시값 |
| `results/comparison.csv` | 11개 코끝 × 3규칙 = 33개 연결 결과 |
| `results/summary.json`, `results/timing.json` | 임시 연결 기준과의 일치 수, 시간, 안정성 |
| `posthoc_direction/` | 후속 규칙, 11개 × 5각도 = 55개 연결, 원시 결과 |
| `review/nearest_failure_examples.png` | 다른 말의 모자와 가까워지는 두 사례 |
| `review/direction_vs_connection.png` | 방향 기준선과 코끝–모자 연결선의 차이 |
| `review/*_reference_links.png` | 전체 수동 연결 초안 검수용 그림 |
| `evidence/environment.json`, `evidence/cpu.json` | 실제 실행 환경 및 코드·입력 해시 |
| `evidence/verification.json` | 마스킹·입력 순서 불변성 8개 확인 |
| `SHA256SUMS` | 최종 산출물 해시 |

사람이 확인한 기존 마번 라벨은 **38.5초의 6개 모자**에 한정됩니다. 이번 코끝 위치와 코끝–기수 대응은 assistant가 육안 검토한 초안이며 사용자·전문가 검수 전입니다. 이를 자동 검출 정확도나 전문가 정답 정확도로 표현하지 않습니다.
