# ON-AIR 모자·복색 식별 실험

서버 경로: `/home/ohead/ai-workspace/projects/onair_cap_silks_20261116`

맥 경로: `/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_cap_silks_20261116`

실행일은 2026-11-16(Asia/Seoul). 서버의 기존 `/home/ohead/ai-workspace/.venv/bin/python`과 Ollama loopback `http://127.0.0.1:11435`를 사용했다. 새 패키지·모델을 설치하거나 사용자 가상환경을 변경하지 않았다. SSH는 맥 `/Users/ohead/Documents/ChatGPT/SSH연결/ai-desktop`으로 수행했다. 인증정보를 실험 파일에 기록하지 않았다.

## 실제 실행과 범위

12개 수동 영역 × H(모자)/S(옷)/HS(둘 다) = 36회 VLM 호출을 수행했다. 첫 1회를 확인하고 나머지 35회를 실행했다. 모든 H가 보류해 별도 `direct_cap/`에서 모자 단독 색상 분류 6회를 추가했다. 총 42회며 재시도 0회다. 기존 중계 진단의 20회와는 별도 실험이다.

기존 HSV 함수로도 36개 예측을 생성했으며, 각 조합을 200번 반복 계산해 시간만 측정했다. 새 신경망 분류기의 학습이나 검출·추적·순위 성능 평가는 하지 않았다.

사람이 검수한 **G–L 6개만** 정확도 계산에 사용한다. A–F는 라벨 초안이며 이번 정확도에 포함하지 않는다. `human_labels.json`에 사용자 확인의 정확한 범위를 남겼다.

## 파일

- `REPORT_KO.md`: 결과·해석·한계.
- `comparison.csv`: 78개 출력(본 VLM 36, 규칙 36, 추가 VLM 6)의 개별 비교. 미검수 정답·정오 칸은 비어 있다.
- `metrics.csv`, `summary.json`: 사람 검수 6개 기준의 정답·오답·보류·응답률·시간.
- `samples.json`: PTS, 원본 해시, 모자/옷 좌표, 모자 제거 마스크, 조건별 파일 해시.
- `config.json`: 본 비교의 고정 모델·설정·범위.
- `reference_manifest.json`: 제공된 경주복 11개의 마번·마주 대응과 파일 해시.
- `source_frames/`: 원본 전체 프레임. 방송 그래픽이 있으므로 모델에 사용하지 않는다.
- `crops/`, `inputs/`: 실제 비교 영역과 36개 조건 입력.
- `references/registry.png`: 모델이 받은 복색 기준표.
- `review/`: 사람이 보기 위한 이미지. 마번 정답이 적힌 검수 이미지는 모델 입력에서 제외한다.
- `prompts/`, `vlm_request_ledger.jsonl`, `vlm_raw.jsonl`: 전체 프롬프트·요청 SHA256·원시 HTTP 응답.
- `color_raw.jsonl`, `color_method.json`: 고정 HSV 규칙 결과 및 반복 시간 측정.
- `direct_cap/`: 사후 추가 진단의 단독 입력·프롬프트·원시 응답·변경 사유.
- `evidence/`: 공식 출전표 PDF/렌더, 자료 대조, 환경, 백엔드 로그.
- `verification.json`: 200개 무결성·입력 통제 검사.

## 실행 코드

서버에서 실험 폴더로 이동한 뒤 사용한 명령이다. 완료된 VLM 호출은 건너뛴다.

```bash
/home/ohead/ai-workspace/.venv/bin/python code/prepare_crops.py
/home/ohead/ai-workspace/.venv/bin/python code/run_crop_comparison.py --max-new 1
/home/ohead/ai-workspace/.venv/bin/python code/run_crop_comparison.py --max-new 35
/home/ohead/ai-workspace/.venv/bin/python code/run_color_baseline.py
/home/ohead/ai-workspace/.venv/bin/python code/run_direct_cap.py
/home/ohead/ai-workspace/.venv/bin/python code/summarize_verify.py
```

완료 폴더에서 `prepare_crops.py`를 다시 실행하면 입력이 덮어써질 수 있고, `run_color_baseline.py`는 기존 시간 측정 파일을 덮어쓰므로 재현은 새 폴더 복사본에서 수행한다. **추론 없이 검증·채점만 다시 하려면 마지막 `summarize_verify.py`만 실행한다.** 해당 명령은 Pillow·NumPy가 있는 서버 가상환경에서 실행한다.

`legacy_color.py`는 팀 문서 11-1의 색상 함수들을 보존한 사본이다. 이번 데이터로 계수를 맞추지 않았고 검출기·Hungarian 배정은 포함하지 않았다. 규칙 점수는 확률이 아니다.

## 시간 해석

VLM 시간은 서버 내부 HTTP 요청의 전체 응답 시간이다. 규칙 시간은 이미 메모리에 있는 크롭에서 CPU 특징·점수를 계산하는 시간이다. 영상 읽기·자동 영역 검출·추적·크롭 생성·해설 음성 지연은 포함하지 않는다. 같은 캔버스의 H/S/HS 비교는 정보 차이를 통제하기 위한 것이며, 작은 입력으로 배포했을 때의 속도 절감량을 측정한 것은 아니다.

모자 단독 추가 진단은 기준표 제거, 대상 확대, 색상 분류 질문, 짧은 출력이 함께 바뀐다. 본 비교를 대체하는 결과로 섞거나, 단일 요인의 효과라고 해석하지 않는다.
