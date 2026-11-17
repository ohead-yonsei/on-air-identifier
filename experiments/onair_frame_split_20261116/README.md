# ON-AIR 프레임 분리 전달 후속 실험

- 실행일: 2026-11-16 (Asia/Seoul)
- 서버: ai-desktop, Ubuntu 24.04.4 LTS, RTX 3090 24,576 MiB
- 모델: qwen3.5:9b, Q4_K_M, Ollama 0.32.15
- 모델 digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`
- Python: `/home/ohead/ai-workspace/.venv/bin/python`
- API: 서버 내부 `http://127.0.0.1:11435`
- 서버 결과: `/home/ohead/ai-workspace/projects/onair_frame_split_20261116`
- 맥 결과: `/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_frame_split_20261116`

이번 4회는 F만 실행했다. 원래 16회와 합계 20회이며 Jev 호출은 0회다. SDK·모델 설치, 기존 환경 변경 없이 기존 실행 코드를 사용했다. 원래 실험 폴더는 보존했다.

## 실제 실행 방법

맥 접속 래퍼는 `/Users/ohead/Documents/ChatGPT/SSH연결/ai-desktop`이다. 아래는 그 래퍼로 실제 실행한 서버 명령이다.

```bash
/home/ohead/ai-workspace/.venv/bin/python \
  /home/ohead/ai-workspace/projects/onair_diag_20261116/code/run_separated_followup.py \
  /home/ohead/ai-workspace/projects/onair_frame_split_20261116 \
  --scene S2_bend --condition F --max-new 1

/home/ohead/ai-workspace/.venv/bin/python \
  /home/ohead/ai-workspace/projects/onair_diag_20261116/code/run_separated_followup.py \
  /home/ohead/ai-workspace/projects/onair_frame_split_20261116 \
  --condition F --max-new 3
```

첫 호출을 확인한 다음 나머지를 실행했다. 순서는 S2, S1, S3, S4다. 완료된 scene/condition은 실행 코드에서 건너뛴다. 새 재현 실험에는 별도의 새 결과 경로를 사용한다.

## 산출물과 검증

- `REPORT_KO.md`: 실제 결과와 한계.
- `human_review.json`: 사용자가 확인한 S2 선두 9번과 검수한 프레임 시각.
- `raw_responses.jsonl`: 실제 4회 원시 HTTP 응답과 파싱 결과.
- `request_ledger.jsonl`: 요청 전 기록한 프롬프트·설정·이미지 해시·요청 SHA256.
- `comparison.csv`: 원래 F와 이번 F의 나란한 비교. 미확인 정답은 빈칸이며 0점이 아니다.
- `config.json`, `scene_inputs.json`, `frames/`: 동일 설정·입력 시각·이미지.
- `prompts/`: 실행 코드가 준비한 16개 조합의 프롬프트. 실제 실행은 F 4개뿐이다. 파일 존재를 실행 증거로 세면 안 된다.
- `code/`: 실제 실행 코드 사본과 결과 검증·비교 코드.
- `evidence/`: 이미지 처리 로그와 GPU·실행 상태.
- `verification.json`: 164개 검증 결과.

기존 실험 폴더가 같은 상위 디렉터리에 있을 때, 다음 명령은 **추론 없이** 요청 복원·해시 검증·비교 표 생성을 수행한다.

```bash
python3 code/summarize_followup.py
```

서버 `frames/`는 원래 실험 폴더의 프레임을 참조한다. 맥에는 요청에 쓰인 64개 JPEG를 동일 바이트로 복사했다. 모델에는 마스킹된 JPEG만 전달했고 원 음성·정답·사후 순위를 전달하지 않았다. 사용자 검수 파일도 모델 요청에 포함하지 않았다.

후속 요청의 차이는 명시적 이미지 태그 사이에 프레임 번호와 상대 시각을 넣은 것이다. 분리 효과와 시간 표시 효과가 함께 바뀌었으며, 속도 비교에는 적재·캐시 상태라는 한계가 있다. S2 외 장면은 사람 정답이 없어 정확도를 계산하지 않았다.
