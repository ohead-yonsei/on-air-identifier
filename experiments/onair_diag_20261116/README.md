# ON-AIR 초기 진단 — 2026-11-16

서버에서 실제 경마 영상으로 4장면 × 4조건 = **16회** 생성했다. 원래 문서의 2026-09-06 자료는 없었으며, 사용자가 유일한 영상이라고 확인한 2026-10-10 서울 10경주로 대체했다. 명확한 두 장면의 선두는 모두 9번이므로 **서로 다른 선두라는 설계 조건은 충족하지 못했다**. 전문가 정답은 없고 판정은 근거를 연결한 AI 시각 검수 초안이다. 가설 입증이나 정확도 벤치마크로 사용하지 않는다.

- 서버 폴더: `/home/ohead/ai-workspace/projects/onair_diag_20261116`
- 맥 폴더: `/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_diag_20261116`
- 원본 영상: `/Users/ohead/Downloads/(서울) 2026.10.10 10경주.mp4` — 변경하지 않음
- 서버 복사본: `candidate_20261010_R10.mp4`
- 핵심 결과: `REPORT_KO.md`, `comparison.csv`, `claim_review.csv`, `raw_responses.jsonl`
- 검수 이미지: `evidence/representative_frames.jpg`, `frames/*/contact.jpg`
- 모델에 들어간 정확한 이미지: `frames/*/f001.jpg`~`f016.jpg`
- 무음 검수 영상: `frames/*/input_review.mp4` (실제 입력 2fps 이미지로 재구성)

맥의 `/Users/ohead/Documents/ChatGPT/SSH연결/README.md`, `ai-desktop`, `ai-shell`을 먼저 읽었다. `ai-desktop`은 그 폴더의 SSH 설정을 사용하는 래퍼다. `ai-shell`은 원격 `~/ai-workspace/.venv`를 활성화하는 대화형 셸이다. 자동 실행은 `ai-desktop`과 원격 Python의 절대 경로를 썼다. 호스트키 검증을 유지했고 인증키·토큰 파일을 읽거나 로그에 넣지 않았다.

```bash
cd "/Users/ohead/Documents/ChatGPT/SSH연결"
./ai-desktop nvidia-smi
./ai-desktop 'cd /home/ohead/ai-workspace/projects/onair_diag_20261116 && /home/ohead/ai-workspace/.venv/bin/python code/verify_run.py'
```

첫 샌드박스 시도는 `ssh: Could not resolve hostname [server-address-hidden]: -65563`로 실패했다. 허가된 네트워크 실행으로 재시도하여 연결됐다. Tailscale 앱 설정이나 로그인 상태를 변경하지 않았다. DNS 이름을 통한 실제 SSH 성공을 연결 근거로 삼는다.

실제 환경은 Ubuntu 24.04.4 LTS, RTX 3090 24,576 MiB, NVIDIA 595.84, Python 3.12.3, PyTorch 2.12.1+cu132/CUDA 13.2, FFmpeg/ffprobe 6.1.1이다. Ollama 0.32.15가 `127.0.0.1:11435`에서 실행 중이었다. 사용 모델은 `qwen3.5:9b`, 9.7B, GGUF Q4_K_M이다. 모델 digest와 설정은 `config.json`, 전체 출력은 `evidence/environment.json`에 있다. 새 패키지·모델을 설치하거나 기존 서비스 설정을 변경하지 않았다.

실행 순서는 입력 검사 → 64장 시각 검수 → `S2_bend/F` 한 회 → 나머지 15회였다. 매 요청은 system/user 두 메시지뿐이며 이전 응답을 넣지 않았다. 서버 모델 적재·프롬프트 캐시는 유지되므로 독립 요청이 곧 콜드 스타트를 뜻하지 않는다. 생성 설정은 temperature 0, seed 20261116, context 32768, 최대 256 토큰, thinking false. 전체 영어 프롬프트와 행 순서는 `prompts/*.json`에 있다.

```bash
# 서버에서 실행. 완료된 scene/condition은 재실행하지 않음.
cd /home/ohead/ai-workspace/projects/onair_diag_20261116
/home/ohead/ai-workspace/.venv/bin/python code/run_diagnostic.py --max-new 1
/home/ohead/ai-workspace/.venv/bin/python code/score_review.py
/home/ohead/ai-workspace/.venv/bin/python code/verify_run.py
```

`prepare_inputs.py`는 `ffprobe`의 디코딩 프레임 PTS를 읽어 목표 시각 이하의 프레임만 고른다. 8초 구간에서 0.5초 간격으로 16장을 뽑고 평가 시점을 넘는 프레임이 없음을 assert한다. 소스 PTS, 프레임 인덱스, SHA256, 실제 FFmpeg 명령을 `scene_inputs.json`과 각 장면 manifest에 남긴다. 기존 출력이 있으면 덮어쓰지 않고 중단한다. 결과·순위·마번 연결선이 나타나는 다른 구간은 제외했으며 선택 구간의 방송 패널은 동일한 마스크로 가렸다. 원본·검수용 contact sheet·공식 성적 페이지는 모델에 전달하지 않았다.

`run_diagnostic.py`는 전송 직전에 이미지 SHA256을 재검사하고, 정확한 요청의 SHA256·바이트 수를 `request_ledger.jsonl`에 남긴다. Base64는 로그에 중복 저장하지 않으며 저장 이미지와 코드로 요청 전체를 동일하게 복원할 수 있다. 원시 HTTP 응답 문자열과 파싱 결과를 모두 `raw_responses.jsonl`에 보존했다. 자동 재시도는 없었으며 생성 16회, 추가 생성 0회다.

`score_review.py`는 이번 응답에 대한 명시적인 주장별 검수 기록을 CSV로 만드는 코드다. 새로운 응답을 자동으로 참/거짓 판정하는 범용 채점기가 아니다. `claim_review.csv`의 `human_review` 열은 전문가 확인을 위해 비워 두었다. 확인되지 않은 순위·거리·근접성은 미확인으로 남겼다. 원래 `score_ab2.py`는 실행하지 않았다.

백엔드 감사에서 요청 16장이 8개 시각 배치로 처리되는 것을 발견했다. 동일 크기 인접 이미지를 두 프레임씩 합치는 llama.cpp 동작과 일치한다. 이는 8장이 누락되었다는 증거가 아니다. `run_separated_followup.py`는 각 이미지 사이에 프레임 텍스트와 `[img]`를 넣는 **후속 비교용 코드**다. `separated_preview/`는 프롬프트만 만들었고 모델 생성은 하지 않았다. 기존 16회와 혼합하지 않는다.

```bash
# 새로운 경로에 후속 요청만 준비: 추론 없음
/home/ohead/ai-workspace/.venv/bin/python code/run_separated_followup.py \
  /home/ohead/ai-workspace/projects/onair_followup_NEW --write-prompts
# 향후 명시적으로 후속 실험할 때에만 --write-prompts 대신 --max-new 1을 사용
```

이 폴더의 비공개 원본·프레임은 사용자의 검수용이다. 공식 정답을 만들려면 `claim_review.csv`와 입력 프레임을 함께 검수해야 한다. 원래 팀 자료, 전문가 라벨, 서로 다른 선두의 두 번째 경주가 여전히 없다.
