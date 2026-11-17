# ON-AIR 야간 연구 2일차

먼저 [한국어 결과 보고서](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/MORNING_REPORT_KO.md), [검수 목록](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/review/USER_REVIEW_KO.md)을 연다. 최신 실행 상태는 status.json, 단계 기록은 journal.jsonl이다. 보류된 순서 평가는 실패를 숨긴 완료 처리와 구분한다.

## 환경과 접속

맥 접속 폴더: `/Users/ohead/Documents/ChatGPT/SSH연결`. README/ai-desktop/ai-shell을 확인해 기존 SSH/Tailscale 경로로 연결했다. 비밀키·인증정보를 이 폴더에 복사하지 않았다.

```bash
cd '/Users/ohead/Documents/ChatGPT/SSH연결'
./ai-desktop nvidia-smi
./ai-desktop
```

서버 실험 폴더: `/home/ohead/ai-workspace/projects/onair_overnight_20261117`

Python: `/home/ohead/ai-workspace/.venv/bin/python`. 패키지를 새로 설치하지 않았다. RTX3090 24GB, torch2.12.1+cu132, torchvision0.27.1+cu132, Ollama0.32.15. 정확한 환경은 evidence/environment_start.json. Ollama는 서버 내부 `http://127.0.0.1:11435`; 모델 qwen3.5:9b, GGUF Q4_K_M, digest는 config.json에 고정했다.

모델의 모든 요청은 독립 2개 메시지(시스템/사용자)+이미지1장이다. temperature0, seed20261116, num_ctx16384, num_predict96, top_k20, top_p1, repeat_penalty1, think=false. 전체 프로토콜은 protocols/cap_classifier.json이다. keep_alive만 첫 점검/배치 마지막에서0으로 하여 GPU를 반환했으며 생성 설정은 동일하다. 모델 콜드 로딩 시간도 원시 HTTP 지연에 포함한다.

## 자료와 코드

- `PLAN_KO.md`, `config.json`: 실행 전 고정 계획/한도. evidence/plan_lock.json, annotation_lock.json, evaluation_lock.json으로 변경을 검증한다.
- `selection.json`, `frame_manifest.json`, `tracking_frame_manifest.json`: 원본 PTS, 프레임 인덱스, 마스크, 입력 해시. 추출 시 목표보다 늦은 프레임은 제외한다.
- `annotations.json`: 60개 새 모자 초안+사람 확인6개. 좌표는 원본픽셀, 박스는 초안이다. 미확인 색4개를 정답으로 채우지 않는다.
- `raw/request_ledger.jsonl`: 요청 전에 저장한 전체 프롬프트/설정/이미지와 본문 해시. 이미지 base64는 저장 PNG로 복원한다. `raw/vlm.jsonl`은 전체 HTTP 원시 응답·지연·파싱 결과를 포함한다.
- `raw/detector.jsonl`, `raw/pose.jsonl`, `raw/tracking_detector.jsonl`, `raw/tracking*.jsonl`: 박스, 자세점, 프레임별 대응·트랙·이동/컷 근거. 트랙 ID는 마번이 아니다.
- `results/condition_comparison.csv`: 관측×영역×분류기660행. condition_summary.csv, colour_confusion.csv, paired_changes.csv 및 C_SUMMARY.json에 집계. 초안 일치와 사람 확인 정답을 나눴다.
- `results/tracking_summary.json`: 어제22개 초안 점과의 대응 및 드문 시점의 ID 변화. 공식 MOT 평가가 아니다.
- `results/human_mask_control.json`: 최종 육안 점검에서 원 사람 확인 프레임의 그래픽 제거 누락을 발견해 별도 실행한 사후 입력 보정. 검출·자세2회와 VLM10회 추가, 동일 이미지18개 재사용, 영역2개 보류. 원 C 결과를 덮지 않았으며 보고서의 사람 확인 결과는 보정 전후를 구분한다.
- `highlights/`: 실제 입력·출력이 표시된 성공/실패/미확인 이미지. 모델에 주석 이미지를 입력하지 않았다.
- `review/`: 경주별 검수 이미지, 12개 질문, 확인 칸을 비운 CSV. 이전22개 연결과 선두8시점도 미검수로 유지한다.

## 저장된 결과만 다시 검증·집계

다음은 **새 추론 없이** 동작한다. 저장된 원 결과를 먼저 백업하고, 원 결과를 갱신할 필요가 없다면 verify만 실행한다.

```bash
cd /home/ohead/ai-workspace/projects/onair_overnight_20261117
/home/ohead/ai-workspace/.venv/bin/python code/verify_results.py
# 아래 명령은 집계·검수 그림·보고서를 재생성한다. 확인 칸을 작성한 뒤에는 present_results.py를 재실행하지 않는다.
/home/ohead/ai-workspace/.venv/bin/python code/summarize_c.py
/home/ohead/ai-workspace/.venv/bin/python code/present_results.py
/home/ohead/ai-workspace/.venv/bin/python code/write_report.py
/home/ohead/ai-workspace/.venv/bin/python code/package_review.py
```

verify_results.py의 결과는 evidence/final_verification.json에 저장한다. SHA256SUMS는 재생성 후 package_review.py로 갱신한다. 맥에서는 실험 폴더에서 `shasum -a 256 -c SHA256SUMS`로 동기화 파일을 확인할 수 있다.

## 실제 실행한 순서 / 새로운 재실행

원 폴더에서 모델 실험을 다시 돌려 평가 결과를 덮어쓰지 않는다. 새 실험 폴더에 코드·계획·설정·동결 라벨·마스크·프로토콜·영상·캐시를 복사하고 raw/results/logs/status를 새로 시작한다. 절대경로는 config.json과 입력 명세의 소스 경로를 검토한다. 이전 폴더와 원본 영상은 읽기 전용으로 유지한다. 현재 해시 잠금은 원 실험 검증용이므로 새 설계를 변경했다면 새 잠금 기록과 실험명을 만든다.

실제 실행 순서는 아래와 같다. 모든 Python은 위 가상환경을 사용했다. `--first`는 첫 요청을 확인하고 같은 요청을 재사용하므로 중복 호출하지 않는다.

```bash
python tools/verify_videos.py
python code/prepare.py
python code/survey.py
python code/extract_selected.py
python code/detect.py
# 육안 초안은 annotation_coordinates_draft.json에 기록한 뒤 build_annotations.py로 동결했다.
python code/build_annotations.py
python code/pose.py --split development,human_check
python code/classify_regions.py --split development,human_check --first
# 첫 요청의 224px 입력과 JSON 출력을 확인한 뒤 나머지 개발 조건을 실행했다.
python code/classify_regions.py --split development,human_check
# evaluation_lock.json 기록 후 평가 조건은 한 번만 실행했다.
python code/pose.py --split evaluation
python code/classify_regions.py --split evaluation
python code/summarize_c.py
# D는 scenes/공식 시각 대응 자료 부재로 보류했다.
python code/extract_tracking.py
python code/detect.py --manifest tracking_frame_manifest.json --out tracking_detector
python code/check_tracking_logic.py
python code/tracking.py
python code/present_results.py
python code/write_report.py
python code/package_review.py --preliminary
# F의 첫 검토 묶음 이후, 실행 전에 지정했던 여분 실험만 수행했다.
python code/extras.py
python code/tracking.py --extras
# 최종 육안 검수에서 발견한 입력 예외를 기록·고정한 후 별도 보정했다.
python code/human_mask_control.py
python code/present_results.py
python code/write_report.py
python code/verify_results.py
python code/package_review.py
```

위 목록의 `python`은 약칭이다. 공유 기본 Python에 패키지를 설치하거나 환경을 변경하지 않는다. 긴 실행은 `setsid`로 분리했고 evidence/*pid.json 및 *.log에 실제 PID/명령/시간을 기록했다. 같은 항목 재시도는 최대1회, 연속3회 실패·마감·디스크20GB 미만에서는 중단하도록 구현했다. GPU의 다른 계산 프로세스와 연구 잠금은 모델 로딩 전에 확인한다.

실제 총량은 VLM349회(본315+사전 지정 여분24+입력 보정10), 검출/자세315프레임(본313+입력 보정2)이다. C_SUMMARY.json의 호출 수315/vision42는 C 종료 시점의 스냅샷이다. 전체 작업 최종 수는 results/RUN_SUMMARY.json 및 evidence/final_verification.json을 확인한다. 새 가중치 다운로드는237,034,793바이트로 보정 시 추가 다운로드는 없다.

## 보존·이동

원 영상10개, 새 pose 가중치, 이전 실험에서 복사한 검출 가중치는 서버에만 있다. 맥 동기화는 videos, tools/ytdlp-lib, model_cache, __pycache__를 제외했다. 원본 영상 해시는 evidence/video_sources_original.json, 가중치 해시는 evidence/*weight*.json에 남겼다.

`ON_AIR_20261117_REVIEW.zip`은 보고서·대표/검수 이미지·CSV·코드·주요원시JSONL을 담은 내부 검토용이다. 전체 원본 영상, 모델 가중치, 전체 추출 PNG는 제외한다. ZIP의 Markdown 링크는 상대경로로 바꾸었으며 ZIP_SHA256SUMS로 ZIP 내부 파일을 확인한다. 폴더의 SHA256SUMS는 전체 맥 동기화 자료 기준이라 ZIP에 없는 파일도 포함한다. 원본 자료를 외부에 업로드하지 않았다. 노션에는 Markdown만 붙이지 말고 필요한 대표 이미지도 함께 올려야 한다.
