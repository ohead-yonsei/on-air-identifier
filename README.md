# ON-AIR — 중계 화면만 보고 선두권 말 번호 알아내기

경마 중계 AI가 지어내는 말(환각)을 줄이는 벤치마크 프로젝트 **ON-AIR**의 한 부분입니다. 가설은 "어려운 순간에 말을 틀리게 부르는 대신 확정 정보로 메우면 환각이 줄어든다"이고, 이 저장소는 그 확정 정보 가운데 **"지금 선두권에 몇 번 말이 있는가"를 화면에서 알아내는 식별기**를 만들고 시험한 기록입니다.

**정리 페이지: https://ohead-yonsei.github.io/on-air-identifier/**

## 한눈에

기수 모자색은 말 번호마다 정해져 있습니다(1 흰색, 2 노랑, 3 빨강, 4 검정, 5 파랑, 6 초록, 7 갈색, 8 분홍, 9 보라, 10 하늘, 11번부터 하늘색 줄무늬). 식별기는 화면에서 기수를 찾고, 모자색으로 번호를 읽고, 앞선 말을 골라 "선두권에 N번"이라고 말합니다.

| 무엇 | 결과 | 어디서 쟀나 |
|---|---|---|
| 말한 번호가 실제 상위 3두였던 비율 | 94.3% (1459개 중 1376개) | 처음 보는 경주일 엿새, 60개 경주 |
| 맞는 번호를 하나 이상 말한 시점 | 89.0% (874개 중 778개). 날에 따라 82~95% | 같은 엿새 |
| "선두는 N번" | 87.5% (344번 중 301번). 10월 5일·3일에서는 94.0%였음 | 9월 19일~8월 30일의 나흘, 선두 기준값 0.5 |
| "a번이 b번보다 앞" | 91.4% (336번 중 307번). 10월 5일·3일에서는 93.0% | 같은 나흘 |
| Qwen 없이 분류기만 쓴 빠른 구성 | A 94.3%, B 87.6%. 시점당 8.3초 → 2.3초 | 같은 나흘 |
| 틀린 번호가 들어간 중계 문장 | 화면만 본 모델 138개 → 식별기를 쓴 중계 7개 (140개 시점) | 2026-11-15, 11개 경주 |

- 위 표의 숫자는 **방송 순위 띠가 화면에 있는 구간**의 것입니다. 띠가 없는 구간(출발 직후, 마지막 30초쯤)은 공식 경주 기록을 정답으로 따로 쟀습니다: 띠 없이 경주 전체를 돌리면 A 92.6%, B 89.9%(안 쓴 엿새, 43개 경주). 결승 3초 전까지는 띠 구간과 같은 수준이고, 약한 곳은 출발 직후와 결승 순간입니다([finish2/RESULT_KO.md](experiments/onair_labelset_20261117/finish2/RESULT_KO.md)).
- 서울 경마장, 한국마사회 국제 채널(KRBC International) 화면에서만 만들고 확인했습니다.
- 자세한 한계는 정리 페이지와 각 실험의 `RESULT_KO.md`에 있습니다.

## 식별기 버전

문서에서 식별기를 가리킬 때는 아래 버전 이름을 씁니다.

| 버전 | 코드 이름 | 무엇이 달라졌나 | 상태 |
|---|---|---|---|
| v4 | — | 이 작업 이전의 식별기 (색 규칙). `ON-AIR_진행공유.md` | 지나간 것 |
| v5 | — | 화면 한 장에서 기수마다 번호를 냄: 기수 검출 + 색 규칙 + Qwen | v6부터의 부품 |
| v6 | S0 | 처음으로 "선두권 번호"를 냄: v5 + 모자색 분류기, 그림 3장 중 2장, 화면 아래쪽 두 상자 | 지나간 것 |
| v7 | T1 | "확정"을 분류기 확률 0.7 이상으로만, 작은 기수 검출, 3초·6초 전 이어받기 | 지나간 것 (손 규칙 기준선) |
| **v8** | L1 | 앞선 말 고르기를 순위 띠로 학습한 모델로 바꿈. 선두와 앞뒤 순서도 냄 | **프로그램의 기본값** |
| **v9** | F2 | 언어 모델(Qwen)을 뺌 + 빈 시점 메우기 + 선두 기준값 0.5. 시점당 8초 → 2초 | 프로그램의 선택 사항. 벤치마크 기준선과 띠 없는 구간 측정에 씀 |
| v10 (예정) | — | v9 + 공식 기록으로 배운 선두 모델 + 우승마 부르기 | 실험 코드에만 있음 (`finish2/`) |

코드 이름은 파일 이름과 출력(`"configuration": "F2"` 등), 예전 실험 기록에 그대로 남아 있습니다. F1은 v8에서 언어 모델만 뺀 중간 단계입니다. 고정해 둔 설계 문서와 예전 결과 문서는 고치지 않았으므로 코드 이름이나 "아침의 식별기"(= v6) 같은 표현이 남아 있습니다.

## 어디부터 읽나

| 파일 | 내용 |
|---|---|
| [benchmark/v0/README_KO.md](benchmark/v0/README_KO.md) | **벤치마크 초안 v0**: 띠를 가린 화면으로 중계 한 문장을 시키고 띠의 순위로 채점. 문항 1,606개, 채점기, 기준선 |
| [정리 페이지](https://ohead-yonsei.github.io/on-air-identifier/) (`docs/index.html`) | 왜 했나, 무엇을 만들었나, 결과, 뜻, 한계, 더 해볼 것 |
| [ON-AIR_GOAL.md](ON-AIR_GOAL.md) | 목표, 판정 기준, 단계별 진행 기록 전체 |
| [ON-AIR_진행공유.md](ON-AIR_진행공유.md) | 프로젝트 초기 공유 문서(식별기 v4까지) |
| [identifier/README_KO.md](experiments/onair_labelset_20261117/identifier/README_KO.md) | 식별기 프로그램 사용법과 출력 |
| [leader/RESULT_KO.md](experiments/onair_labelset_20261117/leader/RESULT_KO.md) | 선두권 안의 순서(선두, 앞뒤) |
| [night/RESULT_KO.md](experiments/onair_labelset_20261117/night/RESULT_KO.md) | 새 경주일 나흘에서 다시 잰 것, Qwen 없는 빠른 구성, 빈 시점 메우기, 순서를 넣은 중계 |
| [order2/RESULT_KO.md](experiments/onair_labelset_20261117/order2/RESULT_KO.md), [order2b/RESULT_KO.md](experiments/onair_labelset_20261117/order2b/RESULT_KO.md) | v8(L1)의 확인 결과 |
| [중계 스크립트 시연](https://ohead-yonsei.github.io/on-air-identifier/race-call-20261108-r11.html) (`docs/race-call-20261108-r11.html`), [설명](experiments/onair_labelset_20261117/live_script/README_KO.md) | 안 쓴 경주 하나를 띠 없이 화면만 보고 중계한 스크립트. 화면과 문장을 나란히 읽는 페이지 |
| [live_commentary/RESULT_KO.md](experiments/onair_labelset_20261117/live_commentary/RESULT_KO.md) | 식별기를 써서 실제로 중계해 본 시연 |
| [finish2/RESULT_KO.md](experiments/onair_labelset_20261117/finish2/RESULT_KO.md) | 띠 없는 구간을 공식 기록으로 다시 재기: 마지막 36초, 선두 모델, 우승마 부르기, 경주 전체 |
| [finish/RESULT_KO.md](experiments/onair_labelset_20261117/finish/RESULT_KO.md) | 결승선 구간의 처음 시도(결승 시각이 어긋나 있었음. finish2가 바로잡음) |

## 폴더

```
docs/                                  정리 페이지 (GitHub Pages)
benchmark/v0/                          벤치마크 초안 (문항, 정답, 채점기, 기준선)
ON-AIR_GOAL.md                         목표와 진행 기록
experiments/
  onair_labelset_20261117/             이 저장소의 중심. 라벨, 식별기, 실험 전부
    identifier/                        식별기 프로그램 (identify_l1.py)
    code/                              실험 코드
    order2/, order2b/, temporal/       v8(L1)과 v7(T1)의 시험
    leader/                            선두권 안의 순서
    night/                             새 나흘에서 다시 재기, 빠른 구성, 빈 시점 메우기
    finish/, da/, confirm/, weak/      결승선 구간, 검출 바꾸기, 통하지 않은 시도
    commentary2/, live_commentary/     중계 비교 실험과 시연
    team_page/                         정리 페이지 원본 (build.py로 docs/index.html을 만듦)
  onair_*_20261116/, onair_overnight_* 그 전 이틀의 탐색 실험 (진단, 모자·경주복, 야간 실험)
```

실험 폴더마다 **결과를 보기 전에 고정한 설계**(`DESIGN_KO.md`), 그 해시(`DESIGN_LOCK.sha256`), **결과와 한계**(`RESULT_KO.md`)가 짝으로 있습니다. 시험할 날은 내용을 보기 전에 규칙으로 정했고, 이미 본 날에서 좋아진 것은 새 날에서 확인하기 전에는 결론으로 쓰지 않았습니다.

## 실행

GPU 서버에서 실행합니다(RTX 3090에서 확인). Python 패키지(PyTorch, torchvision, OpenCV, scikit-learn)가 필요하고, 기본 구성은 로컬 Ollama의 Qwen 9B도 씁니다. 유료 API는 쓰지 않습니다.

```bash
python experiments/onair_labelset_20261117/identifier/identify_l1.py --video 경주.mp4 --out 결과.json
```

`--fast`를 붙이면 Qwen 없이 돕니다(시점당 2초쯤). `--fast --fallback --leader-tau 0.5`는 v9(코드 이름 F2)입니다. 벤치마크 시험용 엿새에서 A 93.2%, B 93.0%였고, 새 날 이틀로 미리 고정해 둔 확인 시험은 아직 하지 못했습니다.

모델 파일(모자 분류기, 앞선 말 고르기 모델, 선두 모델)과 영상은 저장소에 없습니다. 학습 코드는 `code/`에 있습니다(`wk1_train_weak.py`, `or2_train_final.py`, `ld2_train_final.py` 등).

## 저장소에 없는 것

공개 저장소라서 다음은 올리지 않았습니다.

- **방송 영상, 프레임, 모자 사진** 약 4,400장. 방송 화면은 정리 페이지에 넣은 캡처 4장만 있습니다. 화면이 들어 있는 검토·시연 페이지(`REVIEW.html`, `demo.html`, `commentary.html`)도 뺐습니다.
- **마주 경주복 그림**(`SEOUL_SILKS/`)과 마주 이름이 적힌 파일 2개.
- **모델 파일**(`*.pt`, `*.joblib`).
- 글 파일 안의 **서버 주소와 작업 폴더 이름**은 커밋할 때 가렸습니다(`.gitattributes`). 그래서 일부 파일은 폴더 안 `sha256sums`의 해시와 다를 수 있습니다.

일부 폴더의 `sheets`, `crops`, `frames`는 올리지 않은 이미지 폴더를 가리키는 링크라 저장소에서는 열리지 않습니다.

## 출처

화면 캡처 4장은 한국마사회 국제 채널(KRBC International) 영상에서 가져온 것으로, 연구 내용을 설명하는 데만 썼습니다. 순위 띠와 타이머 등 방송 그래픽은 채점 기준으로만 사용했습니다.
