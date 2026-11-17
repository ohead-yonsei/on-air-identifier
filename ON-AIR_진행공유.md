# ON-AIR 진행 공유 — 식별기·5조건 실험·순서 규칙 (로직과 코드 포함)

> **2026-11-16 기준 요약**
>
> - **된 것**: 화면 속 말이 몇 번인지 알아내는 식별기(v4). 그리고 번호만 알면 상자 좌표로 달리는 순서를 90% 이상 맞히는 규칙을 오늘 찾았습니다.
> - **안 된 것**: 로컬 8B 모델은 화면에서 순서를 읽지 못하고, 프롬프트에 준 표를 베껴 씁니다. 그래서 가설 실험 전체 실행은 멈춘 상태입니다.
> - **다음**: 프롬프트를 고친 뒤(v5) 제미나이로 같은 실험을 돌리고, 식별기와 순서 규칙을 이어 붙여 "확정 정보"로 넣습니다.

## 1. 무엇을 하려는가

경마 실황중계를 하는 AI가 **모르는 것을 지어내는 정도(할루시네이션)** 를 재는 벤치마크를 만들고 있습니다. Snorkel Open Benchmarks Grants 지원이 목표입니다.

**가설**

> 어려운 구간에서 이름을 틀리게 말하는 대신 확정 정보로 메우면 할루시네이션이 얼마나 줄어드는가.

용어를 풀면 이렇습니다.

- **어려운 구간**: 말들이 뭉쳐 있거나 멀리 찍혀서 누가 누구인지 화면만으로 알기 어려운 순간
- **확정 정보**: 틀릴 수 없는 정보. 출전표(말 이름, 기수, 모자색), 경주 거리, 주로 상태, 그리고 식별기가 "확실하다"고 표시한 말
- **할루시네이션**: 화면이나 주어진 정보로 뒷받침되지 않는 진술. 예: 3등인 말을 선두라고 부름, 없는 거리 숫자를 말함

## 2. 경마를 모르는 분을 위한 배경

- 말마다 번호가 있고, **기수의 모자색이 번호를 뜻합니다.** 1번 흰색, 2번 노랑, 3번 빨강, 4번 검정, 5번 파랑, 6번 초록, 7번 갈색, 8번 분홍, 9번 보라, 10번 하늘색. 11번부터는 하늘색 바탕에 줄무늬입니다.
- 기수가 입는 **경주복**은 마주(말 주인)마다 고유한 색과 무늬입니다. 마주별 경주복 그림 파일을 갖고 있습니다.
- 한국 경마는 **반시계 방향**으로 돕니다. 그래서 카메라 위치에 따라 말이 화면 왼쪽으로 달리기도, 오른쪽으로 달리기도, 카메라를 향해 오기도 합니다.
- 한국마사회 공개 API로 경주마다 **통과 지점별 순위와 시간**을 받을 수 있습니다. 다만 이것은 경주가 끝난 뒤에 나오는 기록이라, 중계 모델의 입력으로는 쓰지 않고 **채점용 정답**으로만 씁니다. 실제 중계자도 경주 중에는 이 기록을 받지 못합니다.

## 3. 전체 구조

| 단계 | 하는 일 | 방법 | 상태 |
| --- | --- | --- | --- |
| 1. 기수 찾기 | 화면에서 기수(사람) 상자를 찾음 | PyTorch 물체 검출 모델(Faster R-CNN) | 완료 |
| 2. 번호 맞히기 | 각 상자가 몇 번 말인지 판정 | 모자색 + 경주복 색을 출전표·API 정보와 대조 | 완료 (v4) |
| 3. 순서 계산 | 확인된 말들의 달리는 순서 | 상자 좌표에 규칙 적용 | 오늘 규칙 발견, 연결 전 |
| 4. 중계 생성 | 화면 + 정보를 받아 중계 문장 작성 | 영상 언어 모델(8B → 제미나이 예정) | 8B는 실패 |
| 5. 채점 | 문장을 쪼개 참/거짓 판정 | 판정 모델 + 코드 검사, 정답은 API | 동작, 검증 전 |

핵심 생각은 **"모르는 문제를 닫힌 문제로 바꾼다"** 입니다. 화면 속 말이 누구인지는 열린 질문이 아닙니다. 그 경주에 나온 8~16마리 중 하나이고, 각자의 모자색과 경주복은 미리 알려져 있습니다. 그래서 얼굴 인식 같은 어려운 학습 없이 색 대조만으로 풉니다.

## 4. 식별기 v4 로직

### 4-1. 순서

1. **기수 검출**: torchvision의 사전학습 Faster R-CNN(ResNet-50 FPN v2)으로 사람 상자를 찾습니다. 추가 학습은 하지 않았습니다.
2. **잘못 잡힌 상자 제거**: 화면 가장자리에서 잘린 상자, 왼쪽 위·아래의 방송 그래픽, 다른 상자 안에 들어간 작은 조각(말 머리, 팔)을 뺍니다.
3. **두 부위를 잘라냄**: 상자 위쪽 27%는 모자, 20~58% 구간은 몸통(경주복)입니다.
4. **색 이름 붙이기**: 픽셀을 13가지 색 이름(흰, 검정, 빨강, 밤색, 주황, 노랑, 초록, 하늘, 파랑, 보라, 분홍, 회색, 갈색)으로 바꿉니다. 배경(모래, 잔디, 펜스)은 상자 테두리 색과 비교해 제외합니다.
5. **점수 계산**: 출전한 번호마다 "모자색이 그 번호의 색과 맞는 정도"와 "몸통 색 비율이 그 마주의 경주복 그림과 맞는 정도"를 구합니다. 최종 점수는 모자 0.6, 경주복 0.4입니다.
6. **일대일 배정**: 한 화면에서 같은 번호가 두 번 나올 수 없으므로, 헝가리안 알고리즘으로 상자와 번호를 하나씩 짝짓습니다.
7. **확신 정도 표시**: 각 상자를 확정(sure) / 추정(guess) / 모름(unknown)으로 나눕니다. 점수가 높고 2등 후보와 차이가 크며 모자 증거가 있고 기수 키가 60픽셀 이상일 때만 확정입니다.

### 그림으로 보는 식별 과정

예시는 경주 4의 4코너 화면 한 장입니다. 아래 그림의 번호 ①~⑥은 위 순서와 이어집니다.

**①② 기수 찾기.** PyTorch(torchvision)에 들어 있는 Faster R-CNN 모델은 일반 사진 수십만 장으로 미리 학습되어 '사람'을 찾을 줄 압니다. 우리는 이 모델을 그대로 가져와 기수를 사람으로 찾게 했습니다. 경마 화면으로 따로 학습시키지 않았습니다.

![그림 1. 원본 프레임(왼쪽)과 PyTorch 검출 모델이 찾은 사람 상자(오른쪽)](figs/fig1_detect.jpg)

*그림 1. 원본 프레임(왼쪽)과 PyTorch 검출 모델이 찾은 사람 상자(오른쪽)*

**③④⑤ 색으로 대조하기.** 상자마다 위쪽은 모자, 가운데는 몸통으로 잘라 색 비율을 냅니다. 모자색은 번호 규칙과, 몸통 색은 그 말의 마주가 등록한 경주복 그림의 색 비율과 비교합니다. 경주복 그림은 한국마사회 API의 마주 번호로 연결합니다.

![그림 2. 기수 세 명의 모자·몸통 부위와 색 비율, 등록된 경주복 그림](figs/fig2_patches.jpg)

*그림 2. 기수 세 명의 예. 빨간 실선이 모자 부위, 파란 점선이 몸통 부위입니다. 가운데 막대는 그 부위에서 읽힌 색 비율, 오른쪽은 등록된 경주복 그림과 그 색 비율입니다.*

그림 2에서 볼 점은 두 가지입니다.

- 모자 부위는 한 가지 색이 뚜렷하게 나옵니다(3번 빨강, 5번 파랑, 6번 초록). 그래서 모자에 더 큰 비중(0.6)을 줍니다.
- 몸통 부위는 말 머리, 흙먼지, 옆 기수가 섞여 지저분합니다. 실제 화면의 몸통 색 비율은 등록 그림과 꽤 다릅니다. 그래서 경주복은 보조 증거(0.4)로만 씁니다.

**⑥ 배정과 확신 표시.** 모든 상자와 모든 출전 번호 사이의 점수표를 만든 뒤, 한 번호가 한 상자에만 가도록 전체 점수가 가장 높은 짝을 찾습니다(헝가리안 알고리즘). 그다음 증거가 충분한 상자만 '확정'으로 표시합니다.

![그림 3. 최종 결과](figs/fig3_result.jpg)

*그림 3. 최종 결과. 초록은 확정, 주황(?)은 추정입니다. 이 화면에서는 8개 상자 중 4개가 확정입니다.*

### 4-2. 실제 화면에 맞춘 보정

전문가(Jo)가 16장 화면에 직접 정답을 적었고, 보정을 하나씩 켜고 끄며 정답률을 비교했습니다. 남긴 것만 적습니다.

- **갈색 보정**: 갈색 모자는 그늘에서 검정으로, 햇빛에서 주황으로 읽힙니다. 4번(검정)과 7번(갈색)은 두 상자를 서로 비교해 더 갈색인 쪽을 7번으로 둡니다.
- **가장자리·그래픽 제거**, **모름 상태 추가**, **작은 기수는 확정에서 제외**.
- 흰색·하늘색·모자 가림 보정은 넣었더니 오히려 정답률이 떨어져서 뺐습니다.

### 4-3. 성능

| 항목 | 값 |
| --- | --- |
| 확정으로 표시한 것의 정답률 (정답 있는 16장) | 73%(v1) → 88%(v4) |
| 10개 경주 전체 상자 | 586개 |
| 그중 확정 / 추정 / 모름 | 222 / 266 / 98 |
| 확정이 하나도 없는 장면 | 35개 중 4개 |

> ⚠️ 88%는 보정에 쓴 바로 그 16장에서 잰 값입니다. 처음 보는 화면에서는 더 낮을 수 있습니다. 또 확정 비율이 38%라서, 화면에 있는 말의 절반 이상은 이름을 부를 수 없습니다.

## 5. 5조건 실험 설계

같은 장면, 같은 과제("누가 선두이고, 누가 바로 뒤이며, 주로 어디를 지나는지 서너 문장으로 말하라")를 조건만 바꿔 줍니다.

| 조건 | 모델이 받는 것 | 보려는 것 |
| --- | --- | --- |
| A | 영상 + 출전표 | 기준선 |
| R | A + 규칙(확실할 때만 이름을 말하고, 아니면 모른다고 말하라) | 규칙만으로 줄어드는가 |
| F | A + 확정 정보 | 정보만 주면 줄어드는가 |
| B | A + 확정 정보 + 규칙(불확실하면 확정 정보로 메워라) | 가설 그 자체 |
| C | B + 식별기가 확정한 말 목록(이 말들만 이름을 부를 수 있음) | 식별기를 붙였을 때 |

- 장면: 2026-09-06 서울 10개 경주, 통과 지점 35곳
- 입력: 각 지점 직전 8초를 초당 2장, 가로 768픽셀로 뽑은 16장
- 모델: qwen3-vl 8B (Ollama, 맥북 로컬). 한 번 호출에 약 58초, 전체 175회에 약 2시간 50분

### 어느 구간을 몇 장으로 잘랐고, 왜 그렇게 했는가

![그림 4. 한 장면에서 쓰는 프레임](figs/fig4_sampling.jpg)

*그림 4. 한 장면에서 쓰는 프레임. 파란 띠가 중계 모델에 주는 16장, 초록 세모가 식별기가 보는 2장입니다.*

| 항목 | 설정 | 그렇게 정한 이유 |
| --- | --- | --- |
| 장면의 끝 | 통과 지점에 맞춤 | 공식 순위와 시간이 있는 순간이 통과 지점뿐이라, 채점할 정답이 있는 곳에 장면을 맞췄습니다 |
| 구간 길이 | 직전 8초 | 말이 어느 쪽으로 움직이는지, 주로 어디를 지나는지 보려면 움직임이 필요합니다. 더 길면 그 사이에 순위가 바뀌어 정답이 모호해집니다 |
| 장수 | 초당 2장, 총 16장 | 로컬 8B 모델의 입력 한도와 속도 때문입니다. 16장에 한 번 호출이 약 58초입니다 |
| 크기 | 가로 768픽셀 | 같은 이유입니다. 크게 넣으면 장수를 줄여야 합니다 |
| 식별기 프레임 | 마지막 순간과 2.5초 전, 2장 | 순서는 마지막 순간 기준이고, 그 순간 가려진 말을 한 번 더 잡으려고 한 장을 더 봅니다. 두 장 중 한 번이라도 확정된 번호를 씁니다 |

> ⚠️ 이 설정은 논문에서 가져온 값이 아니라 장비와 시간에 맞춰 정한 실무적 선택입니다. 8초·16장·768픽셀을 다른 값과 비교하는 실험은 아직 하지 않았습니다. 특히 768픽셀에서는 멀리 찍힌 기수의 모자가 몇 픽셀밖에 안 되므로, 해상도를 올리면 결과가 달라질 수 있습니다.

**관련 문헌에 비춰 보면**

- 영상 이해 벤치마크에서는 영상에서 일정 간격으로 정해진 수의 프레임을 뽑아 여러 장의 그림으로 모델에 주는 방식이 표준입니다. Video-MME는 모델별로 뽑는 장수를 달리했고(예: GPT-4V 10장), 긴 입력을 받는 Gemini 1.5 Pro에는 짧은 영상에서 초당 1장을 줬습니다. [Video-MME 보충자료](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Fu_Video-MME_The_First-Ever_CVPR_2025_supplemental.pdf)
- 프레임 선택 연구들은 8장, 16장, 32장을 비교 기준으로 씁니다. 우리의 16장은 이 범위 안에 있습니다. [LENS](https://arxiv.org/pdf/2607.25125)
- 프레임을 어떻게 뽑느냐에 따라 같은 모델의 점수가 달라진다는 보고가 있습니다. 그래서 벤치마크로 낼 때는 추출 방식을 고정하고 명시해야 합니다. [Moments Lab, frame-sampling bias](https://research.momentslab.com/blog-posts/frame-sampling-vlm)
- 우리 구간은 8초로 매우 짧고 초당 2장이라, 위 벤치마크들의 초당 1장보다 촘촘합니다. 경마는 1초에 약 16미터를 달리므로 더 촘촘히 볼 필요가 있다고 판단했습니다.

### 채점 로직

1. 응답 첫 줄 `ORDER:`(모자색을 달리는 순서대로)는 **코드로만** 공식 순위와 비교합니다. 선두 일치, 상위 3두 일치 수, 두 마리씩 짝지은 앞뒤 일치율을 냅니다.
2. 나머지 문장은 판정 모델이 한 문장 한 주장으로 쪼개고 종류를 붙입니다: 위치 / 이름만 언급 / 주로 위치 / 배경 사실 / 모른다고 말함 / 막연한 말 / 기타.
3. 참·거짓은 코드가 정합니다. 위치 주장은 API 공식 순위와 비교하고, 확정 정보에 없는 거리 숫자가 나오면 거짓으로 칩니다.
4. **할루시네이션 = 틀린 위치 + 틀린 주로 위치 + 틀린 사실**, 호출 1회당 개수로 봅니다.

## 5-2. 중계 표현을 재료로 쓴 문항 (초기 파이프라인, 자기 교정 과제)

5조건 실험보다 먼저 만든 것으로, **전문 중계자(Jo)가 실제로 쓰는 영어 중계 표현을 재료로 넣어** 문항을 자동 생성하는 파이프라인입니다.

**무엇을 재는가.** 중계가 이미 틀린 말을 했을 때, 모델이 화면을 보고 그것을 바로잡는지 봅니다.

**어떻게 만드는가**

1. API의 통과 지점별 공식 순위로, 그 시점까지의 **맞는 중계 이력**을 중계 표현에 말 이름을 끼워 만듭니다.
2. 마지막 문장 하나만 **일부러 틀리게** 바꿉니다.
3. 모델에게 영상과 이 이력을 주고 중계를 이어 가게 한 뒤, 틀린 문장을 고치는지 · 그대로 이어 가는지 · 피해 가는지를 봅니다.

**재료로 넣은 중계 표현**

| 쓰임 | 표현 |
| --- | --- |
| 위치(통과 지점) | Early on, / Around the first turn, / Into the back straight, / Approaching the far turn, / Rounding the final bend, / Six hundred metres to go, / Four hundred left, / Two hundred to go! |
| 선두 | … takes them along. / … leads the way. / out in front, it's … / … is dictating terms on the lead. / bowling along in front is … |
| 2위 | Second, sitting on the leader's tail, is … / … is the nearest pursuer. / Tracking the leader is … |
| 3위 | … sits handy in third. / Then comes …, just off the pace. |
| 성급한 단정(오류용) | … has drawn away and won't be caught! / … is clear and coasting, this is all over! |
| 없는 추월(오류용) | Here comes …, flying down the outside! / … swoops around the field! / … is making ground hand over fist! |

맞는 이력에는 **순서를 단정하는 표현만** 썼습니다. 공식 기록으로 검증할 수 있는 것이 순서뿐이기 때문입니다. 착차, 페이스, 안쪽·바깥쪽 같은 표현은 기록으로 확인할 수 없어 넣지 않았습니다.

**오류를 심는 7가지 조건** (서울 2026-09-06 10개 경주, 총 237문항)

| 조건 | 내용 | 문항 수 |
| --- | --- | --- |
| control | 오류 없음 (맞는 중계를 괜히 고치지 않는지 확인) | 35 |
| identity_swap | 선두를 뒤쪽의, 모자색이 헷갈리기 쉬운 말로 바꿈 | 35 |
| rank_swap_1 | 선두를 2위 말로 바꿈 | 35 |
| rank_swap_3 | 선두를 4위 말로 바꿈 | 35 |
| rank_swap_6 | 선두를 7위 말로 바꿈 | 35 |
| premature_call | 이기지 못한 선두마를 "승부 끝"이라고 단정 | 27 |
| phantom_move | 뒤에 그대로 있는 말이 추월해 온다고 말함 | 35 |

**문항 예시** (경주 1, 결승선 600미터 전, control)

> Early on, out in front, it's Daon Hunter. Tracking the leader is Inferno. Myeongseong Party sits handy in third.
> Six hundred metres to go, Inferno takes them along. Tracking the leader is Myeongseong Party. Then comes Daon Hunter, just off the pace.

**상태와 지금 실험과의 관계**

- 문항 생성까지는 실제 데이터로 돌렸습니다. 채점기는 "누가 선두다" 주장만 이해하고, 성급한 단정과 없는 추월은 아직 채점하지 못합니다.
- 지금의 5조건 실험은 이 표현을 모델에 주지 않고 모델이 자유롭게 쓰게 합니다. 두 실험은 재는 것이 다릅니다. 5조건 실험은 "처음부터 틀리지 않는가", 이 문항은 "틀린 것을 고치는가"입니다.
- 표현이 6종류 26개(위치 11, 선두 5, 2위 3, 3위 2, 오류용 5)라 문체가 단조롭습니다. 실제 중계 전사문에서 표현을 더 뽑아 늘릴 수 있습니다.

## 6. 지금까지의 결과

### 6-1. 8B 모델

- 위치 진술 정확도는 약 45%입니다.
- 규칙만 주어서는 "모르겠다"고 하지 않습니다.
- 확정 정보를 주면 그대로 옮겨 적는 경향이 있고, 할루시네이션이 이름에서 거리 숫자 쪽으로 옮겨 갑니다.
- 모자색 자체는 보지만, 화면 왼쪽부터 읽은 순서를 달리는 순서인 것처럼 말합니다.
- **오늘 확인한 결정적 문제**: 새 프롬프트로 2장면을 돌렸더니 ORDER 줄이 `white, yellow, red, black, blue, …` 즉 출전표의 1번~11번 순서 그대로였고, 서로 다른 두 장면에서 글자까지 같았습니다. 화면을 읽은 것이 아니라 표를 베낀 것입니다. 표를 번호순으로 주고 전체 순서를 쓰라고 한 프롬프트 설계에도 원인이 있습니다.

### 6-2. "어려운 구간"을 시간 차로 표시해 본 결과

API 통과 시간으로 35장면을 표시했습니다.

- 2위가 선두와 0.1초 이내인 장면이 26개(74%), 그중 13개는 기록상 동률입니다.
- 선두가 뚜렷한(2위가 0.2초 이상 뒤) 장면은 9개뿐입니다.

즉 대부분의 장면에서 "누가 선두냐"는 공식 기록으로도 가를 수 없는 차이입니다. 공식 1위만 정답으로 치면 맞게 본 것도 오답이 되므로, **선두권 인정 기준**(선두와 일정 시간 이내면 정답)을 채점에 넣어야 합니다.

이 표시로 8B의 오류를 나눠 보면 일관된 경향이 없었습니다.

| 구분 | 장면 수 | A 정확도 | B 정확도 |
| --- | --- | --- | --- |
| 선두 뚜렷 | 9 | 55% | 42% |
| 선두 붙음 | 26 | 39% | 49% |

8B는 특정 구간에서만 틀리는 것이 아니라 전반적으로 틀립니다. 또 화면에서 상자들이 겹쳐 보이는 정도는 실제 시간 차와 거의 관계가 없었습니다(상관 0.17). 화면에서 붙어 보이는지는 카메라 각도가 좌우합니다.

선두 무리(선두와 0.3초 이내)의 크기 분포는 다음과 같습니다.

| 선두 무리 크기 | 1두 | 2두 | 3두 | 4두 | 5두 | 6두 |
| --- | --- | --- | --- | --- | --- | --- |
| 장면 수 (총 35) | 3 | 13 | 9 | 6 | 1 | 3 |

### 6-3. 오늘의 발견: 순서는 규칙으로 풀린다

전문가 정답이 있는 15장에서, 상자 좌표만으로 달리는 순서를 맞힐 수 있는지 쟀습니다. 두 마리씩 짝지어 앞뒤가 공식 순위와 맞는 비율이며, 0.1초 이내로 붙은 쌍은 제외했습니다.

| 판정 기준 | 코너 정면 화면 6장 | 옆 화면 9장 | 전체 |
| --- | --- | --- | --- |
| 좌우 위치 (방향은 정답을 보고 고름) | 76% | 93% | 86% |
| **세로 위치: 화면 아래쪽이 앞 (고정 규칙)** | **96%** | 90% | **92%** |
| 상자 크기: 큰 것이 앞 | 68% | 48% | 56% |
| 샷별로 나눔 (정면은 세로, 옆은 좌우) | 96% | 93% | 94% |

- 카메라가 높은 곳에서 내려다보기 때문에, 카메라에 가까운 말일수록 화면 아래에 찍힙니다. 코너를 돌아 나오는 화면에서는 앞선 말이 카메라에 더 가깝습니다.
- 세로 규칙은 화면마다 방향을 정할 필요가 없는 고정 규칙입니다.
- 샷별로 나누는 방식은 숫자가 조금 높지만, 옆 화면의 93%는 진행 방향을 정답을 보고 고른 값이라 부풀려져 있습니다. 방향 판정을 한 번 틀리면 순서 전체가 뒤집히는 위험도 있습니다. 그래서 **지금은 세로 규칙 하나로 가고, 샷별 방식은 뒤로 미뤘습니다.**

> ⚠️ 이 표는 3개 경주 15장, 서울 경마장 카메라에서만 확인한 값이고, 번호는 식별기 출력이 아니라 사람이 적은 정답을 넣었습니다. 실제 식별기 번호로 다시 재야 합니다.

![그림 5. 세로 규칙의 예](figs/fig5_order.jpg)

*그림 5. 세로 규칙의 예(경주 4, 4코너 정면 화면). 상자 아랫변이 화면 아래에 있을수록 앞선 말로 봅니다. 규칙이 낸 순서는 3-6-8-1-2-5-7, 공식 순위는 6-8-3-2-1-5-7입니다. 앞의 세 마리(6·8·3번)는 기록상 동률이라 서로의 순서는 가릴 수 없고, 그 뒤 그룹과의 앞뒤는 모두 맞았습니다.*

화면별 결과입니다. 숫자는 0.1초 넘게 벌어진 쌍 중 앞뒤가 맞은 비율입니다.

| 화면 | 샷 | 말 수 | 좌우 위치 | 세로 위치 |
| --- | --- | --- | --- | --- |
| R1_37.4_b | 옆 | 10 | 89% | 83% |
| R2_33.6_b | 옆 | 6 | 71% | 79% |
| R2_44.9_a | 옆 | 9 | 100% | 100% |
| R2_44.9_b | 옆 | 9 | 97% | 88% |
| R4_30.6_a | 미확인 | 6 | 77% | 100% |
| R4_80.7_b | 옆 | 6 | 100% | 92% |
| R4_88.4_a | 옆 | 7 | 100% | 82% |
| R4_88.4_b | 옆 | 6 | 100% | 100% |
| R2_50.8_a | 코너 정면 | 10 | 63% | 92% |
| R2_50.8_b | 코너 정면 | 7 | 94% | 100% |
| R4_43.4_a | 코너 정면 | 7 | 89% | 100% |
| R4_43.4_b | 코너 정면 | 6 | 77% | 100% |
| R4_80.7_a | 코너 정면 | 8 | 78% | 91% |
| R4_95.5_b | 코너 정면 | 7 | 71% | 100% |

R4_30.6_b는 벌어진 쌍이 하나뿐이라 표에서 뺐습니다. R4_30.6_a는 화면을 직접 확인하지 않아 집계에서는 옆 화면으로 넣었습니다.

### 6-4. 지금까지 가장 성공한 모델

식별기 v4입니다. 중계 문장을 만드는 쪽에서는 아직 성공한 모델이 없습니다.

## 7. 막힌 지점

1. **8B는 순서를 읽지 못한다.** 모델 한계와 프롬프트 결함이 섞여 있어, 프롬프트를 고친 뒤 큰 모델로 다시 봐야 구분됩니다.
2. **식별기 확정 비율 38%.** 순서 규칙이 좋아도 번호를 모르는 말은 부를 수 없습니다.
3. **나란히 뛰는 말.** 선두권의 74%가 0.1초 이내라 화면으로도 기록으로도 앞뒤를 가릴 수 없습니다.
4. **채점기 미검증.** 판정 모델이 문장을 제대로 분류하는지 사람이 40문장 정도 대조해야 합니다.
5. **데이터 양.** 하루 10개 경주, 정답 화면 16장입니다.
6. 데이터 품질… 영상 화질이 작아요 유튜브에서 받아서 그런데요, 고화질로 받으면 모델을 돌려보는데 비용과 시간이 걸려서 ㅜㅜ 난제입니다.

## 8. 다음 단계

- [ ] 프롬프트 v5: 모자색 표를 장면마다 섞기, ORDER는 앞 3두만 쓰고 `unclear` 허용, 베낌 검사, 선두권 인정 기준(0.1초 / 0.3초 둘 다 계산)
- [ ] 같은 프롬프트로 제미나이 실행 (유료 등급 키 사용. 무료 등급은 보낸 내용이 제품 개선에 쓰일 수 있어 중계 화면에는 부적절)
- [ ] 식별기 v4 번호 + 세로 규칙으로 계산한 순서를 공식 순위와 대조
- [ ] 그 순서를 C 조건의 확정 정보에 추가
- [ ] 채점기 검증 40문장
- [ ] 35장면 샷 종류 라벨 (샷별 방식 재검토용, 보류)
- [ ] 다른 날짜 경주 추가, 두 번째 라벨러, 한국마사회 권리 확인

## 9. 데이터와 보안 원칙

- 공개할 데이터는 **무음 영상 + 시간 표시 + 라벨**만입니다. 중계 음성은 넣지 않습니다.
- API 키(한국마사회 서비스 키, 제미나이 키)는 각자 컴퓨터의 환경변수에만 두고, 코드나 문서나 대화에 적지 않습니다.
- 원본 영상 폴더는 읽기만 하고 수정하지 않습니다.
- 경주 후 기록(통과 순위·시간)은 채점에만 쓰고 모델 입력에는 넣지 않습니다.

## 10. 실행 방법

준비물: Python 가상환경(torch, torchvision, opencv-python, scipy, numpy, ollama), ffmpeg, Ollama와 `qwen3-vl:8b-instruct` 모델.

```bash
# 1) 식별기: 이미 뽑아 둔 프레임에 대해 다시 실행 (인자 = 경주복 그림 폴더)
cd ~/onair_work/silkid
bash rerun.sh ~/Downloads/SEOUL_SILKS
# 결과: out4/R{경주번호}/visible.json, 확인용 그림 out4/sheet_R*.jpg

# 2) 5조건 실험: 먼저 2장면만
cd ~/onair_work/ab2
source ~/onair_work/onair/.venv/bin/activate
python3 run_ab2.py scenes.json ~/onair_work/onair/clips responses.jsonl --limit 2

# 3) 전체 실행 (약 2시간 50분, 잠자기 방지)
caffeinate -i python3 run_ab2.py scenes.json ~/onair_work/onair/clips responses.jsonl

# 4) 채점
python3 score_ab2.py scenes.json responses.jsonl scored.jsonl
```

파일 형식

- `owners.json`: `"경주번호_말번호": [마주번호, 마주명, 말이름, 기수명]`
- `visible.json`: 프레임마다 상자 목록. 항목은 `no`(번호), `state`(sure/guess/unknown), `score`, `x`, `h_px`, `box`
- `scenes.json`: 장면마다 `race`, `scene`, `point`(통과 지점), `video_end`, `racecard`, `facts`(확정 정보), `passed`, `ahead`, `order_now`(공식 순위)

## 참고 문헌

- Ren, He, Girshick, Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS 2015. [논문](https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html) — 식별기의 기수 검출에 쓴 모델
- Kuhn. The Hungarian Method for the Assignment Problem. Naval Research Logistics Quarterly, 1955 — 상자와 번호의 일대일 배정
- Fu 외. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. CVPR 2025. [보충자료](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Fu_Video-MME_The_First-Ever_CVPR_2025_supplemental.pdf) — 프레임을 일정 간격으로 뽑아 평가하는 방식
- LENS: Adaptive Spatio-Temporal Zooming for Keyframe Sampling in Long-Form Videos. [arXiv](https://arxiv.org/pdf/2607.25125) — 8·16·32장 비교 기준
- Moments Lab. Frame-sampling bias in video benchmarks. [글](https://research.momentslab.com/blog-posts/frame-sampling-vlm) — 추출 방식이 점수를 바꾼다는 보고

## 11. 코드

### 11-1. 식별기 match.py (1/3) — 색 이름 붙이기와 점수

```python
"""Closed-set rider identification: silks colours + cap colour, one-to-one assignment per frame.
usage: python match.py <frames_glob> <race_no> <owners.json> <silks_dir> <outdir>"""
import sys, json, glob, os
import numpy as np, cv2, torch
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import nms

import pickle
FX = set(os.environ.get("FX","brown,rel47,edge,unknown,small").split(","))      # which fixes are on (for ablation against the labelled frames)
PAL = ["white","black","red","maroon","orange","yellow","green","skyblue","blue","purple","pink","grey","brown"]
CAP = {1:"white",2:"yellow",3:"red",4:"black",5:"blue",6:"green",7:"maroon",8:"pink",9:"purple",10:"skyblue"}

def quant(bgr, ref=False):
    """pixel-wise named colour (calibrated on the 2026-09-06 Seoul broadcast, which is washed out); index into PAL"""
    hsv = cv2.cvtColor(bgr.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3).astype(int)
    h,s,v = hsv[:,0],hsv[:,1],hsv[:,2]
    out = np.full(len(hsv), PAL.index("grey"))
    def put(n,m): out[m] = PAL.index(n)
    warm = (h>=15)&(h<=45)
    warm = (h>=20)&(h<=45)
    # low-saturation bright pixels: sand is pinkish (hue >=125 or <15), a white cap/silk is cool (45-125), a pale yellow cap is warm (15-45)
    if ref: put("white",(s<30)&(v>185))
    else:   put("white",(s<30)&(v>228)&(h>=45)&(h<125))
    put("yellow",warm&(s>=10)&(v>150))
    put("black",(v<70)&(s<90)); put("blue",(v<110)&(s>=60)&(h>=105)&(h<128))              # navy
    put("red",(s>45)&((h<8)|(h>=168))&(v>110)); put("maroon",(s>45)&((h<8)|(h>=160))&(v>=60)&(v<=110))
    (put("brown",(h>=8)&(h<28)&(s>=45)&(v>=60)&(v<=185)), put("orange",(s>120)&(h>=8)&(h<20)&(v>170))) if "brown" in FX else put("orange",(s>70)&(h>=8)&(h<18)&(v>140))      # tan / brown cap in sunlight; saturated bright = orange
    put("green",(s>35)&(h>=38)&(h<85)&(v>70)); put("skyblue",(s>30)&(h>=85)&(h<108)&(v>140))
    put("blue",(s>70)&(h>=108)&(h<124)&(v>=110)); put("purple",(s>55)&(h>=124)&(h<146)&(v>80)); put("pink",(s>38)&(h>=146)&(h<168)&(v>130))
    return out
def fgmask(patch, thr=20):
    """pixels that differ from the patch border colour (sand / turf / rail behind the rider)"""
    L = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.median(np.concatenate([L[0,:], L[:,0], L[:,-1]]), 0)
    return (np.linalg.norm(L-bg, axis=2) > thr).reshape(-1)
A = np.eye(len(PAL))
for a,b,v in [("brown","maroon",.4),("brown","orange",.4),("brown","black",.3),("red","maroon",.6),("skyblue","blue",.6),("orange","yellow",.5),("orange","red",.5),("maroon","pink",.25),("purple","blue",.3),("maroon","purple",.25)]:
    A[PAL.index(a),PAL.index(b)] = A[PAL.index(b),PAL.index(a)] = v
G = PAL.index("grey")
def hist(idx, w=None, torso=None):
    hh = np.bincount(idx, weights=w, minlength=len(PAL)).astype(float)
    known = 1 - hh[G]/max(hh.sum(),1e-6); hh[G] = 0
    if torso: hh[PAL.index("white")] *= 0.5          # every jockey wears white breeches; rails are white
    return hh/max(hh.sum(),1e-6), known
def soft(a,b):
    """how much of b's colour mass is explained by a (with near-colour credit)"""
    return float(np.minimum(b, A@a).sum())

def cap_score(ch, n, ref=None):
    """share of the (foreground) cap patch that has this number's cap colour. Calibrated on the user's corrections (2026-11-16):
    brown caps read black in shade, shaded red caps read maroon, sky-blue striped caps read blue+white."""
    I = PAL.index; w, bk, br = ch[I("white")], ch[I("black")], ch[I("brown")]; sky, bl = ch[I("skyblue")], ch[I("blue")]
    if n == 1:
        v = w
        if "white" in FX:
            for c in ("skyblue","yellow","blue"):
                if ref is None or ref[I(c)] < 0.15: v -= 1.5*ch[I(c)]
    elif n == 3 and "brown" in FX: v = ch[I("red")] + 0.8*ch[I("maroon")]
    elif n == 4 and "brown" in FX: v = bk + 0.3*br
    elif n == 7 and "brown" in FX: v = br + 0.45*bk
    elif n == 5 and "sky" in FX: v = bl*max(0.0, 1-1.5*w) + 0.3*sky
    elif n == 10 and "sky" in FX: v = (sky + 0.5*bl)*max(0.0, 1-1.2*w)
    elif n >= 11 and "sky" in FX:
        base = sky + bl; stripe = ch[I(CAP[n-10])] if n != 15 else bl
        v = min(1.0, base + stripe) if (base > 0.25 and stripe > 0.08) else 0.3*base
    elif n >= 11: v = (sky + 0.6*ch[I(CAP[n-10])]) if sky > 0.12 else 0.0
    else: v = ch[I(CAP[n])] + (0.5*ch[I("red")] if CAP[n]=="maroon" else 0)
    return float(np.clip(v*1.5, 0, 1))

def silk_ref(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.shape[2]==4: m = im[:,:,3].reshape(-1)>128; px = im[:,:,:3].reshape(-1,3)[m]
    else: px = im.reshape(-1,3)
    return hist(quant(np.ascontiguousarray(px), ref=True))[0]
```

### 11-2. 식별기 match.py (2/3) — 검출과 부위 자르기

위 코드에 바로 이어지는 `main` 함수의 앞부분입니다.

```python
def main(pattern, race, owners, silkdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    ow = json.load(open(owners)); field = {}
    for k,v in ow.items():
        r,n = map(int,k.split("_"))
        if r!=race: continue
        fs = sorted(glob.glob(f"{silkdir}/{v[0]}_*"))
        if not fs: print("NO SILK FILE", n, v); continue
        field[n] = {"ref": silk_ref(fs[0]), "horse": v[2], "jockey": v[3], "owner": v[1]}
    nums = sorted(field)
    alike = {n: [m for m in nums if m != n and 0.5*soft(field[n]["ref"],field[m]["ref"]) + 0.5*soft(field[m]["ref"],field[n]["ref"]) >= 0.9] for n in nums}
    for n in nums: print(n, field[n]["horse"], {PAL[i]:round(x,2) for i,x in enumerate(field[n]["ref"]) if x>0.05})
    model = None
    res = {}
    for f in sorted(glob.glob(pattern)):
        img = cv2.imread(f); H,W = img.shape[:2]
        cf = os.path.join(os.environ["DETCACHE"], os.path.basename(f)+".pkl") if os.environ.get("DETCACHE") else None
        if cf and os.path.exists(cf): b = pickle.load(open(cf,"rb"))
        else:
            if model is None: model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.5).eval()
            t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255
            with torch.no_grad(): r = model([t])[0]
            m = r["labels"]==1; b,s = r["boxes"][m], r["scores"][m]
            keep = nms(b,s,0.45); b = b[keep].numpy()
            if cf: pickle.dump(b, open(cf,"wb"))
        riders = []; medw = float(np.median([q[2]-q[0] for q in b])) if len(b) else 0
        for x1,y1,x2,y2 in b:
            cx,cy = (x1+x2)/2,(y1+y2)/2
            if "edge" in FX and (x1 <= 3 or x2 >= W-3) and (x2-x1) < 0.75*medw: continue                       # rider cut off by the frame edge
            if "edge" in FX and cx < 0.19*W and (cy < 0.17*H or cy > 0.83*H): continue                         # race title / clock graphics
            bw,bh = x2-x1,y2-y1
            if bh<16 or bh>H*0.6 or y1 < H*0.05: continue
            tor = img[int(y1+bh*0.20):int(y1+bh*0.58), int(x1+bw*0.15):int(x2-bw*0.15)]
            cap = img[int(y1):int(y1+bh*0.24), int(x1+bw*0.15):int(x2-bw*0.15)]
            if tor.size==0 or cap.size==0: continue
            th,tw = tor.shape[:2]; yy,xx = np.mgrid[0:th,0:tw]
            wt = np.exp(-(((xx-tw/2)/(tw/2+1e-6))**2+((yy-th/2)/(th/2+1e-6))**2)).reshape(-1)   # centre weighting
            cap = img[int(y1):int(y1+max(4,bh*0.27)), int(x1+bw*0.08):int(x2-bw*0.08)]
            box_full = img[int(y1):int(y2), int(x1):int(x2)]; Lb = cv2.cvtColor(box_full, cv2.COLOR_BGR2LAB).astype(np.float32)
            bgc = np.median(np.concatenate([Lb[0,:], Lb[:,0], Lb[:,-1]]), 0)                 # background colour around this rider
            def fg(p): return (np.linalg.norm(cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32)-bgc, axis=2) > 20).reshape(-1).astype(float)
            capbox = (x1+bw*0.08, y1, x2-bw*0.08, y1+max(4,bh*0.27)); ca = max(1.0,(capbox[2]-capbox[0])*(capbox[3]-capbox[1])); cov = 0.0
            for q in b:
                if abs(q[0]-x1)<1 and abs(q[1]-y1)<1: continue
                cov = max(cov, max(0,min(capbox[2],q[2])-max(capbox[0],q[0]))*max(0,min(capbox[3],q[3])-max(capbox[1],q[1]))/ca)
            riders.append({"capcov":float(cov), "box":[float(a) for a in (x1,y1,x2,y2)], "h":float(bh),
                           "th":hist(quant(np.ascontiguousarray(tor)),wt*fg(tor),torso=True), "ch":hist(quant(np.ascontiguousarray(cap)),fg(cap)+1e-9)})
        S = np.zeros((len(riders), len(nums))); C = np.zeros_like(S); K = np.zeros_like(S)
        for i,rd in enumerate(riders):
            for j,n in enumerate(nums):
                ref = field[n]["ref"]
                th,tk = rd["th"]; ch,ck = rd["ch"]
                silk = 0.5*soft(th,ref) + 0.5*soft(ref,th)
                cs = cap_score(ch, n, ref) * min(1.0, ck*4) * (0.0 if ("capcov" in FX and rd["capcov"] > 0.4) else 1.0)   # cap patch sits on another rider: do not trust it
                S[i,j] = (0.4*silk + 0.6*cs) * min(1.0, 0.4 + tk*2); C[i,j] = cs; K[i,j] = silk*min(1.0, tk*3)
```

### 11-3. 식별기 match.py (3/3) — 일대일 배정과 확신 정도

`main` 함수의 나머지입니다 (프레임 반복문 안쪽에서 이어집니다).

```python
        out = []
        if len(riders):
            def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
            def inter(a,b): return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            frag = set()                                   # small box mostly inside a bigger one: horse head, arm, cap ...
            for i,a in enumerate(riders):
                for k,b in enumerate(riders):
                    if i!=k and area(a["box"]) < 0.6*area(b["box"]) and inter(a["box"],b["box"]) > 0.5*area(a["box"]): frag.add(i)
            med = float(np.median([area(r["box"]) for r in riders]))
            for i,a in enumerate(riders):                  # much smaller than the others and touching a bigger box: horse head with blinkers etc.
                if area(a["box"]) < 0.45*med and any(i!=k and area(b["box"]) > area(a["box"]) and inter(a["box"],b["box"]) > 0 for k,b in enumerate(riders)): frag.add(i)
            main = [i for i in range(len(riders)) if i not in frag]
            pairs = []
            if main:
                ri,cj = linear_sum_assignment(-S[main]); pairs += [(main[a],b) for a,b in zip(ri,cj)]
            used = {b for _,b in pairs}; rest = [j for j in range(len(nums)) if j not in used]; fl = sorted(frag)
            if fl and rest:
                ri,cj = linear_sum_assignment(-S[np.ix_(fl,rest)]); pairs += [(fl[a],rest[b]) for a,b in zip(ri,cj)]
            asg = dict(pairs); inv = {j:i for i,j in pairs}                                   # black (4) vs brown (7): decide relative to each other
            if "rel47" in FX and 4 in nums and 7 in nums:
                j4,j7 = nums.index(4),nums.index(7); I_ = PAL.index
                def brownness(i): ch = riders[i]["ch"][0]; return ch[I_("brown")]/(ch[I_("brown")]+ch[I_("black")]+1e-6), ch[I_("brown")]+ch[I_("black")]
                if j4 in inv and j7 in inv:
                    (b4,d4),(b7,d7) = brownness(inv[j4]),brownness(inv[j7])
                    if d4 > 0.3 and d7 > 0.3 and b4 - b7 > 0.25: asg[inv[j4]],asg[inv[j7]] = j7,j4
                pairs = list(asg.items())
            for i,j in pairs:
                row = np.sort(S[i])[::-1]; margin = row[0]-row[1] if len(row)>1 else row[0]
                sure = S[i,j] >= 0.40 and (S[i,j] >= row[0]-1e-9 and margin >= 0.08) and C[i,j] >= 0.25 and C[i,j] >= C[i].max()-0.15 and i not in frag and ("small" not in FX or riders[i]["h"] >= 60*W/1280)
                twins = alike[nums[j]]
                unknown = "unknown" in FX and (S[i,j] < 0.30 or (C[i,j] < 0.15 and len(twins) > 0))                 # weak evidence, or silks shared with another runner and no cap evidence
                if unknown: sure = False
                out.append({"state": "sure" if sure else ("unknown" if unknown else "guess"), "alike": twins,
                            "no":nums[j], "horse":field[nums[j]]["horse"], "jockey":field[nums[j]]["jockey"],
                            "score":round(float(S[i,j]),2), "margin":round(float(margin),2), "sure":bool(sure),
                            "x":round((riders[i]["box"][0]+riders[i]["box"][2])/2), "h_px":round(riders[i]["h"]), "box":riders[i]["box"]})
                out[-1]["best"] = nums[int(np.argmax(S[i]))]; out[-1]["bestscore"] = float(S[i].max())
            held = {}
            for d in out:
                if d["no"]==d["best"]: held[d["no"]] = d
            kept = []
            for d in out:
                o = held.get(d["best"])
                if o is not None and o is not d:
                    bx,ox = d["box"],o["box"]; gapx = max(bx[0],ox[0]) - min(bx[2],ox[2]); gapy = max(bx[1],ox[1]) - min(bx[3],ox[3])
                    if gapx < 0.5*(ox[2]-ox[0]) and gapy < 0.5*(ox[3]-ox[1]): continue      # same rider, split box
                if d["bestscore"] < 0.22: continue                                           # nothing rider-coloured in it
                kept.append(d)
            out = kept
        out.sort(key=lambda d:d["x"])
        for d in out:
            x1,y1,x2,y2 = map(int,d["box"]); c = (0,255,0) if d["sure"] else ((160,160,160) if d["state"]=="unknown" else (0,165,255))
            cv2.rectangle(img,(x1,y1),(x2,y2),c,2); tx = "?" if d["state"]=="unknown" else (f"{d['no']}" + ("" if d["sure"] else "?"))
            cv2.putText(img,tx,(x1,max(22,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),5); cv2.putText(img,tx,(x1,max(22,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,1.0,c,2)
        name = os.path.basename(f); cv2.imwrite(f"{outdir}/{name}", img, [cv2.IMWRITE_JPEG_QUALITY,88])
        res[name] = [{k:v for k,v in d.items() if k not in ("best","bestscore")} for d in out]
        hs = [d["h_px"] for d in out]
        print(f"{name:16} riders={len(out):2} h={np.median(hs) if hs else 0:4.0f}px  " + " ".join((f"{d['no']}{'' if d['sure'] else '?'}({d['score']})" if d["state"]!="unknown" else "?") for d in out))
    json.dump(res, open(f"{outdir}/visible.json","w"), ensure_ascii=False, indent=1)
main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
```

### 11-4. 식별기 실행 스크립트 rerun.sh

```bash
#!/bin/bash
# re-run identification on frames already extracted. usage: bash rerun.sh <silks_dir>
HERE="$(cd "$(dirname "$0")" && pwd)"; rm -rf "$HERE/out4"; mkdir -p "$HERE/out4"
for r in 1 2 3 4 5 6 7 8 9 10; do
  ls "$HERE"/frames/R${r}_*.jpg >/dev/null 2>&1 || continue
  echo "== race $r"
  python3 "$HERE/match.py" "$HERE/frames/R${r}_*.jpg" $r "$HERE/owners.json" "$1" "$HERE/out4/R$r" 2>&1 | grep "riders="
done
python3 "$HERE/sheet.py" "$HERE/out4"
```

### 11-5. 확인용 그림 sheet.py

```python
"""contact sheets (one per race) + summary counts"""
import sys, glob, json, os, cv2, numpy as np
out = sys.argv[1]; tot = sure = 0
for d in sorted(glob.glob(f"{out}/R*"), key=lambda p: int(os.path.basename(p)[1:])):
    fs = sorted(glob.glob(f"{d}/*.jpg")); ims = []
    for f in fs:
        im = cv2.resize(cv2.imread(f), (960, 540)); cv2.putText(im, os.path.basename(f)[:-4], (300, 34), 0, 0.9, (0, 255, 255), 2); ims.append(im)
    while len(ims) % 2: ims.append(np.zeros((540, 960, 3), np.uint8))
    for k in range(0, len(ims), 6):
        part = ims[k:k+6]; rows = [np.hstack(part[i:i+2]) for i in range(0, len(part), 2)]
        cv2.imwrite(f"{out}/sheet_{os.path.basename(d)}_{k//6+1}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    v = json.load(open(f"{d}/visible.json")); n = sum(len(x) for x in v.values()); s = sum(1 for x in v.values() for y in x if y["sure"])
    tot += n; sure += s; print(f"{os.path.basename(d):4} frames={len(v):2} boxes={n:3} sure={s:3}")
print(f"total boxes={tot} sure={sure}\nsheets: {out}/sheet_R*.jpg  (green = sure, orange ? = unsure)")
```

### 11-6. 5조건 실험 run_ab2.py (ab2_v4)

> ⚠️ 이 버전의 프롬프트에는 알려진 결함이 두 가지 있습니다. 출전표를 번호순으로 주고 전체 순서를 요구해서 8B가 표를 베낍니다. 그리고 "In a head-on shot the horse nearest the camera is in front" 문장은 검증 없이 넣은 단순화입니다. v5에서 고칠 예정입니다.

```python
"""Five-arm test (2026-11-16). Same task in every arm: say who leads, who is close behind, and where the field is.
  A = video + racecard
  R = A + rule (name only when sure; otherwise say you cannot tell; no unseen numbers or situations)
  F = A + verified facts (no rule)
  B = A + verified facts + rule (when unsure, fill with a verified fact)
  C = B + CONFIRMED ON SCREEN list from the identifier (silkid v4): only listed horses may be named
Local only (Ollama). usage: python3 run_ab2.py scenes.json <clips_dir> responses.jsonl --ident ~/onair_work/silkid/out4 [--conds A,R,F,B,C] [--limit N]"""
import argparse, json, os, subprocess, tempfile, time, glob

FPS, WINDOW, WIDTH = 2, 8.0, 768
TASK = ("You are the live race caller for this horse race. You get the racecard and the pictures of the last 8 seconds, in order. "
        "In three or four sentences say: which horse leads, which horses are closest behind it, and where on the track the field is. "
        "Use the horses' names. Identify a horse by the colour of its jockey's cap (helmet), using the racecard; do not go by the jacket. "
        "The saddle cloth has the same colour as the cap and carries the horse's number: use the number too if you can clearly read it. "
        "The 'SEOUL' graphic at the top left shows the race number, not a horse's number. "
        "The horses run anticlockwise. Depending on the camera, the leader can be at the left or at the right of the picture: "
        "work out which way is forward from the direction the horses face and move across the pictures. "
        "In a head-on shot the horse nearest the camera is in front. "
        "FORMAT: your first line must be exactly 'ORDER: ' followed by the cap colours you can see in RUNNING ORDER, leader first, "
        "separated by commas (running order, not left to right). Then an empty line, then your call.")
RULE_R = (" RULE: name a horse or give its position only when you clearly see its cap colour and you are sure. If you are not sure, say plainly "
          "that you cannot tell which horse it is. Do not state any distance, number or situation that you cannot see in the pictures.")
FACTS = " You also get VERIFIED FACTS. They are guaranteed correct and you may use them."
RULE_B = (" RULE: name a horse or give its current position only when you clearly see its cap colour and you are sure. Whenever you are not sure, "
          "do NOT guess a name and do NOT speak vaguely: say a verified fact instead (where the field is, the conditions, or a horse and its jockey). Do not state any distance or number that is "
          "not in the verified facts. Do not describe a situation you cannot clearly see.")
RULE_C = (" You also get a CONFIRMED ON SCREEN list from an identification system: these horses are certainly in the pictures. The list does NOT "
          "give the running order. You may name ONLY horses on this list as being in the pictures now, and give a position for one only if you "
          "clearly see it. If the list is empty, do not name any horse as being anywhere now. For everything else follow the rule above.")

def frames(clip, end):
    d = tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(max(0.0, end - WINDOW)), "-i", clip,
                    "-vf", f"fps={FPS},scale={WIDTH}:-2", "-q:v", "3", f"{d}/f%03d.jpg"], check=True)
    return sorted(glob.glob(f"{d}/*.jpg"))

def confirmed(sc, ident):
    """numbers the identifier is sure about in the two analysed frames of this scene (last frame and 2.5 s earlier)"""
    f = os.path.join(os.path.expanduser(ident), f"R{sc['race']}", "visible.json")
    if not os.path.exists(f): return None
    v = json.load(open(f)); nos = set()
    for ab in "ab":
        for d in v.get(f"{sc['scene']}_{ab}.jpg", []):
            if d.get("sure") and d.get("state", "sure") == "sure": nos.add(d["no"])
    return sorted(nos)

def messages(sc, cond, imgs, conf):
    card = "\n".join(f"No.{h['no']} {h['name']} - {h['cap']} cap" for h in sc["racecard"])
    text = f"RACECARD\n{card}\n"; system = TASK
    if cond == "R": system += RULE_R
    if cond in ("F", "B", "C"):
        system += FACTS; text += "\nVERIFIED FACTS\n" + "\n".join("- " + f for f in sc["facts"]) + "\n"
    if cond in ("B", "C"): system += RULE_B
    if cond == "C":
        system += RULE_C; by = {h["no"]: h for h in sc["racecard"]}
        text += "\nCONFIRMED ON SCREEN (order unknown)\n" + ("\n".join(f"- {by[n]['name']} ({by[n]['cap']} cap)" for n in conf if n in by) if conf else "- none") + "\n"
    text += f"\nThe pictures are the last {WINDOW:.0f} seconds, {FPS} frames per second, in order. Call the race now."
    return [{"role": "system", "content": system}, {"role": "user", "content": text, "images": imgs}]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes"); ap.add_argument("clips"); ap.add_argument("out")
    ap.add_argument("--ident", default="~/onair_work/silkid/out4"); ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--conds", default="A,R,F,B,C"); ap.add_argument("--reps", type=int, default=1); ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    scenes = json.load(open(a.scenes))[: a.limit]; conds = a.conds.split(",")
    if "C" in conds:
        miss = [s["scene"] for s in scenes if confirmed(s, a.ident) is None]
        if miss: raise SystemExit(f"identifier results not found for {len(miss)} scenes under {a.ident} (run silkid v4 first, or drop C with --conds A,R,F,B)")
    if a.dry_run:
        for c in conds: print(f"===== {c}\n" + json.dumps(messages(scenes[0], c, ["<16 frames>"], confirmed(scenes[0], a.ident) or []), indent=1, ensure_ascii=False))
        raise SystemExit
    import ollama
    done = {(r["scene"], r["cond"], r["rep"]) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    t0 = time.time(); k = 0
    with open(a.out, "a") as fh:
        for n, sc in enumerate(scenes, 1):
            todo = [(c, r) for c in conds for r in range(a.reps) if (sc["scene"], c, r) not in done]
            if not todo: continue
            hit = glob.glob(os.path.join(os.path.expanduser(a.clips), sc["clip_glob"]))
            if not hit: print("NO CLIP", sc["scene"]); continue
            imgs = frames(hit[0], sc["video_end"]); conf = confirmed(sc, a.ident) if "C" in conds else None
            for c, r in todo:
                res = ollama.chat(model=a.model, messages=messages(sc, c, imgs, conf), options={"temperature": 0.2, "num_predict": 500, "num_ctx": 32768})
                txt = res["message"]["content"].strip(); k += 1
                fh.write(json.dumps({"scene": sc["scene"], "cond": c, "rep": r, "model": a.model, "response": txt, "confirmed": conf if c == "C" else None,
                                     "n_frames": len(imgs), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n"); fh.flush()
                print(f"[{n}/{len(scenes)}] {sc['scene']} {c}: {txt[:100]!r}")
            print(f"   elapsed {time.time()-t0:.0f}s, {k} calls, {(time.time()-t0)/max(1,k):.0f}s per call")
```

### 11-7. 채점기 score_ab2.py

```python
"""Score the five-arm responses statement by statement (v2: also checks where-on-track / distance claims and counts explicit "cannot tell").
usage: python3 score_ab2.py scenes.json responses.jsonl scored.jsonl [--judge qwen3-vl:8b-instruct]"""
import argparse, json, os, re, csv, collections

JUDGE = """You label the statements of a horse-race commentary. Return JSON only.
RUNNERS: {names}
WHERE THE FIELD IS NOW: it has already passed: {passed}. It has NOT yet reached: {ahead}.
VERIFIED FACTS:
{facts}

COMMENTARY:
{text}

Split the commentary into separate statements, one claim each (split "A leads, followed by B and C" into three). For each give:
- "text": the statement
- "type": one of
    "position" = where a NAMED horse is in the race right now (leads, second, third, at the back, ahead of / behind another horse)
    "named"    = a NAMED horse is doing something now, with no checkable position
    "location" = where the field is on the track now, or how far there is to go (turns, straights, distance marks, metres to go)
    "fact"     = background: conditions, race distance, number of runners, which jockey rides which horse, cap colours
    "abstain"  = says it cannot tell / cannot identify who is where
    "vague"    = talk about the race now with no name and nothing checkable ("they are bunched up", "the pace is strong", "no clear leader")
    "other"    = anything else
- "horse": for position/named: the runner's name exactly as in RUNNERS, else null
- "rank": for position: 1 if the horse leads / is in front, 2 second, 3 third ...; "last" if at the rear; null if no rank is stated
- "ahead_of": for position: name of a runner this horse is said to be in front of, else null
- "loc_ok": for location: true if it agrees with WHERE THE FIELD IS NOW and the VERIFIED FACTS (saying the field approaches the next point not yet reached is fine); false if it places the field somewhere else, or gives a distance that is not in the VERIFIED FACTS
- "fact_ok": for fact: true if the VERIFIED FACTS support it, false otherwise
Return: {{"statements": [ ... ]}}"""

def judge(client, model, sc, text):
    p = JUDGE.format(names=", ".join(h["name"] for h in sc["racecard"]), passed="; ".join(sc["passed"]), ahead="; ".join(sc["ahead"]),
                     facts="\n".join("- " + f for f in sc["facts"]), text=text)
    for _ in range(2):
        r = client.chat(model=model, messages=[{"role": "user", "content": p}], format="json", options={"temperature": 0, "num_predict": 1500, "num_ctx": 8192})
        try:
            d = json.loads(r["message"]["content"]); st = d.get("statements") if isinstance(d, dict) else d
            if isinstance(st, list): return [s for s in st if isinstance(s, dict)]
        except Exception: pass
    return None

def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())
COL = [("sky", 10), ("light green", 6), ("green", 6), ("white", 1), ("yellow", 2), ("red", 3), ("black", 4), ("blue", 5), ("brown", 7), ("pink", 8), ("purple", 9), ("violet", 9)]
def split_order(text):
    """-> (ORDER line as list of cap numbers, rest of the text). sky blue caps (10+) are all mapped to 10."""
    m = re.match(r"\s*\**ORDER\**\s*:\s*(.*)", text or "", flags=re.I)
    if not m: return None, text
    line = m.group(1).split("\n")[0]; rest = (text or "")[m.end(1) - len(m.group(1)) + len(line):].strip(); nos = []
    for part in re.split(r"[,;>]| then ", line.lower()):
        for k, n in COL:
            if k in part: nos.append(n); break
    return nos, rest
def order_metrics(sc, nos):
    off = [min(n, 10) for n in sc["order_now"]]
    if not nos: return {"has_order": 0, "leader_ok": 0, "top3_hits": 0, "pairs_ok": None}
    hits = len(set(nos[:3]) & set(off[:3])); seen = [n for n in dict.fromkeys(nos) if n in off]
    pairs = [(a, b) for i, a in enumerate(seen) for b in seen[i+1:]]; ok = sum(off.index(a) < off.index(b) for a, b in pairs)
    return {"has_order": 1, "leader_ok": int(nos[0] == off[0]), "top3_hits": hits, "pairs_ok": round(ok / len(pairs), 2) if pairs else None}
def allowed_numbers(sc):
    return {int(x) for f in sc["facts"] for x in re.findall(r"(\d+) ?(?:metres|metre|m)\b", f)}
def distance_numbers(text):
    return {int(x) for x in re.findall(r"(\d{2,4})[ -]?(?:metres|metre|meters|meter|m)\b", (text or "").lower())}

KEYS = ["true_pos", "false_pos", "named", "loc_ok", "loc_bad", "fact_ok", "fact_bad", "abstain", "vague", "other"]
def verdict(sc, s):
    t = s.get("type"); byname = {norm(h["name"]): h["no"] for h in sc["racecard"]}; order = sc["order_now"]; n = len(order)
    bad_number = bool(distance_numbers(s.get("text")) - allowed_numbers(sc))            # a distance that is not in the verified facts
    if t == "position":
        no = byname.get(norm(s.get("horse")))
        if no is None: return "other"
        if no not in order: return "false_pos"
        true = order.index(no) + 1; checks = []; rk = s.get("rank")
        if isinstance(rk, str) and rk.strip().lower() == "last": checks.append(true >= n - 1)
        elif isinstance(rk, (int, float)) or (isinstance(rk, str) and rk.strip().isdigit()):
            rk = int(rk); checks.append(true == rk if (rk == 1 or true == 1) else abs(true - rk) <= 1)
        other = byname.get(norm(s.get("ahead_of")))
        if other is not None and other in order: checks.append(true < order.index(other) + 1)
        if not checks: return "named"
        return "true_pos" if all(checks) else "false_pos"
    if t == "named": return "named" if norm(s.get("horse")) in byname else "other"
    if t == "location": return "loc_bad" if (bad_number or s.get("loc_ok") is not True) else "loc_ok"
    if t == "fact": return "fact_bad" if (bad_number or s.get("fact_ok") is not True) else "fact_ok"
    if t == "abstain": return "abstain"
    if t == "vague": return "loc_bad" if bad_number else "vague"
    return "loc_bad" if bad_number else "other"

def summarise(rows):
    out = {}
    for c in [c for c in "ARFBCS" if any(r["cond"] == c for r in rows)]:
        rs = [r for r in rows if r["cond"] == c and r.get("verdicts") is not None]; cnt = collections.Counter(v for r in rs for v in r["verdicts"]); k = max(1, len(rs))
        hall = cnt["false_pos"] + cnt["loc_bad"] + cnt["fact_bad"]; good = cnt["true_pos"] + cnt["loc_ok"]; pos = cnt["true_pos"] + cnt["false_pos"]
        out[c] = {"calls": len(rs), "statements": sum(cnt.values()), **{x: cnt[x] for x in KEYS},
                  "HALLUCINATIONS_per_call": round(hall / k, 2), "correct_live_claims_per_call": round(good / k, 2),
                  "verified_facts_per_call": round(cnt["fact_ok"] / k, 2), "vague_per_call": round(cnt["vague"] / k, 2),
                  "position_accuracy": round(cnt["true_pos"] / max(1, pos), 2), "calls_with_any_hallucination": sum(bool({"false_pos", "loc_bad", "fact_bad"} & set(r["verdicts"])) for r in rs)}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scenes"); ap.add_argument("responses"); ap.add_argument("out")
    ap.add_argument("--judge", default="qwen3-vl:8b-instruct"); a = ap.parse_args()
    scenes = {s["scene"]: s for s in json.load(open(a.scenes))}
    done = {(r["scene"], r["cond"], r["rep"]): r for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else {}
    import ollama
    rows = []
    with open(a.out, "a") as fh:
        for r in map(json.loads, open(a.responses)):
            key = (r["scene"], r["cond"], r["rep"])
            if key in done: rows.append(done[key]); continue
            nos, body = split_order(r["response"]); r["order_line"] = nos; r["order"] = order_metrics(scenes[r["scene"]], nos)
            st = judge(ollama, a.judge, scenes[r["scene"]], body)
            r["statements"] = st; r["verdicts"] = None if st is None else [verdict(scenes[r["scene"]], s) for s in st]
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush(); rows.append(r)
            print(r["scene"], r["cond"], r["verdicts"])
    base = os.path.splitext(a.out)[0]
    with open(base + "_statements.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["scene", "cond", "verdict", "statement", "type", "horse", "rank", "ahead_of", "official_order_now(numbers)", "field_has_passed", "your_check(O/X)"])
        for r in rows:
            for s, v in zip(r.get("statements") or [], r.get("verdicts") or []):
                sc = scenes[r["scene"]]
                w.writerow([r["scene"], r["cond"], v, s.get("text"), s.get("type"), s.get("horse"), s.get("rank"), s.get("ahead_of"), " ".join(map(str, sc["order_now"])), sc["passed"][-1], ""])
    sm = summarise(rows); json.dump(sm, open(base + "_summary.json", "w"), indent=1)
    rowsk = ["calls", "statements"] + KEYS + ["HALLUCINATIONS_per_call", "correct_live_claims_per_call", "verified_facts_per_call", "vague_per_call", "position_accuracy", "calls_with_any_hallucination"]
    print("\n" + f"{'':30}" + "".join(f"{c:>8}" for c in sm))
    for k in rowsk: print(f"{k:30}" + "".join(f"{sm[c][k]:>8}" for c in sm))
    print("\nORDER line (checked by code against the official order, no judge):")
    print(f"{'':30}" + "".join(f"{c:>8}" for c in sm))
    for lab, fn in [("calls_with_ORDER_line", lambda rs: sum(r["order"]["has_order"] for r in rs)),
                    ("leader_correct", lambda rs: sum(r["order"]["leader_ok"] for r in rs)),
                    ("top3_hits_of_3 (mean)", lambda rs: round(sum(r["order"]["top3_hits"] for r in rs) / max(1, len(rs)), 2)),
                    ("pairwise_order_correct (mean)", lambda rs: (lambda v: round(sum(v) / len(v), 2) if v else "-")([r["order"]["pairs_ok"] for r in rs if r["order"]["pairs_ok"] is not None]))]:
        print(f"{lab:30}" + "".join(f"{fn([r for r in rows if r['cond'] == c and r.get('order')]):>8}" for c in sm))
    print("\nfailed to parse:", sum(r.get("verdicts") is None for r in rows), "| statements:", base + "_statements.csv")
```

### 11-8. 시간 차로 "붙어 있는 무리" 표시 (오늘 분석)

API 원자료에서 통과 지점별 순위와 누적 시간을 읽어, 선두와의 시간 차를 장면마다 계산합니다.

```python
import json, glob
def find(o):
    if isinstance(o, dict):
        if 'chulNo' in o: return [o]
        return [y for v in o.values() for y in find(v)]
    if isinstance(o, list): return [y for v in o for y in find(v)]
    return []
rows = {}
for f in glob.glob('raw/*.json'):                      # KRA API 응답 원본
    for x in find(json.load(open(f))): rows[(int(x['rcNo']), int(x['chulNo']))] = x
# 통과 지점 -> (순위 필드, 누적 시간 필드)
KEY = {'S1F': ('sjS1fOrd', 'seS1fAccTime'), '1C': ('sj_1cOrd', 'se_1cAccTime'), '2C': ('sj_2cOrd', 'se_2cAccTime'),
       '3C': ('sj_3cOrd', 'se_3cAccTime'), '4C': ('sj_4cOrd', 'se_4cAccTime'), 'G3F': ('sjG3fOrd', 'seG3fAccTime'), 'G1F': ('sjG1fOrd', 'seG1fAccTime')}
out = []
for s in json.load(open('scenes.json')):
    ko, kt = KEY[s['point'].split('/')[0]]
    hs = [(int(x[ko]), float(x[kt]), n) for (r, n), x in rows.items()
          if r == s['race'] and str(x.get(ko, '')).isdigit() and 0 < int(x[ko]) < 90 and x.get(kt) not in (None, '', 0, '0')]
    hs.sort(); t0 = hs[0][1]; gaps = [round(t - t0, 1) for _, t, _ in hs]
    lead = [n for (_, t, n) in hs if t - t0 <= 0.3 + 1e-9]                 # 선두와 0.3초 이내 = 선두 무리
    out.append(dict(scene=s['scene'], point=s['point'], field=len(hs), gaps=gaps, lead_group=lead, lead_n=len(lead),
                    top3_spread=gaps[2], second_gap=gaps[1],
                    tied_pairs=sum(1 for a, b in zip(gaps, gaps[1:]) if b - a <= 0.1 + 1e-9)))
json.dump(out, open('hard.json', 'w'), indent=0)
```

### 11-9. 상자 좌표로 순서 맞히기 (오늘 분석)

정답 번호가 붙은 상자들에 대해, 좌우 위치 / 세로 위치 / 상자 크기 각각이 공식 순위의 앞뒤를 얼마나 맞히는지 셉니다. `gold`는 프레임마다 `(상자, 정답 번호)` 목록, `order_now`는 공식 순위, `gaps`는 위 스크립트의 시간 차입니다.

```python
import itertools

def conc(vals, rk, pairs, sign):
    """값이 큰 쪽(sign=1) 또는 작은 쪽(sign=-1)을 '앞'이라고 했을 때 공식 순위와 앞뒤가 맞는 쌍의 수"""
    return sum(((vals[a] - vals[b]) * sign > 0) == (rk[a] < rk[b]) for a, b in pairs)

def order_rules(boxes_with_no, order_now, gaps):
    """boxes_with_no: [((x1,y1,x2,y2), 번호), ...]  ->  규칙별 (맞은 쌍, 전체 쌍)"""
    gp = dict(zip(order_now, gaps)); seen = {}
    for bx, n in boxes_with_no:
        if n in gp and n not in seen: seen[n] = bx
    rk = {n: order_now.index(n) for n in seen}
    # 0.1초 이내로 붙은 쌍은 기록으로도 앞뒤를 가를 수 없으므로 제외
    far = [(a, b) for a, b in itertools.combinations(seen, 2) if abs(gp[a] - gp[b]) > 0.1 + 1e-9]
    X = {n: (b[0] + b[2]) / 2 for n, b in seen.items()}      # 가로 중심
    Y = {n: b[3] for n, b in seen.items()}                   # 상자 아래 변 (화면 아래일수록 큼)
    Ht = {n: b[3] - b[1] for n, b in seen.items()}           # 상자 높이
    return {"x_best_dir": (max(conc(X, rk, far, 1), conc(X, rk, far, -1)), len(far)),   # 방향은 사후 선택 (부풀려짐)
            "y_lower_ahead": (conc(Y, rk, far, 1), len(far)),                            # 고정 규칙: 아래쪽이 앞
            "height_bigger_ahead": (conc(Ht, rk, far, 1), len(far))}

def running_order(visible_frame):
    """식별기 출력 한 프레임 -> 확정된 말만, 화면 아래쪽부터 = 앞선 순서 (연결 예정, 아직 검증 전)"""
    sure = [d for d in visible_frame if d.get("state") == "sure"]
    return [d["no"] for d in sorted(sure, key=lambda d: -d["box"][3])]
```

### 11-10. 중계 표현으로 문항 만들기 make_items.py (초기 파이프라인)

```python
"""T3 self-correction items: a true call in race-caller language with ONE corrupted final passage.
Phrases come from a professional caller's own stock (style 'caller_v1'). Only phrases that assert
running ORDER are used for the true history, because order is all the official record can verify
(no margins, no pace, no rail/wide positions)."""
import json, glob, sys, random
from caps import CAPS, confusion_level

CLIP_AFTER = 5.0
WHERE = {"S1F": "Early on,", "1C": "Around the first turn,", "2C": "Into the back straight,",
         "3C": "Approaching the far turn,", "4C": "Rounding the final bend,",
         "G8F": "Sixteen hundred to go,", "G6F": "Twelve hundred to go,", "G4F": "Eight hundred to go,",
         "G3F": "Six hundred metres to go,", "G2F": "Four hundred left,", "G1F": "Two hundred to go!"}
LEAD = ["{h} takes them along.", "{h} leads the way.", "out in front, it's {h}.",
        "{h} is dictating terms on the lead.", "bowling along in front is {h}."]
SECOND = ["Second, sitting on the leader's tail, is {h}.", "{h} is the nearest pursuer.", "Tracking the leader is {h}."]
THIRD = ["{h} sits handy in third.", "Then comes {h}, just off the pace."]
PREMATURE = ["{h} has drawn away and won't be caught!", "{h} is clear and coasting, this is all over!"]
PHANTOM = ["Here comes {h}, flying down the outside!", "{h} swoops around the field!", "{h} is making ground hand over fist!"]


def build(race):
    rng = random.Random(race["race_id"])
    name = {h["no"]: h["name"].title() for h in race["horses"]}
    st = race["states"]
    pick = [(rng.randrange(len(LEAD)), rng.randrange(len(SECOND)), rng.randrange(len(THIRD))) for _ in st]

    def call(k, first, second, third):
        a, b, c = pick[k]
        return " ".join([WHERE[st[k]["point"].split("/")[0]], LEAD[a].format(h=name[first]),
                         SECOND[b].format(h=name[second]), THIRD[c].format(h=name[third])])

    card = [{"no": h["no"], "name": name[h["no"]], "cap_and_saddlecloth": CAPS[h["no"]]} for h in race["horses"]]
    items = []
    for i in range(1, len(st) - 1):                              # history before, checkpoint after
        order, nxt = st[i]["order"], st[i + 1]["order"]
        lead, second, third = order[:3]
        history = [call(k, *st[k]["order"][:3]) for k in range(i)]
        true_last = call(i, lead, second, third)
        t0 = race["t_start_video"]
        base = {"race_id": race["race_id"], "point": st[i]["point"], "style": "caller_v1", "racecard": card,
                "conditions": {k: race[k] for k in ("distance", "weather", "going")},
                "video_file": race.get("video_file"), "video_start": 0.0,
                "inject_at": round(t0 + st[i]["t_race"], 1), "video_end": round(t0 + st[i]["t_race"] + CLIP_AFTER, 1),
                "true_leader": lead, "true_leader_name": name[lead], "true_order": order,
                "next_leader": nxt[0], "next_leader_name": name[nxt[0]],
                "next_checkpoint_in_s": round(st[i + 1]["t_race"] - st[i]["t_race"], 1),
                "target_visible": None, "number_readable": None}          # filled in by a human

        def add(cond, last, false_leader=None, **meta):
            items.append({**base, "item_id": f"{race['race_id']}-{st[i]['point']}-{cond}", "condition": cond,
                          "history": history + [last], "false_leader": false_leader,
                          "false_leader_name": name.get(false_leader), **meta})

        add("control", true_last)
        rear = order[3:]
        if rear:
            rank = {"hard": 0, "medium": 1, "easy": 2}
            sub = min(rear, key=lambda h: rank[confusion_level(lead, h)])
            add("identity_swap", call(i, sub, second, third), sub, colour_confusion=confusion_level(lead, sub))
        for k in (1, 3, 6):
            if k < len(order):
                other = order[k]
                last = call(i, other, lead, third) if k == 1 else call(i, other, second, third)
                add(f"rank_swap_{k}", last, other, rank_gap=k)
        if race["result"][0] != lead:                            # only a real error if the leader does NOT win
            add("premature_call", true_last + " " + rng.choice(PREMATURE).format(h=name[lead]), None,
                planted_horse=lead, planted_horse_name=name[lead], claim="race_decided")
        back = [h for h in order[len(order) * 2 // 3:] if nxt.index(h) >= len(nxt) // 2]   # still in rear half later
        if back:
            ghost = back[-1]
            add("phantom_move", true_last + " " + rng.choice(PHANTOM).format(h=name[ghost]), None,
                planted_horse=ghost, planted_horse_name=name[ghost], claim="phantom_move")
    return items


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    items = [it for f in sorted(glob.glob(f"{src}/*.json")) for it in build(json.load(open(f)))]
    with open(out, "w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(items), "items;", dict(Counter(i["condition"] for i in items)))
```
