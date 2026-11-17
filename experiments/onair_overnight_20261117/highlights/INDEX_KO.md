# 의미 있는 사례

모델에 넣은 입력에는 여기 표시한 번호·정답·출력이 없습니다. 성공·실패 사례는 결과를 본 뒤 설명 목적으로 선택했으며 표본 평균을 대신하지 않습니다.

## 1. 노랑 초안: 넓은 영역의 빨강 판정이 수동 영역에서 노랑으로 바뀜

근거 범위: `assistant_draft`.

![사례 1](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/highlights/01_yellow.jpg)

## 2. 초록 초안: 넓은 영역의 파랑 판정이 수동 영역에서 초록으로 바뀜

근거 범위: `assistant_draft`.

![사례 2](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/highlights/02_green.jpg)

## 3. 사람 확인 10번: 마스킹 보정 후에도 파랑 5번으로 혼동

근거 범위: `human_confirmed_colour_only`.

![사례 3](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/highlights/03_human_sky.jpg)

## 4. 사람 확인 11번: 마스킹 보정에서도 수동 영역은 1번으로 악화

근거 범위: `human_confirmed_colour_only`.

![사례 4](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/highlights/04_human_stripes.jpg)

## 5. 초안도 미확인: 모델의 번호를 정답으로 삼지 않음

근거 범위: `unresolved_draft`.

![사례 5](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/highlights/05_unresolved.jpg)

## 6. 동일 화면 안에서 사람 트랙 ID가 바뀌며, 박스 비율 뷰 추정은 측면을 놓침

근거 범위: `assistant geometry drafts, not MOT ground truth`.

![사례 6](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/highlights/06_tracking.jpg)

## 7. 사전 지정 첫 표본: 축소 후 모자 픽셀과 실제 출력 비교

근거 범위: `assistant draft`.

![사례 7](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/highlights/07_resolution.jpg)

