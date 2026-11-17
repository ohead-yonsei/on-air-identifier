# 핵심 근거 모음

모든 그림은 실제 영상의 자르기·확대·좌표 표시로 만들었다. 이미지 생성·장면 복원·가림 제거를 사용하지 않았다. `draft`는 새 사용자 검수 전의 육안 초안을 뜻한다.

## 1. 한 카메라 화면 안에서도 뷰가 변한다

32.733~43.067초 사이에는 컷 없이 왼쪽 측면→정면→오른쪽 측면으로 바뀐다. 녹색 화살표는 수동으로 정한 화면 축이며 보정된 주로 방향은 아니다.

![같은 화면의 뷰 변화](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/highlights/01_same_shot_view_changes.jpg)

## 2. 같은 모자도 영역을 어떻게 자르는지에 따라 답이 바뀐다

같은 8프레임·같은 모델·같은 분류 프롬프트에서 자동 기수 영역은 보라색 3/8, 모자만 수동으로 자른 영역은 보라색 8/8이었다. 아래는 대표 4개다. 모두 같은 보라색 말이므로 다중 클래스 정확도로 일반화하지 않는다.

![영역 차이](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/highlights/02_roi_vs_tight_cap.jpg)

## 3. 가까운 모자가 다른 말의 모자일 수 있다

37·48초에는 1번 대응 코끝이 9번 모자와 더 가깝다. 방향 제한과 거리 차 보류 기준도 이 오류를 막지 못했다. 이번 수동 좌표에서는 일대일 배정이 해결했다.

![연결 오류](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/highlights/03_new_association_failures.jpg)

## 4. 큰 모델도 다른 말을 선두로 지정했다

34초의 실제 정규화 좌표를 원본에 표시했다. 녹색은 선두 초안 9번의 모자, 빨강·노랑은 두 모델의 예측 모자 박스다. 색만 읽는 문제와 선두를 찾는 문제를 분리해야 한다.

![전체 화면 위치 추정](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/highlights/04_full_frame_localization.jpg)

## 5. 이동 방향이 곧 순위 좌표는 아니다

좌우 반전 검사를 통과한 자동 움직임 벡터도 화면 좌표를 단순 정렬하면 37초에 1번 대응 코끝을 선택했다. 선두 초안은 9번이다. 움직임 자체의 정답과 순위 비교용 좌표는 각각 검증해야 한다.

![방향과 순위의 차이](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/highlights/05_motion_is_not_rank_calibration.jpg)

## 6. 화면 색이 바뀐 것과 카메라 컷은 다르다

32.733초 후보는 실제 카메라 전환이다. 44.933초 후보는 색·노출이 달라졌지만 같은 화면이 이어진다. 후보 자동 검출 뒤 검토가 필요했다.

![컷 후보 검토](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/highlights/06_cut_vs_colour_change.jpg)

## 7. 영상 전체를 바로 입력하면 안 되는 이유

초반 동시 카메라 화면과 말 번호를 알려주는 방송 라벨이다. 이 그림은 조사용이며 핵심 8프레임 비교의 모델 입력에는 쓰지 않았다. 고정된 모서리·하단 마스크만으로 모든 방송 그래픽이 없어지는 것은 아니다.

![입력에서 제외한 구간](/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/highlights/07_input_hazards.jpg)
