"""Generate the seven-chapter Korean report from saved results, without inference."""
import collections,datetime,json,statistics
from pathlib import Path
from common import ROOT,CFG,readlines,dump,now
MAC=Path('/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117')
def L(p,title=None):return f'[{title or p}]({MAC/p})'
def table(h,rows):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def main():
 s=json.loads((ROOT/'results/C_SUMMARY.json').read_text());groups=s['groups']['by_split_method_classifier'];g={(r['split'],r['method'],r['classifier']):r for r in groups};ctrl=json.loads((ROOT/'results/human_mask_summary.json').read_text()) if (ROOT/'results/human_mask_summary.json').exists() else None;methods=CFG['methods'];vlm=readlines(ROOT/'raw/vlm.jsonl');vision=readlines(ROOT/'raw/vision_ledger.jsonl');e=json.loads((ROOT/'results/tracking_summary.json').read_text());extras=(ROOT/'results/resolution_extra.json').exists() and (ROOT/'results/tracking_extra_summary.json').exists();end=(ROOT/'evidence/final_verification.json').exists();dt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
 def triple(row):return f"{row['match']} / {row['mismatch']} / {row['abstain']}"
 txt=f'''# ON-AIR 야간 연구 2일차 결과

작성 {dt}. **상태: {'A~F 및 사전 지정 여분 실험 처리 완료. D는 필요한 정답 파일 부재로 보류. 예정 큐가 소진되어 종료하며 별도 야간 작업은 남아 있지 않다.' if extras else 'A~F 본 실험·보고 자료 준비 완료. D는 자료 부재로 보류. 다음 순서는 사전 지정 여분 실험이다.'}** 실제 실행은 11월 17일 01:50 KST부터 진행했다. 11월 18일 아침까지 계속 실행한 결과라고 주장하지 않는다.

## 1. 세 줄 결론

1. **모자를 좁게 자르는 효과는 색·경주에 따라 달랐다.** 평가 6경주에서 Qwen의 초안 일치는 상단 35% 17/34 → 수동 영역 19/34였다. 6개가 좋아졌지만 4개는 나빠졌다. 일반적인 개선을 입증한 결과는 아니다.
2. **자동으로 좁히는 두 방법은 이번 설정에서 이득을 보이지 않았다.** 자세 기반과 군집 기반은 각각 12/34 초안 일치였다. 사람 확인 6개는 입력 마스킹 보정 후 넓은 영역 5/6 → 수동 영역 4/6이었다. 원 실험의 6/6→4/6은 보정 전 값으로 따로 보존했다.
3. **지금은 정답 검수와 모자 분류의 실패 원인 분리가 우선이다.** 추적의 배경 보정도 ID 안정성을 개선하지 못했고, 공식 시각 대응 자료가 없어 순서 정확도는 계산하지 않았다.

## 2. 어제와 달라진 점

어제 보라색 9번 한 말의 8장면에서 보았던 “좁은 모자 영역이 더 잘 맞는다”는 관찰은 그대로 보존한다. 그러나 여러 색·다른 경주로 넓히자 **항상 좋아진다는 해석은 성립하지 않았다.** 특히 사람 확인 10번은 수동 영역에서 파랑 5번, 11번은 흰색 1번이 됐다. U는 사람이 제공한 모자 위치의 비교 기준일 뿐, 분류 정확도의 보장된 수학적 상한선도 자동 시스템 성능도 아니다.

카메라 컷 사이에도 뷰가 바뀐다는 어제의 관찰은 유지된다. 32.8~43.0초 같은 화면 구간의 34→37→40→42초를 다시 남겼다. 새 코드는 매 프레임 갱신하지만, **갱신 주기와 뷰를 올바르게 판단하는 능력은 별개**다. 말 박스의 가로세로 비율만 쓴 이번 뷰 추정은 측면 장면을 구분하지 못했다.

## 3. 실제 수행한 범위

서버 접속에 성공했고 **NVIDIA GeForce RTX 3090, VRAM 24,576 MiB**를 실제 사용했다. 시작 시 381 MiB, GPU 사용률 0%, 다른 계산 프로세스가 없었다. 기존 연구 잠금 파일은 있었지만 잠금은 잡혀 있지 않았다. 다른 프로세스나 서비스 설정을 변경하지 않았다.

Python 3.12.3, PyTorch 2.12.1+cu132, torchvision 0.27.1+cu132, CUDA 빌드 13.2, OpenCV 5.0.0, Ollama 0.32.15. 모델은 **qwen3.5:9b / 9.7B / GGUF Q4_K_M**, digest `{CFG['model_digest']}`다. 버전·GPU 원문은 {L('evidence/environment_start.json','환경 기록')}에 있다.

9월 6일 서울 R1~R10은 **10개 모두 재검증 통과**, 각각 1920×1080·60fps·H.264·음성 없음이다. 이미 있던 약 896MB 영상은 다시 받거나 수정하지 않았다. 길이는 R1 96.35, R2 107.65, R3 153.32, R4 155.32, R5 120.93, R6 123.87, R7 124.77, R8 134.72, R9 162.58, R10 150.42초다. 경주별 제목·경로·SHA256과 검증 결과는 {L('evidence/video_sources.json')} 및 {L('evidence/video_reverification.json')}에 있다.

각 경주 2시점×3기수 = **60개 모자 관측**, 이 중 56개 번호 초안·4개 미확인이다. 경주 내 초안 번호를 묶으면 55개의 경주–번호 조합이다. 실제 개체 55마리를 독립 검증했다는 뜻은 아니다. 10월 영상의 사람 확인 6개를 더해 **11경주·21프레임·66개 관측**을 비교했다. 1~11번 색/무늬가 전체 초안에 포함되지만, 12번 이상 무늬나 모든 출전마를 포괄하지 않는다.

개발 R3/R5/R6/R7, 평가 R1/R2/R4/R8/R9/R10을 경주 단위로 미리 분리했다. 개발 24개 중 번호 초안 22개, 평가 36개 중 34개다. 평가 전에 조건·코드·라벨 해시를 고정했고 개발 결과를 보고 변경하지 않았다. 다만 **기존 v4 색 규칙은 과거 R1/R2/R4로 조정된 적이 있으므로, v4에 역사적으로 처음 보는 경주라고 부를 수 없다.**

현재 사람 정답은 10월 38.5초의 **9·1·2·10·5·11번 모자 색 6개만**이다. 그 확인은 코끝, 순위, 수동 모자 박스의 정확성까지 확정하지 않는다. Jo의 16장과 scenes/visible/owners 파일은 지정 위치에 없었다. 어제의 22개 검수 확인 칸도 비어 있어 모두 초안으로 유지했다.

실제 VLM 요청은 **{len(vlm)}/400회**, 실패 {sum(r['failure'] for r in vlm)}회, 재시도 {sum(r['attempt']>1 for r in vlm)}회다. 본 실험은 330개 영역 조건 중 자세 추정 실패 15개를 호출 없이 보류하여 **315회** 호출했다. 일반 검출·자세 추정은 **{len(vision)}/3,000프레임 추론**이다(공통 검출 21 + 자세 21 + 시간 추적 검출 271 + 사람 프레임 마스킹 보정 2). 공개 자세 가중치 다운로드는 237,034,793바이트로 1GB 이내, 유료 API·Jev·Gemini 호출은 0회, 학습은 0회다.

실행 전 계획의 315회 본 실험과 24회 여분 외에, 최종 점검에서 발견한 사람 프레임의 그래픽 제거 누락을 바로잡는 **사후 입력 보정**으로 10회 추가 호출했다. 18개는 동일한 PNG 바이트·프로토콜을 검증해 기존 응답을 재사용했고, 2개는 영역 보류였다. 원 결과·라벨·임계값을 덮어쓰지 않았다.

모든 추출은 소스 PTS의 목표 시각 이하 프레임만 사용했다. 9월 입력은 원본 폭 1920 좌표에서 상단 좌측·상단 우측·하단 그래픽을 가렸고, 10월 시간 추적 입력은 폭1280 전용 마스크를 사용했다. 단, 원 사람 확인용 전체 프레임은 그래픽이 남은 채 앞단 검출·자세 모델에 들어갔음을 최종 육안 검수에서 발견했다. VLM 모자 조각에는 그래픽이 없었다. 이 예외를 숨기지 않고 전체 프레임부터 마스킹한 별도 점검 결과를 아래에 제시했다. 실제 모델 요청은 **한 번에 모자 crop 한 장**, 224×224 회색 여백·종횡비 유지다. 음성·출전표·공식 순위·초안 번호는 요청에 넣지 않았다. 전체 프롬프트와 입력/요청 해시는 {L('raw/request_ledger.jsonl')}, 원시 응답은 {L('raw/vlm.jsonl')}에 보존했다.

## 4. 단계별 결과 표

### C. 모자 영역 비교 — 주 결과

**평가 6경주, 번호 초안 34개. 아래는 정확도가 아니라 ‘초안 일치 / 불일치 / 보류’다.** 미확인 2개는 이 분모에서 제외했다. R0_27은 상단27%+좌우8% 제외, R0_35는 상단35% 전체폭, R1은 자세 점, R2는 Lab 군집, U는 좁은 수동 박스다. 공통 기수 박스는 동일하다.

'''
 txt+=table(['영역','Qwen 일치/불일치/보류','v4 색 부품 일치/불일치/보류','HSV high_evidence 불일치'],[(m,triple(g['evaluation',m,'vlm']),triple(g['evaluation',m,'hsv']),g['evaluation',m,'hsv']['high_evidence_mismatch']) for m in methods])
 txt+='\nHSV는 문서 11-1의 quant/hist/cap_score를 복사한 **색 규칙 부품**이다. 경주복·전역 번호 배정까지 포함하는 전체 v4의 sure 성능이 아니다. high_evidence는 사전에 정한 점수·차이·색 근거·크기 기준으로, 보정된 확률이 아니다. Qwen 프로토콜은 확신도를 출력하지 않으므로 “확신 오답”은 측정 불가이며 위 불일치는 보류하지 않고 낸 번호다. 제공 정보에 없다는 이유로 거짓 처리한 것이 아니라 번호 초안과의 일치만 비교했다.\n\n'
 txt+=table(['영역','평균 모자 IoU','모자 중심 포함','영역 생성','Qwen ms 중앙값','HSV ms 중앙값'],[(m,f"{g['evaluation',m,'vlm']['cap_iou_mean_all_including_no_region']:.3f}",f"{g['evaluation',m,'vlm']['cap_center_included']}/35",f"{g['evaluation',m,'vlm']['region_available']}/36",f"{g['evaluation',m,'vlm']['median_region_plus_classification_ms_available']:.2f}",f"{g['evaluation',m,'hsv']['median_region_plus_classification_ms_available']:.3f}") for m in methods])
 detector=statistics.median(r['forward_ms'] for r in readlines(ROOT/'raw/detector.jsonl'));pose=statistics.median(r['forward_ms'] for r in readlines(ROOT/'raw/pose.jsonl'))
 txt+=f'\n박스가 특히 불확실한 1개를 영역 품질에서 제외해 분모는 35다. U의 IoU=1은 자기 박스와 비교하므로 당연한 값이다. R1은 중심 포함률과 영역 제공률이 낮았다. 시간은 영역 계산+분류의 가용 입력 중앙값이며, 공통 검출 약 {detector:.1f}ms/프레임과 R1 자세 추정 약 {pose:.1f}ms/프레임은 별도다. 영상 읽기·전체 파이프라인 지연이 아니다. 모델을 새로 올린 호출의 약 5초 지연은 원시 로그에 포함되지만 중앙값에 드러나지 않는다.\n\n'
 txt+=table(['영역','모자 박스 바깥 비율','얼굴 겹침 대리값(n)','옷 겹침 대리값(n)'],[(m,f"{g['evaluation',m,'vlm']['outside_cap_box_fraction_mean']:.3f}",f"{g['evaluation',m,'vlm']['face_proxy_mean']:.3f} ({g['evaluation',m,'vlm']['face_proxy_n']})",f"{g['evaluation',m,'vlm']['torso_proxy_mean']:.3f} ({g['evaluation',m,'vlm']['torso_proxy_n']})") for m in methods])
 txt+='\n혼입 값은 assistant가 만든 **초안 사각형 간 중첩의 대리값**이다. 실제 피부·옷 픽셀의 분할 비율이 아니다. 얼굴/옷 박스가 없거나 영역 생성이 실패한 입력은 해당 평균에서 빠진다.\n\n'
 txt+='**사람 확인 6개: 이 표만 색 판정 정확도를 계산할 수 있다.** 동일 영상·동일 프레임의 재점검이며 미지 경주 일반화 평가는 아니다.\n\n'
 txt+=table(['영역','Qwen 정답/오답/보류','Qwen 정답률','HSV 정답/오답/보류'],[(m,triple(g['human_check',m,'vlm']),f"{g['human_check',m,'vlm']['match']}/6",triple(g['human_check',m,'hsv'])) for m in methods])
 if ctrl:
  txt+='\n위 표는 **보정 전 원 실험**이다. 전체 프레임에서 그래픽을 제거한 사후 입력 보정은 아래와 같다. 분류기의 입력 픽셀이 동일한 18개는 응답을 재사용했다. 새로운 사람 표본이나 독립 반복 실험으로 세지 않는다.\n\n'
  txt+=table(['영역','보정 후 Qwen 정답/오답/보류','보정 후 HSV 정답/오답/보류'],[(m,' / '.join(str(ctrl['methods'][m]['vlm'][k]) for k in ['correct','wrong','abstain']),' / '.join(str(ctrl['methods'][m]['hsv'][k]) for k in ['correct','wrong','abstain'])) for m in methods])
  txt+='\n마스킹 후 10번의 공통 기수 박스가 조금 달라지며 R0_35 crop 높이가32→31픽셀이 됐고, Qwen 출력도 sky_blue(10)→blue(5)로 바뀌었다. 9번의 자세 기반 영역은 purple→unknown이었다. **그래픽을 읽었기 때문이라고 단정할 수 없으며, 앞단 박스·crop 픽셀 민감도가 드러난 것이다.** 수동 U의 6개 입력은 바이트가 같아 4/6 결과가 유지됐다. 자세 모델을 다시 실행한 프레임까지 동일한 마스크를 적용한 보정 점검 결과는 results/human_mask_control.json에 있다.\n\n'
 txt+='\n평가 경주별 R0_35→U Qwen 초안 일치: R1 3→4/6, R2 2→2/6, R4 3→5/6, R8 3→4/5, R9 3→1/5, R10 3→3/6. 일치율 차이를 유의한 개선으로 해석하지 않는다. 같은 경주·인접 시점은 독립 표본이 아니다.\n\n'
 for key,col,title in [('by_split_method_classifier_view','view','뷰별'),('by_split_method_classifier_size_group','size_group','기수 크기별')]:
  sub=[r for r in s['groups'][key] if r['split']=='evaluation' and r['classifier']=='vlm'];cats=sorted(set(r[col] for r in sub));txt+=f'**{title} Qwen 초안 일치 수 / 번호 초안 수**\n\n';txt+=table(['구분']+methods,[[c]+[f"{next(r for r in sub if r[col]==c and r['method']==m)['match']}/{next(r for r in sub if r[col]==c and r['method']==m)['labelled_observations']}" for m in methods] for c in cats]);txt+='\n'
 txt+='뷰도 육안 초안이다. 크기는 기수 박스 높이를 폭1280 기준으로 환산해 작음<60, 중간<120, 큼으로 나눴다. 전체 색별 혼동표, 개발 결과, 관측별 기록은 '+L('results/colour_confusion.csv')+', '+L('results/condition_summary.csv')+', '+L('results/condition_comparison.csv')+'에 있다. 흰↔하늘, 보라↔파랑, 검정↔갈색을 포함하며 미확인 색을 오답으로 세지 않았다.\n\n'
 txt+='R1은 COCO 사전학습 Keypoint R-CNN의 코·눈·귀 점을 사용했다. 모델 정보는 [torchvision 공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html), 실제 사용 enum·가중치 SHA256·설치 버전은 '+L('evidence/pose_weights.json')+'에 있다. 자세 모델은 모자 전용 검출기로 학습된 것이 아니다.\n\n'
 txt+='### D. 식별 번호 + 세로 순서 규칙\n\n**보류.** 저장된 scenes.json/order_now, 통과시각 차이, 영상 시점과 공식 라벨의 대응 자료가 없어 0.1초 쌍 및 0.1/0.3초 선두권 점수를 계산하지 않았다. 결승 순위로 임의 시점의 정답을 만들거나 API/키를 찾지 않았다. C에서 높은 점수를 보인 수동 영역을 전체 v4에 붙여 순서 성능이 좋아졌다고 주장하지 않는다.\n\n'
 txt+='### E. 말–기수 대응과 시간 추적\n\n10월 영상 31.0~58.0초, 0.1초 간격 271프레임. 일반 검출 박스의 Hungarian 일대일 대응에 미배정을 허용했고, 과거→현재의 인과적 추적만 했다. 말–기수 배정은 2,608건, 말 미배정 229건, 사람 미배정 659건이다. 자동 코끝 검출기나 컷을 넘는 마번 재식별기를 학습한 실험은 아니다.\n\n'
 txt+=table(['추적 조건','같은 컷의 초안 비교쌍','사람 ID 변경','전체 새 트랙 수'],[(k,v['sparse_same_shot_transitions'],v['sparse_person_id_changes'],v['new_tracks_over_sequence']) for k,v in e['summary'].items()])
 txt+='\n새 트랙에는 말·사람, 화면 진입·미검출·컷 리셋이 섞인다. 위 ID 변경은 드문 8시점의 같은 초안 번호를 비교한 것으로 **공식 MOT ID switch 정확도가 아니다.** 컷을 넘는 비교 8개는 제외했다. 배경 보정은 이번 설정에서 더 불안정했으며 이 결과를 보고 임계값을 바꿔 원 결과를 대체하지 않았다.\n\n'
 txt+='어제 22개 코끝–모자 초안과의 배정 일치는 **17/22**다. 코끝·모자점을 포함하는 가장 작은 검출 박스를 고르는 평가 대리 규칙도 사용했다. 31초 1번과 9번 코끝이 같은 말 검출 박스에 들어가는 사례처럼, 불일치를 전부 배정기의 오답으로 확정할 수는 없다.\n\n'
 txt+='컷 후보는 시작31.0초, 전환32.8·43.1·51.7초다. 첫 프레임은 초기화이지 실제 컷 검출 성과가 아니다. 어제 육안 구간 32.733~43.067초와 0.1초 샘플 내에서 대응하나 별도 컷 정답이 없어 정확도는 계산하지 않았다. 말 박스 비율 뷰 규칙(≤1.1 정면/후면 유사, ≥1.6 측면 유사)은 PLAN에 상세 규칙이 없고 E 구현 중 추가한 **탐색용 대리 지표**다. 사전 지정 뷰 분류기의 성능으로 제시하지 않는다. 매 프레임 이 뷰 추정은 front_or_rear_like 260, oblique_like 11, side_like 0이었다. **측면→정면→반대 측면 변화 구간을 찾는 목적에는 실패했다.** 배경 보정 잔여 이동은 266프레임에서 산출됐지만 이동 방향을 순위 축이나 정면/후면 정답으로 사용하지 않는다.\n\n'
 if extras:
  x=json.loads((ROOT/'results/resolution_extra.json').read_text());base=[r for p in (ROOT/'results').glob('regions_*.json') for r in json.loads(p.read_text()) if r['method']=='U_manual'];ids={r['sample_id'] for r in x};rr=[];totals={}
  for width in [768,1280,1920]:
   a=[r for r in x if r['width']==width] if width!=1920 else [r for r in base if r['sample_id'] in ids];known=[r for r in a if r['reference_number']];counts={}
   for cls in ['vlm','hsv']:
    counts[cls]=[sum(r[cls]['number']==r['reference_number'] for r in known),sum(r[cls]['number'] not in [0,r['reference_number']] for r in known),sum(r[cls]['number']==0 for r in known)]
   totals[str(width)]={**counts,'labelled':len(known),'unresolved':len(a)-len(known)};rr.append([width,' / '.join(map(str,counts['vlm'])),' / '.join(map(str,counts['hsv']))])
  j=json.loads((ROOT/'results/R2_jitter_extra.json').read_text());te=json.loads((ROOT/'results/tracking_extra_summary.json').read_text());dump(ROOT/'results/EXTRAS_SUMMARY.json',{'resolution':totals,'jitter_rows':len(j),'changed_predictions':sum(r['changed_number'] for r in j),'median_iou':statistics.median(r['iou_with_baseline'] for r in j),'tracking':te['summary'],'selection_rule':'predeclared earliest selected frame, first two IDs in each evaluation race'})
  txt+='### 사전 지정 여분 실험\n\nF의 첫 검토 묶음을 저장한 뒤에만 실행했다. 평가 경주마다 가장 이른 선택 시점의 첫 두 모자, 총 12개를 결과와 무관하게 골라 원본 전체 화면을 축소한 후 같은 수동 박스를 비례 적용했다. 추가 VLM 24회, 1920 결과는 재사용했다. 조건을 바꿔 재학습하지 않았다.\n\n';txt+=table(['원본 축소 폭','Qwen 초안 일치/불일치/보류','HSV 초안 일치/불일치/보류'],rr)
  txt+=f"\n개발 24개 기수 박스를 상하좌우 2px씩 움직인 96조건에서 R2의 HSV 번호가 **{sum(r['changed_number'] for r in j)}/96** 바뀌었다. 영역 IoU 중앙값은 {statistics.median(r['iou_with_baseline'] for r in j):.3f}였다. 이 96개는 새 말 96마리가 아니라 같은 입력의 민감도 점검이다.\n\n"
  txt+=table(['배경 보정 추적 dummy','같은 컷 초안 ID 변경 / 비교쌍','새 트랙 수'],[(k,f"{v['sparse_person_id_changes']}/{v['sparse_same_shot_transitions']}",v['new_tracks_over_sequence']) for k,v in te['summary'].items() if k.startswith('affine')]);txt+='\n기본 추적 결과와 별도로 보존했다. 이 민감도 점검으로 사람 정답 추적 정확도나 최적 임계값을 확정하지 않는다.\n\n'
 txt+='## 5. 사례 이미지 '+('7' if extras else '6')+'개\n\n성공·실패 사례는 설명을 위해 결과를 본 뒤 선택했다. 사진 위 출력은 저장된 JSON의 color_class 그대로이며 번호는 고정 색 대응으로 변환했다. 대표 사례만 보고 전체 성능을 평가하지 않는다.\n\n'
 for i,m in enumerate(json.loads((ROOT/'highlights/manifest.json').read_text()),1):txt+=f"**{i}. {m['caption']}**\n\n![사례 {i}]({MAC/m['file']})\n\n"
 txt+='## 6. 10분 검수 목록\n\n'+L('review/USER_REVIEW_KO.md','여기서 사진을 보며 검수')+'를 열면 질문 하나에 이미지 하나가 나온다. 평가 6경주를 먼저, 개발 4경주를 이어서 확인하고, 어제의 선두 9번 여부와 코끝–모자 연결을 마지막에 본다. 경주별 40초×10 + 선두60초 + 연결120초 = 약9분40초다.\n\n번호 초안60개, 어제 연결22개, 선두8시점은 각각 CSV로 나누었고 **확인 칸은 모두 비어 있다.** 번호 확인을 박스 좌표·코끝·순위까지 승인한 것으로 확대하지 않는다. 애매한 항목은 미확인으로 남기면 된다.\n\n'
 txt+='## 7. 실패·한계·다음 우선순위\n\n- **측정·자료 병목:** 새 경주 라벨이 초안이며 Jo의 정답과 시각 대응 자료가 없다. 이를 검수하기 전 정확도·순서 성능을 주장할 수 없다. 기수 검출에 성공하고 볼 수 있는 3명씩을 골랐으므로 전체 출전마의 검출 재현율도 측정하지 않았다.\n- **영역·분류 병목:** R1 머리점과 R2 군집이 얼굴·옷·배경을 고르거나 모자를 놓쳤다. 수동 영역에서도 색/줄무늬 혼동이 남아 영역만 고쳐서는 충분하지 않다. 작은 모자 픽셀을224로 키우는 것은 새 세부정보를 만드는 일이 아니다.\n- **뷰·추적 병목:** 박스 비율 뷰 규칙은 목적에 맞지 않았고 배경 보정이 트랙 안정성을 악화시켰다. 배경을 맞추는 변환이 움직이는 말의 속도까지 예측하지는 못한다. 다만 이번 결과만으로 악화 원인이 한 가지라고 확정할 수는 없다.\n- **실행상 사건:** 초기 서버→맥 동기화 뒤 맥 전체 폴더를 역방향으로 복사하면서 일부 진행 상태/로그가 오래된 사본으로 덮였다. 분류 전 사건이며 원본 영상·추출 프레임·입력 명세는 유지됐다. 조사 완료를 129개 survey 기록으로 재확인하고 사건을 evidence/execution_events.jsonl에 남겼다. 이후 서버로는 지정 코드 파일만 보내고 결과는 서버→맥으로 복사했다. 원래 진행 로그 전체가 온전히 남았다고 주장하지 않는다.\n- **입력 보정 사건:** 최종 점검에서 사람 확인용 원본1장의 방송 그래픽을 발견했다. 기존 결과는 보존하고 그래픽 제거 후 검출·자세2회, 색 분류10회를 별도로 실행했다. 9월 평가 결과는 바꾸지 않았다. 변경 내역은 evidence/human_mask_control_plan.json 및 execution_events.jsonl에 있다.\n- **모델 실패와 알고리즘 실패 구분:** HTTP/JSON 실패는 0회다. R1 영역 보류15개, 색 불일치, 잘못된 뷰 출력은 모델 호출 실패가 아니라 실제 행동 결과다. 원시 출력과 미확인 라벨을 삭제하지 않았다.\n- **후속 한 가지:** 우선 이 검수 세트의 모자 번호와 박스를 확정한 뒤, 경주 단위 분할을 유지해 작은 모자 전용 검출·분류기를 학습하고 현재 R0_35와 같은 입력·미확인 기준에서 비교한다. 자동 영역 품질과 번호 오답/보류를 함께 측정하고, 새로운 경주를 최종 평가에 남긴다. 순서·해설 확장은 그다음이다.\n\n'
 txt+='원본과 이전 실험은 보존했다. '+L('README.md','재실행/검증 방법')+', '+L('highlights/INDEX_KO.md','근거 이미지 모음')+', '+L('SHA256SUMS','파일 해시')+'를 함께 제공한다. 보고서의 링크 이미지를 노션에 옮길 때는 ZIP의 이미지를 같이 업로드해야 하며, 맥 절대경로만 붙여넣으면 다른 기기에서 보이지 않는다.\n'
 (ROOT/'MORNING_REPORT_KO.md').write_text(txt)
 print('report written; extras=',extras,'requests=',len(vlm))
if __name__=='__main__':main()
