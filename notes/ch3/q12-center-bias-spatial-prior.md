# q12 — 모델은 사진 가운데를 "먼저" 보나? 중앙 편향은 어떤 형태로 존재하나?

관련: [q09 — 지름길 학습](q09-shortcut-learning.md), [q10 — Grad-CAM 원리](q10-gradcam-mechanism.md)
· [`scripts/position_sensitivity.py`](../../scripts/position_sensitivity.py)
· [notes.md 곁가지 진단](notes.md#곁가지-진단-위치-민감도-fivecrop--tta--2026-07-15)

## 무슨 질문이었나

지름길 학습을 배우다 나온 의문: "사진이라면 가운데에 제일 중요한 정보가 있을
텐데, 모델은 가운데부터 보나? 주변부터 보나?" — 그리고 이어진 질문: "사람이
찍은 사진이면 이 편향이 '순서'가 아니라 **가중치** 형태로 존재하지 않나?
테슬라처럼 기계가 찍은 데이터면 다르지 않나?"

## 1. "보는 순서"는 없다 — CNN은 전체를 동시에 본다

- 사람 눈은 중심와(fovea)로 한 점을 찍고 시선을 옮기며(saccade) **순차적으로**
  본다. CNN(convolutional neural network)은 다르다: 합성곱(convolution)은 같은
  필터를 **모든 위치에 동시에** 적용하는 연산이라 "여기부터"라는 시작점이 없다.
- 우리 ResNet18의 `layer4`가 내놓는 7×7 특징 지도 49칸도 한 번의 순전파에서
  전부 동시에 계산된다. Grad-CAM의 "여기를 봤다"는 **시선 순서가 아니라
  기여도**다 ([q10](q10-gradcam-mechanism.md)).
- 마지막의 전역 평균 풀링(global average pooling)은 49칸을 **똑같은 가중치로**
  평균낸다 — 풀링 단계에서 중앙 우대는 없다.

진짜 "먼저"는 공간이 아니라 **학습 순서**에 있다: 경사하강법은 손실을 가장
빨리 줄이는 쉬운 특징부터 배운다 — **simplicity bias** (Shah et al., *"The
Pitfalls of Simplicity Bias in Neural Networks"*, NeurIPS 2020). 칠판 글씨(크고
일관됨)가 "쓰는 팔 동작"(작고 다양함)을 학습 경쟁에서 이기는 이유가 이것이고,
지름길([q09](q09-shortcut-learning.md))의 정체다. 위치가 아니라 **쉬움**이 가른다.

## 2. 그러나 "가중치 형태의 중앙 편향"은 실재한다 — 관련 논문

질문의 직관이 맞았다. 두 층위로 나뉜다:

**데이터 층위 — photographer's bias (구도 편향)**
- Torralba & Efros, *"Unbiased Look at Dataset Bias"* (CVPR 2011): 유명
  데이터셋들은 각자 고유한 "찍는 스타일"(피사체를 가운데·크게)이 있어서,
  어느 데이터셋 출신인지 모델이 맞힐 수 있을 정도.
- Tatler, *"The central fixation bias in scene viewing"* (2007): 사람은 사진을
  볼 때 내용과 무관하게 중앙을 응시하는 편향이 있다. 시선 예측(saliency)
  분야에선 "중앙에 가우시안 깔기(center prior)"가 오래 강력한 베이스라인이었다.

**모델 층위 — CNN도 위치를 인코딩할 수 있다**
- Islam et al., *"How Much Position Information Do CNNs Encode?"* (ICLR 2020):
  **zero-padding**(가장자리를 0으로 채우는 관행)이 "여기가 가장자리"라는 신호를
  줘서, CNN 특징에 절대 위치 정보가 스며든다.
- Kayhan & van Gemert, *"On Translation Invariance in CNNs"* (CVPR 2020):
  이론상 위치 불변이라던 합성곱이 실제로는 "무엇이 어디 있는지"로 분류 가능.
- Liu et al., *"An Intriguing Failing of CNNs and the CoordConv Solution"*
  (NeurIPS 2018): 반대로 좌표 (x, y)를 입력 채널로 명시해 주는 CoordConv —
  위치가 의미 있는 과제에서는 위치 가중치를 일부러 준다.

## 3. 우리 실측 — 중앙 prior의 크기는 +7%p (2026-07-15)

[`position_sensitivity.py`](../../scripts/position_sensitivity.py)로 test set을
FiveCrop(네 모서리+중앙)해서 위치별 단독 정확도를 쟀다:

| | 모서리 평균 | 중앙 | 격차 | 5-crop 평균(TTA) |
|---|---|---|---|---|
| exp02 | 68.57% | 75.65% | +7.08%p | 77.02% |
| exp04 | 68.66% | 75.80% | **+7.14%p** | **77.77%** |

- **중앙 crop이 모서리보다 +7%p** — Stanford 40(웹에서 모은, 사람이 찍은 사진)의
  photographer's bias가 이 크기로 실재한다.
- 증강을 세게 한 exp04도 격차가 exp02와 같다(+7.1%p) — 이 격차는 모델이 아니라
  **데이터의 구도**에서 온다는 방증.
- 덤: 다섯 crop의 확률을 평균(TTA, test-time augmentation)하면 학습 없이
  **75.80 → 77.77%** (+2.0%p). "중앙 하나만 보는 건 손해"의 실증이자,
  목표 80%까지 남은 거리를 2.2%p로 줄인 카드.

## 4. 사람 사진 vs 기계 눈 (테슬라)

| | 사람이 찍은 사진 (우리, ImageNet, COCO) | 기계 눈 (테슬라, CCTV, 의료) |
|---|---|---|
| 편향의 원천 | 촬영자의 구도 (photographer's bias) | 고정된 카메라 기하 |
| 위치의 의미 | **우연** — 피사체가 가운데 "오는 경향"일 뿐 | **인과** — 왼쪽 영역 = 옆 차선, 아래 = 도로 |
| 취급 | 교정 대상 (RandomResizedCrop, TTA) | **활용 대상** (카메라 캘리브레이션 → BEV 변환) |
| 깨지는 순간 | 구도가 다른 사진 (모서리 crop −7%p가 그 예고편) | 카메라 장착이 틀어질 때, 차종이 바뀔 때 |

테슬라 데이터는 사람이 구도를 잡지 않는 **machine-captured** 데이터라
photographer's bias가 없다. 대신 "도로는 항상 아래, 지평선은 그 높이"라는
훨씬 강한 기하 prior가 있고, 이건 우연이 아니라 인과라서 지름길이 아니라
**정당한 구조**다 — 실제로 카메라 장착 위치를 명시적으로 써서 여러 카메라를
조감도(BEV, bird's-eye view) 공간으로 합친다. CoordConv 정신의 대형 버전.

## 한 줄 요약

> CNN에 "보는 순서"는 없다(전 위치 동시 계산). 중앙 편향은 **데이터의 spatial
> prior(구도) + padding 경유 위치 인코딩**의 가중치 형태로 실재하며, 우리 실측
> 크기는 +7%p다. 사람 사진에선 우연한 편향이라 교정(TTA로 +2%p 회수), 고정
> 카메라에선 인과적 구조라 활용 — **같은 '위치 정보'도 데이터의 출신에 따라
> 지름길이 되기도, 정당한 특징이 되기도 한다.**
