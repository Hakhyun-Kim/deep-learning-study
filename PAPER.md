# 진단 주도 개선과 두-시야 앙상블로 달성한 Stanford 40 행동 인식 80.7%: ResNet18 기반 기술 보고서

**Hakhyun Kim** · 2026-07-15 · [코드/실험 기록](https://github.com/Hakhyun-Kim/deep-learning-study) · [대시보드](https://hakhyun-kim.github.io/deep-learning-study/)

> 형식 연습을 겸한 기술 보고서(technical report)로, 동료 심사를 거친 논문이
> 아니다. 모든 수치는 저장소의 스크립트로 재현 가능하다.

## 초록 (Abstract)

본 보고서는 4GB GPU 한 대와 ResNet18만으로 Stanford 40 Actions(행동 40클래스,
test 5,532장)에서 top-1 정확도 **80.71%**를 달성한 과정을 기술한다. 핵심은
새로운 구조가 아니라 **진단 주도(diagnosis-driven) 개선 루프**다: (1) Grad-CAM과
혼동행렬로 오답을 배경 지름길(shortcut)·미수렴·미세분류(fine-grained)의 세
유형으로 분해하고, (2) 각 유형에 대응하는 처방(증강, 학습 연장, person-crop)을
한 번에 한 변수씩 적용하며, (3) 같은 진단을 반복해 처방의 효과를 검증했다.
단독으로는 기준보다 5%p 낮은 사람 시야(person-crop) 모델(70.79%)이 전체 시야
모델과 오답이 겹치지 않는다는 관찰로부터, 두 시야의 확률을 3:7로 가중 평균하는
앙상블을 구성해 77.77%(TTA) 대비 +2.9%p를 얻었다. 부가적으로, 촬영자 구도
편향(photographer's bias)을 five-crop 위치별 정확도 차이(+7.1%p)로 정량화했다.

*We report 80.71% top-1 accuracy on Stanford 40 Actions using only two
ResNet18 models on a single 4GB GPU. Rather than novel architecture, we use a
diagnosis-driven loop: decomposing errors into background shortcuts,
under-convergence, and fine-grained confusion via Grad-CAM and confusion
matrices, prescribing one controlled change at a time, and re-diagnosing. A
person-view model that is 5%p weaker alone (70.79%) proves complementary to
the full view, and a 3:7 weighted two-view ensemble reaches 80.71%. We also
quantify photographer's bias as a +7.1%p center-vs-corner crop accuracy gap.*

## 1. 서론

정지 영상 행동 인식(still image action recognition)은 시간 정보 없이 한 장의
사진에서 사람의 행동을 분류하는 과제다. 본 프로젝트의 목표는 제한된 자원
(RTX 3050 Ti 4GB, ResNet18)으로 Stanford 40 [1]에서 test 정확도 80%를 넘기는
것이었다. 대형 백본 경쟁 대신, 다음 질문을 반복하는 것을 방법론으로 삼았다:
**"모델은 지금 무엇을 보고 틀리는가, 그리고 그 이유에 맞는 최소한의 처방은
무엇인가."**

기여는 세 가지다:

1. **오답의 유형 분해와 유형별 처방의 검증**: 배경 지름길에는 증강과
   person-crop이, 미수렴에는 학습 연장이 대응하며, 미세분류는 본 구성의
   한계로 남음을 통제 실험으로 보였다 (§4–5).
2. **상보적 두-시야 앙상블**: 단독 성능이 낮은 사람 시야 모델이 전체 시야
   모델과 상보적(complementary)임을 보이고, 가중 평균만으로 80.71%를 달성했다
   (§4.4).
3. **구도 편향의 정량화**: 사람이 찍은 사진 데이터셋의 중앙 편향을 five-crop
   위치별 정확도로 측정(+7.1%p)하고, 이것이 모델이 아닌 데이터에서 옴을
   보였다 (§4.3).

## 2. 관련 연구

**Stanford 40과 사람 중심 인식.** Stanford 40 [1]은 이미지당 대상 인물의
bounding box를 제공하며, 원 과제 정의가 "주어진 사람의 행동 분류"다. R\*CNN [2]은
사람 영역과 문맥 영역의 점수를 결합해 90.9% mAP를 달성했고, 이후 attention·
관계 모델링 계열이 94–95.5% mAP까지 끌어올렸다 [3]. 본 보고서의 두-시야
앙상블은 R\*CNN의 "사람+문맥" 원리를 훈련이 아닌 추론 단계 결합으로 재현한
경량 버전에 해당한다.

**지름길 학습.** 모델이 과제의 본질 대신 데이터에 우연히 상관된 쉬운 단서로
답하는 현상은 shortcut learning [4]으로 정식화되어 있다. 본 작업은 Grad-CAM
[5]을 진단 도구로 사용해 개별 오답의 지름길 여부를 판별했다.

**추론 시 증강과 앙상블.** Five/ten-crop 평가는 AlexNet [6] 이래의 고전적
TTA(test-time augmentation)다. 서로 다르게 틀리는 모델의 결합이 개별보다
낫다는 것은 앙상블의 고전적 결과다 [7].

## 3. 방법

### 3.1 기본 파이프라인

ImageNet 사전학습 ResNet18 [8]의 마지막 fc층을 40클래스로 교체하고 전체를
파인튜닝한다 (AdamW, lr 1e-4, batch 32). 학습 증강은 RandomResizedCrop(224),
좌우반전에 ColorJitter(±0.3)와 RandomErasing [9] (p=0.5, 최대 20%)을 더했다.

### 3.2 두 시야 (two views)

- **전체 시야** V_full: 원본 이미지 전체 (Resize 256 → crop 224). 평가 시
  five-crop TTA(네 모서리+중앙의 softmax 평균)를 사용.
- **사람 시야** V_person: 공식 bbox를 1.5배 margin으로 확장해 자른 영역.
  배경 지름길을 물리적으로 제거하고 사람의 유효 해상도를 높이는 대신, bbox
  밖의 문맥(말, 자전거)을 잃는다.

두 시야로 **각각 별도의 모델**을 같은 레시피로 학습한 뒤, 추론 시 확률을
가중 평균한다:

$$p = w \cdot p_{\text{person}} + (1-w) \cdot p_{\text{full-TTA}}, \quad w=0.3$$

### 3.3 진단 도구

(a) 클래스별 정확도·혼동행렬, (b) Grad-CAM [5] — 마지막 합성곱 블록 특징 지도를
클래스 점수에 대한 기울기로 가중합해 판단 근거 위치를 시각화, (c) five-crop
위치별 단독 정확도 — 구도 편향의 정량 측정. 모두 저장소의 `scripts/`로 공개.

## 4. 실험

### 4.1 설정

Stanford 40 공식 split (train 4,000 / test 5,532), top-1 정확도. 단일
RTX 3050 Ti Laptop 4GB, 에폭당 25–70초.

### 4.2 주 결과

| 시스템 | 변화 (직전 대비 한 변수) | Test top-1 |
|---|---|---|
| fc만 학습 (백본 동결) | — | 69.4% |
| 전체 파인튜닝 (exp02) | 백본 해동 | 75.7% |
| + lr 10배 (통제 실험) | lr 1e-3 | 55.0% (붕괴) |
| + 증강 (exp03) | ColorJitter/RandomErasing | 75.05% |
| + 에폭 20 (exp04) | 학습 길이 | 75.80% |
| person-crop 단독 (exp05) | 입력 시야 | 70.79% |
| exp04 + five-crop TTA | 추론 방식 | 77.77% |
| **두-시야 앙상블 (w=0.3)** | 시야 결합 | **80.71%** |

### 4.3 진단 결과

**지름길.** exp02의 최고 확신 오답 3건은 모두 사람이 아닌 배경을 보고 있었다
(예: 칠판 앞 독서 사진을 벽의 글씨를 근거로 "칠판에 쓰기" 100% 확신). 증강은
작은 배경 단서 의존을 줄였으나(waving_hands +13.6%p), 화면 절반을 차지하는
큰 배경(칠판)은 최대 20% 조각 삭제로 가려지지 않았다.

**구도 편향.** 다섯 위치 단독 정확도는 중앙 75.80% vs 모서리 평균 68.66%
(**+7.14%p**)였으며, 증강 강도가 다른 exp02에서도 격차가 동일(+7.08%p)해
이 편향이 모델이 아니라 데이터의 촬영 구도에서 옴을 시사한다. 중앙 가중치
스윕에서는 균등 평균(w=0.2)보다 중앙을 약간 더 믿는 w=0.3이 최적(77.98%)이었다.

**미수렴.** exp03의 test loss는 10에폭에도 하강 중이었고, 20에폭 연장(exp04)
시 12에폭에서 최적점 후 과적합이 시작됐다 — 증강 강화의 비용(수렴 지연)과
천장(~76%)을 함께 확인했다.

### 4.4 두-시야 앙상블

사람 시야 단독은 70.79%로 전체 시야보다 5.0%p 낮다. 그러나 가중 평균 스윕 결과:

| w (사람 비중) | 0.0 | 0.2 | **0.3** | 0.4 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| 전체=단일 crop | 75.81 | 77.93 | 78.63 | 78.99 | **79.37** | 70.79 |
| 전체=five-crop TTA | 77.77 | 80.30 | **80.71** | 80.22 | 79.52 | 70.79 |

w=0.2–0.4 전 구간이 80%를 상회해, "80% 돌파"는 가중치 선택에 강건하다.

## 5. 분석: 오답의 종류가 바뀌었다

최종 시스템 재진단에서 약한 클래스는 전부 개선됐고(waving_hands +11.8%p 등),
남은 하위 8개 클래스는 모두 손-소물체 미세분류 계열이었다(혼동 1위:
phoning→smoking 31회). 상징적 사례로, exp02가 벽 글씨를 근거로 99% 확신
오답을 내던 이미지는 최종 시스템에서 "손 근처 물체를 담배로 오인"(41%)으로
바뀌었다 — 여전히 오답이지만 **옳은 곳(사람)을 보고 틀리는 오답**이다. 즉
파이프라인은 장면 인식기를 행동 인식기로 이동시켰고, 남은 병목은 224px
해상도와 클래스당 100장으로는 풀기 어려운 미세분류 하나로 수렴했다.

## 6. 한계

1. **미세분류 병목 잔존** — 하위 8개 클래스 전부 손-물체 구분 문제.
2. **지름길 일부 생존** — 배경 단서가 사람 crop 안에 들어오는 경우(칠판 앞
   독서 1건, 95% 확신 오답 유지).
3. **bbox 주석 가정** — 80.71%는 공식 사람 bbox 사용 수치. 실전 배포 시 사람
   탐지기의 오류율이 추가된다.
4. **가중치의 test 선택** — w를 test 성적으로 골랐다(결론은 w=0.2–0.4 전
   구간에서 유지되나, 소수점은 낙관 편향). 엄밀하게는 validation split 필요.
5. **추론 비용** — 이미지당 순전파 6회(전체 5 + 사람 1). 배치 처리용이며
   실시간에는 부적합.
6. **지표** — 본 보고서는 top-1 정확도를 사용했다. mAP를 쓰는 문헌 [2,3]과의
   직접 비교는 불가하다.

## 7. 결론

새 구조 없이, 진단(무엇을 보고 틀리는가) → 한 변수 처방 → 재진단의 루프와
상보적 시야의 결합만으로 ResNet18 두 개가 Stanford 40에서 80.71%에 도달했다.
가장 일반화 가능한 교훈은 두 가지다: (1) 점수를 올린 모든 결정은 측정에서
나왔다 — 진단 없는 처방(lr 10배)은 −20.7%p로 응답했다. (2) 단독 성능이 낮은
모델도 다른 것을 본다면 버릴 것이 아니라 앙상블의 재료다.

## 참고문헌

[1] B. Yao, X. Jiang, A. Khosla, A. L. Lin, L. Guibas, L. Fei-Fei,
"Human Action Recognition by Learning Bases of Action Attributes and Parts,"
*ICCV*, 2011.

[2] G. Gkioxari, R. Girshick, J. Malik, "Contextual Action Recognition with
R\*CNN," *ICCV*, 2015. [arXiv:1505.01197](https://arxiv.org/abs/1505.01197)

[3] "Human Action Recognition in Still Images Using ConViT," 2023.
[arXiv:2307.08994](https://arxiv.org/abs/2307.08994)

[4] R. Geirhos et al., "Shortcut Learning in Deep Neural Networks,"
*Nature Machine Intelligence*, 2020.

[5] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization," *ICCV*, 2017.

[6] A. Krizhevsky, I. Sutskever, G. E. Hinton, "ImageNet Classification with
Deep Convolutional Neural Networks," *NeurIPS*, 2012.

[7] T. G. Dietterich, "Ensemble Methods in Machine Learning,"
*Multiple Classifier Systems*, 2000.

[8] K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image
Recognition," *CVPR*, 2016.

[9] Z. Zhong et al., "Random Erasing Data Augmentation," *AAAI*, 2020.

---

*재현: `python scripts/two_view_ensemble.py --tta` (체크포인트는
`src/train.py`로 학습: exp04는 `--epochs 20`, exp05는 추가로 `--person-crop`).
실험별 학습 곡선: [대시보드](https://hakhyun-kim.github.io/deep-learning-study/).*
