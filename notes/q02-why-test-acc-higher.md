# Q2. 왜 test_acc가 train_acc보다 높지?

> 2026-07-11 · 관련: exp01 학습 곡선 ([대시보드](https://hakhyun-kim.github.io/deep-learning-study/))
> 상태: 🤔 스스로 생각해보기 — 답을 정리하면 상태를 ✅로 바꾸기

## 관찰

exp01에서 매 에폭 test_acc가 train_acc보다 5~30%p 높았다.
보통 "모델은 학습 데이터에서 더 잘한다(과적합 방향)"고 배우는데 반대다.

| 에폭 | train_acc | test_acc |
|---|---|---|
| 1 | 25.1% | 54.6% |
| 10 | 63.9% | 69.2% |

## 힌트

1. [`src/dataset.py`](../src/dataset.py)의 `build_transform()`에서 train과 test의 변환이 어떻게 다른가?
   학습 정확도는 **어떤 이미지**에 대해 측정되고 있나?
2. [`src/train.py`](../src/train.py)의 `train_one_epoch()`은 에폭 **도중에** 정확도를 누적한다.
   에폭 초반의 모델과 끝날 때의 모델은 같은 모델인가?

## 내 답 (직접 채우기)

(여기에 정리)

## 확인 실험 아이디어

- train 데이터를 test용 변환(가운데 크롭)으로 평가해보면 train_acc가 어떻게 변할까?
