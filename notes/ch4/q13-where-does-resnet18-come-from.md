# q13 — 우리가 쓰던 ResNet18은 어디서 온 거야?

관련: [`train.py`](../../src/train.py) `build_model()` ·
[ch1 q01 — 백본 얼림](../ch1/q01-backbone-freeze.md) ·
[ch3 q11 — PyTorch 산업 스택](../ch3/q11-pytorch-industry-stack.md)

## 한 줄 답

**설계도는 He et al.(Microsoft Research, 2015) 논문에서, 코드 구현은
torchvision에서, 시각 지식(가중치)은 ImageNet 사전학습에서** 왔다. 우리는
그 위에 Stanford 40을 얹었을 뿐이다. 세 층위를 구분하는 게 핵심.

## 층위 ① 설계도 — Deep Residual Learning (CVPR 2016 최우수논문)

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun,
  *"Deep Residual Learning for Image Recognition"* (CVPR 2016).
  Microsoft Research 베이징에서 나왔고, 2015년 ImageNet 대회(ILSVRC) 우승.
  딥러닝 역사상 최다 인용급 논문.
- 풀던 문제 — **degradation problem**: 층을 깊게 쌓으면 좋아져야 할 것 같은데,
  실제로는 20층보다 56층이 **학습조차** 더 안 됐다 (과적합이 아니라 최적화 실패).
- 해법 — **잔차 연결 (residual/skip connection)**: 블록이 출력 전체 대신
  입력에 더할 **수정분(residual)** 만 배우고, 입력은 지름길로 그대로 흘린다.

```
입력 x ──[합성곱 블록 F(x)]──(+)── 출력 F(x) + x
   └──────── 지름길(skip) ────┘
```

- 이 덧셈 덕에 기울기가 깊은 층까지 잘 흘러 152층, 이후 1000층+도 학습 가능해짐.
- "18" = 가중치 있는 층 18개. 형제: ResNet-34/50/101/152. 우리가 18을 쓴 이유는
  가장 작아서(11.2M 파라미터) — 4GB GPU에 배치 32로 여유 있게 들어간다.

## 층위 ② 구현 — torchvision

[`train.py`](../../src/train.py)의 `resnet18(weights=...)` 함수는
**torchvision**(PyTorch 공식 비전 라이브러리)의 구현이다. He의 설계도를
PyTorch 팀이 코드로 옮겨 검증해둔 것 — 소스 공개라 직접 읽고 수정도 가능
([ch3 q11](../ch3/q11-pytorch-industry-stack.md)).

## 층위 ③ 가중치 — ImageNet 사전학습

- `weights=ResNet18_Weights.IMAGENET1K_V1`: 구조는 빈 그릇이고, "이미지 보는
  법"은 **ImageNet-1k**(1,000클래스, 학습 약 128만 장 — Deng et al. 2009,
  Russakovsky et al. 2015)로 PyTorch 팀이 미리 학습한 1,120만 개 숫자에 있다.
- 첫 실행 때 `download.pytorch.org`에서 ~45MB를 받아 로컬 캐시(`~/.cache/torch`)
  에 저장. 이후엔 캐시 사용.
- 우리의 전이학습 = 이 세 층위를 받아와 fc층만 40클래스로 교체
  ([`train.py`](../../src/train.py) `build_model`) 후 4,000장으로 미세조정.
  밑바닥부터였으면 4,000장으로는 불가능 (exp01~02에서 체감한 그대로).

## 테슬라 연결

FSD 인식 백본도 이 계보다 — AI Day 2021에 공개된 HydraNet의 백본 **RegNet**은
ResNet의 잔차 블록 설계 공간을 체계적으로 최적화한 직계 후손
(Radosavovic et al., *"Designing Network Design Spaces"*, CVPR 2020).
2015년의 skip connection 하나가 지금 도로 위에서 실시간으로 돌고 있다.
여담: Kaiming He는 이후 Mask R-CNN, MoCo(자기지도학습 —
[ch1 q10](../ch1/q10-learning-paradigms-and-llms.md))를 만들었고 현재 MIT 교수.

## 한 줄 요약

> 모델 하나에도 출처가 셋이다: **아이디어(논문) · 구현(라이브러리) · 지식(사전학습
> 가중치)**. 전이학습은 이 셋을 물려받아 마지막 층만 내 문제로 바꾸는 것.
