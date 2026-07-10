# Deep Learning Study — Stanford 40 Actions

ResNet18 파인튜닝으로 [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html)
(사람 행동 40클래스 분류) **테스트 정확도 80% 달성**을 목표로 하는 학습 프로젝트.

## 환경

- Windows 11 / NVIDIA RTX 3050 Laptop (4GB) / Python 3.14
- PyTorch (CUDA 12.8 빌드)

## 시작하기

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 데이터셋 다운로드 (~291MB, data/ 폴더에 풀림)
python scripts/download_data.py
```

> 데이터셋은 스탠포드가 연구용으로 배포하는 자료라 이 저장소에 포함하지 않습니다.
> 위 스크립트 한 번 실행으로 동일하게 준비됩니다.

## 데이터셋

- 이미지 9,532장, 행동 40클래스 (applauding, cooking, riding_a_horse, ...)
- 공식 split: 클래스당 100장 학습(총 4,000장), 나머지 5,532장 테스트
- split 목록: `data/ImageSplits/<클래스명>_train.txt` / `_test.txt`

## 계획 구조

```
scripts/    데이터 다운로드 등 유틸리티
src/        Dataset, 학습/평가 코드
notebooks/  실험·분석 노트북 (Grad-CAM 시각화 등)
data/       데이터셋 (git 제외)
checkpoints/ 학습된 가중치 (git 제외)
```

## 실험 기록

| # | 설정 | Test Acc | 메모 |
|---|------|----------|------|
| 1 | (예정) fc층만 학습 베이스라인 | - | - |

wandb 대시보드: (추가 예정)

## 학습 로드맵

1. 이론: [모두의 딥러닝 시즌1](https://www.youtube.com/playlist?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)
2. 실습: [모두의 딥러닝 시즌2 PyTorch](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
3. 프로젝트: ResNet18 파인튜닝 → 베이스라인 → 전체 파인튜닝 → 증강/스케줄러 실험
4. 분석: [wandb](https://docs.wandb.ai/examples) 학습 곡선, [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) 오분류 분석
