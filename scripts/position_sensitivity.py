# -*- coding: utf-8 -*-
"""모델이 '이미지의 어느 위치'에 얼마나 기대는지 잰다 (위치 민감도 진단).

    python scripts/position_sensitivity.py --name exp03_aug_stronger

test set의 각 이미지를 Resize(256) 후 **다섯 위치**(네 모서리 + 중앙)로 잘라
같은 모델로 따로따로 추론한다 (torchvision FiveCrop).

- 중앙 crop 정확도가 모서리보다 유의하게 높으면 → 모델과 데이터가
  "중요한 건 가운데"라는 공간 prior(spatial prior)에 기대고 있다는 실증.
  사람이 찍은 사진의 photographer's bias (Torralba & Efros,
  "Unbiased Look at Dataset Bias", CVPR 2011) 참고.
- 덤: 다섯 crop의 softmax **평균**(= TTA, test-time augmentation)의 정확도도
  출력한다. 평소 평가(CenterCrop 하나 = 다섯 중 '중앙'과 동일)보다 높으면
  "중앙 하나만 보는 건 손해"라는 뜻 — ch3 concepts의 'TTA' 카드 미리보기.

학습이 아니라 추론만이라 몇 분이면 끝난다. 관련: notes/ch3/q12(예정),
notes/ch3/concepts.md. ch3(분석) 곁가지 도구.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import (IMAGENET_MEAN, IMAGENET_STD, Stanford40,  # noqa: E402
                     load_class_names)
from train import build_model  # noqa: E402

CHECKPOINT_DIR = ROOT / "checkpoints"

# torchvision FiveCrop이 돌려주는 순서
POSITIONS = ["왼쪽 위", "오른쪽 위", "왼쪽 아래", "오른쪽 아래", "중앙"]


class StackCrops:
    """FiveCrop이 준 PIL 5장을 (5, 3, 224, 224) 텐서 하나로 쌓는다.

    lambda로 쓰면 Windows DataLoader 워커가 pickle을 못 해서 클래스로 정의.
    """

    def __init__(self):
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, crops):
        return torch.stack(
            [self.normalize(transforms.functional.to_tensor(c)) for c in crops])


def build_fivecrop_transform() -> transforms.Compose:
    """이미지 1장 → (5, 3, 224, 224) 텐서 (네 모서리 + 중앙 crop)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),  # PIL 5장 튜플로
        StackCrops(),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="exp03_aug_stronger",
                        help="실험 이름 (checkpoints/<name>_best.pt 를 읽음)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = CHECKPOINT_DIR / f"{args.name}_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt}")

    classes = load_class_names()
    test_set = Stanford40("test", transform=build_fivecrop_transform())
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(freeze_backbone=False, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for crops, labels in loader:            # crops: (B, 5, 3, 224, 224)
            b = labels.size(0)
            flat = crops.view(-1, *crops.shape[2:]).to(device)  # (B*5, 3, 224, 224)
            all_probs.append(model(flat).softmax(dim=1).view(b, 5, -1).cpu())
            all_labels.append(labels)
    probs = torch.cat(all_probs)                # (N, 5, 40)
    labels = torch.cat(all_labels)              # (N,)
    seen = labels.size(0)

    print(f"\n=== {args.name} — 위치 민감도 (test {seen}장, FiveCrop) ===")
    print("[위치별 단독 정확도]  (평소 평가 = '중앙'과 동일)")
    correct = (probs.argmax(dim=2) == labels[:, None]).sum(dim=0)  # 위치별
    for name, c in zip(POSITIONS, correct.tolist()):
        bar = "#" * round(c / seen * 50)
        print(f"  {name:6s} {c / seen * 100:5.2f}%  {bar}")
    corner_avg = correct[:4].sum().item() / (seen * 4)
    center = correct[4].item() / seen
    print(f"\n  모서리 평균 {corner_avg * 100:.2f}% vs 중앙 {center * 100:.2f}% "
          f"(격차 {(center - corner_avg) * 100:+.2f}%p)")

    # "중앙을 얼마나 더 믿어야 하나" 스윕:
    #   최종 확률 = w * 중앙 + (1-w) * 모서리 평균.
    #   w=1.0 이면 평소 평가(중앙만), w=0.2 이면 균등 평균(=보통의 TTA).
    print("\n[중앙 가중치 스윕]  w * 중앙 + (1-w) * 모서리평균")
    corners_mean = probs[:, :4].mean(dim=1)     # (N, 40)
    center_p = probs[:, 4]                      # (N, 40)
    best_w, best_acc = 0.0, 0.0
    for w10 in range(0, 11):
        w = w10 / 10
        mixed = w * center_p + (1 - w) * corners_mean
        acc = (mixed.argmax(dim=1) == labels).float().mean().item()
        tag = {0.2: " ← 균등 평균(TTA)", 1.0: " ← 평소 평가(중앙만)"}.get(w, "")
        print(f"  w={w:.1f}  {acc * 100:5.2f}%{tag}")
        if acc > best_acc:
            best_w, best_acc = w, acc
    print(f"\n  최적: w={best_w:.1f} → {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
