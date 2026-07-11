# -*- coding: utf-8 -*-
"""파인튜닝이 백본을 ImageNet 가중치에서 얼마나 멀리 옮겼는지 측정.

    python scripts/measure_backbone_drift.py

체크포인트(exp02, exp02b)를 사전학습 가중치와 비교해 상대 L2 이동량을 출력한다.
"학습률이 크다/작다"를 곡선이 아니라 가중치 공간에서 직접 확인하는 도구.
관련: notes/ch2/q08-lr-big-relative-to-what.md
"""
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent.parent
# BatchNorm 통계(running_*)는 기울기로 학습되는 값이 아니라서 제외
SKIP = ("running_mean", "running_var", "num_batches_tracked")


def rel_displacement(pre: dict, ckpt_path: Path):
    """사전학습 대비 상대 이동량 ‖θ_ft − θ_pre‖ / ‖θ_pre‖ (백본 전체, 층별)."""
    ft = torch.load(ckpt_path, map_location="cpu")
    moved, base, per_layer = 0.0, 0.0, {}
    for k, v in pre.items():
        if k.startswith("fc.") or any(s in k for s in SKIP):
            continue  # fc는 새로 만든 층이라 비교 대상이 아님
        v = v.float()
        d = (ft[k].float() - v).pow(2).sum().item()
        b = v.pow(2).sum().item()
        moved, base = moved + d, base + b
        per_layer[k] = (d ** 0.5) / (b ** 0.5)
    return (moved ** 0.5) / (base ** 0.5), per_layer


def main() -> None:
    pre = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()

    w = pre["layer3.0.conv1.weight"].float()
    print(f"사전학습 합성곱 가중치 크기 예시 (layer3.0.conv1): "
          f"표준편차 {w.std():.4f}, 평균|w| {w.abs().mean():.4f}")
    print("에폭당 스텝: 4000/32 = 125 → 10에폭 = 1,250스텝\n")

    for name, fname in [("exp02  (lr 1e-4)", "exp02_full_finetune_best.pt"),
                        ("exp02b (lr 1e-3)", "exp02b_full_lr1e-3_best.pt")]:
        path = ROOT / "checkpoints" / fname
        if not path.exists():
            print(f"{name}: 체크포인트 없음 ({path.name}) — 해당 실험을 먼저 실행")
            continue
        total, per_layer = rel_displacement(pre, path)
        print(f"{name}: 백본 이동량 {total * 100:.1f}% (ImageNet 가중치 대비 상대 L2)")
        print(f"   첫 합성곱(conv1): {per_layer['conv1.weight'] * 100:.1f}%   "
              f"마지막 합성곱(layer4.1.conv2): {per_layer['layer4.1.conv2.weight'] * 100:.1f}%")


if __name__ == "__main__":
    main()
