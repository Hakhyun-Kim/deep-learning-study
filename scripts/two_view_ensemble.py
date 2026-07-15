# -*- coding: utf-8 -*-
"""두-시야 앙상블: '사람 시야' 모델과 '전체 시야' 모델의 의견을 가중 평균한다.

    python scripts/two_view_ensemble.py \
        --person exp05_person_crop --full exp04_aug_epochs20

최종 확률 = w * P_사람시야 + (1-w) * P_전체시야 로 w를 0~1 스윕.
- 사람 시야(person-crop): 미세분류(손의 물건)에 강하고 배경 지름길에 면역.
  대신 말·자전거 같은 바깥 문맥을 잃는다.
- 전체 시야: 그 반대. 서로 다른 걸 보므로 앙상블이 통할 조건을 만족.

설계의 원형: R*CNN (Gkioxari et al., "Contextual Action Recognition with
R*CNN", ICCV 2015) — 사람 영역 + 문맥 영역 점수를 합쳐 행동 분류.
관련: notes/ch3/notes.md exp05, scripts/position_sensitivity.py의 가중치 스윕.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import Stanford40, build_transform, load_class_names  # noqa: E402
from position_sensitivity import build_fivecrop_transform  # noqa: E402
from train import build_model  # noqa: E402

CHECKPOINT_DIR = ROOT / "checkpoints"


@torch.no_grad()
def predict_probs(name, person_crop, crop_margin, device, batch_size, workers,
                  tta=False):
    """체크포인트 name으로 test set 전체의 (N, 40) 확률을 만든다.

    tta=True 면 FiveCrop(네 모서리+중앙) 확률의 균등 평균을 쓴다.
    """
    classes = load_class_names()
    transform = build_fivecrop_transform() if tta else build_transform(train=False)
    test_set = Stanford40("test", transform=transform,
                          person_crop=person_crop, crop_margin=crop_margin)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    model = build_model(freeze_backbone=False, num_classes=len(classes)).to(device)
    ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    probs, labels = [], []
    for images, lbls in loader:
        if tta:
            b = lbls.size(0)
            flat = images.view(-1, *images.shape[2:]).to(device)
            probs.append(model(flat).softmax(dim=1).view(b, 5, -1).mean(dim=1).cpu())
        else:
            probs.append(model(images.to(device)).softmax(dim=1).cpu())
        labels.append(lbls)
    return torch.cat(probs), torch.cat(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", default="exp05_person_crop",
                        help="사람 시야로 학습된 실험 이름")
    parser.add_argument("--full", default="exp04_aug_epochs20",
                        help="전체 시야로 학습된 실험 이름")
    parser.add_argument("--crop-margin", type=float, default=1.5,
                        help="사람 시야 평가에 쓸 bbox margin (학습 때와 같게)")
    parser.add_argument("--tta", action="store_true",
                        help="전체 시야를 FiveCrop TTA(5-crop 평균)로 평가")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사람 시야({args.person}, margin {args.crop_margin}) 추론 중...")
    p_person, labels = predict_probs(args.person, True, args.crop_margin,
                                     device, args.batch_size, args.workers)
    print(f"전체 시야({args.full}{', TTA' if args.tta else ''}) 추론 중...")
    p_full, labels2 = predict_probs(args.full, False, args.crop_margin,
                                    device, args.batch_size, args.workers,
                                    tta=args.tta)
    assert torch.equal(labels, labels2), "두 시야의 test 순서가 다름"
    seen = labels.size(0)

    print(f"\n=== 두-시야 앙상블 (test {seen}장) ===")
    print(f"  사람 시야 단독: {(p_person.argmax(1) == labels).float().mean() * 100:.2f}%")
    print(f"  전체 시야 단독: {(p_full.argmax(1) == labels).float().mean() * 100:.2f}%")
    print("\n[가중치 스윕]  w * 사람시야 + (1-w) * 전체시야")
    best_w, best_acc = 0.0, 0.0
    for w10 in range(0, 11):
        w = w10 / 10
        acc = ((w * p_person + (1 - w) * p_full).argmax(1) == labels
               ).float().mean().item()
        print(f"  w={w:.1f}  {acc * 100:5.2f}%")
        if acc > best_acc:
            best_w, best_acc = w, acc
    print(f"\n  최적: w={best_w:.1f} → {best_acc * 100:.2f}%")
    print("  주의: w를 test set으로 고르면 살짝 과적합 — 결론은 '~그 근처' 정도로.")


if __name__ == "__main__":
    main()
