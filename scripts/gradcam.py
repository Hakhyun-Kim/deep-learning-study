# -*- coding: utf-8 -*-
"""Grad-CAM: 모델이 이미지의 '어디를 보고' 그 예측을 했는지 열지도로 본다.

    python scripts/gradcam.py                         # 기본 예시 몇 장
    python scripts/gradcam.py --images phoning_253.jpg smoking_012.jpg

결과는 runs/gradcam/<파일명>.png 로 저장된다 (data/ 처럼 gitignore 대상).
각 그림은 [원본 | 예측 클래스에 대한 열지도 | 정답 클래스에 대한 열지도].

원리(한 줄): 마지막 합성곱 블록(layer4)이 내놓은 특징 지도(7x7x512)에 대해
"이 클래스 점수를 올리는 데 각 채널이 얼마나 기여했나"(기울기)를 가중치로 곱해
더하면, '이 예측의 근거가 이미지 어디에 있었는지'가 7x7 지도로 나온다.
그걸 224로 키워 원본에 겹친다. ch3(분석) 도구. 관련: notes/ch3/notes.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt

# 제목에 한글을 쓰므로 한글 글꼴을 우선 지정 (Windows: Malgun Gothic).
# 없으면 DejaVu로 떨어지고 한글이 네모로 보일 수 있다.
for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == _font for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import (IMAGENET_MEAN, IMAGENET_STD, Stanford40,  # noqa: E402
                     build_transform, load_class_names)
from train import build_model  # noqa: E402

CHECKPOINT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "runs" / "gradcam"

# analyze_errors.py 가 뽑아준 "가장 확신하며 틀린 예" 중 일부 (손-얼굴 계열 위주)
DEFAULT_IMAGES = [
    "phoning_253.jpg", "waving_hands_151.jpg", "reading_076.jpg",
    "taking_photos_191.jpg", "applauding_125.jpg", "watching_TV_017.jpg",
]


def compute_cam(model, layer, x, class_idx):
    """이미지 텐서 x(1,3,224,224)에 대해 class_idx의 Grad-CAM 열지도(224x224)를 만든다."""
    acts, grads = {}, {}

    def fwd_hook(_m, _in, out):
        acts["v"] = out
        out.register_hook(lambda g: grads.__setitem__("v", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        logits = model(x)
        model.zero_grad()
        logits[0, class_idx].backward()      # 이 클래스 점수를 기준으로 역전파
    finally:
        handle.remove()

    A = acts["v"][0]                          # (512, 7, 7) 특징 지도
    weights = grads["v"][0].mean(dim=(1, 2))  # 채널별 기여도 = 기울기의 공간 평균
    cam = F.relu((weights[:, None, None] * A).sum(0))  # (7,7), 양의 기여만
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=224, mode="bilinear",
                        align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


def denormalize(x):
    """정규화된 텐서를 사람이 보는 0~1 이미지로 되돌린다."""
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    img = (x[0].cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="exp02_full_finetune")
    parser.add_argument("--images", nargs="*", default=DEFAULT_IMAGES,
                        help="JPEGImages 안의 파일명들 (생략 시 기본 예시)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_class_names()
    ckpt = CHECKPOINT_DIR / f"{args.name}_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt}")

    model = build_model(freeze_backbone=False, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 파일명 → 정답 라벨 (test split 기준)
    test_set = Stanford40("test")
    label_of = {p.name: lbl for p, lbl in test_set.samples}
    transform = build_transform(train=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in args.images:
        path = ROOT / "data" / "JPEGImages" / fname
        if not path.exists():
            print(f"건너뜀(파일 없음): {fname}")
            continue
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        x = transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = model(x).softmax(1)[0]
        pred = int(probs.argmax())
        true = label_of.get(fname, pred)

        cam_pred = compute_cam(model, model.layer4, x, pred)
        cam_true = compute_cam(model, model.layer4, x, true)
        base = denormalize(x)

        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        axes[0].imshow(base)
        axes[0].set_title(f"원본\n정답: {classes[true]}", fontsize=10)
        axes[1].imshow(base)
        axes[1].imshow(cam_pred, cmap="jet", alpha=0.45)
        axes[1].set_title(f"예측: {classes[pred]} ({probs[pred] * 100:.0f}%)\n"
                          f"→ 여기를 보고 이렇게 판단", fontsize=10)
        axes[2].imshow(base)
        axes[2].imshow(cam_true, cmap="jet", alpha=0.45)
        axes[2].set_title(f"정답 클래스 근거\n{classes[true]} ({probs[true] * 100:.0f}%)",
                          fontsize=10)
        for ax in axes:
            ax.axis("off")
        # 한글 폰트가 없으면 제목의 한글이 깨질 수 있으니 파일명은 영문으로
        correct = "OK" if pred == true else "WRONG"
        fig.suptitle(f"[{correct}] {fname}", fontsize=11)
        fig.tight_layout()
        out = OUT_DIR / f"{path.stem}_gradcam.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        mark = "✓" if pred == true else "✗"
        print(f"{mark} {fname}: 정답 {classes[true]} / 예측 {classes[pred]} "
              f"({probs[pred] * 100:.0f}%) → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
