# -*- coding: utf-8 -*-
"""학습이 끝난 모델이 test set에서 '무엇을' 틀리는지 진단한다.

    python scripts/analyze_errors.py                       # exp02 기본
    python scripts/analyze_errors.py --name exp02b_full_lr1e-3

전체 정확도 하나로는 "어디가 약한지"가 안 보인다. 이 스크립트는 test set을
한 번 추론하면서 다음을 뽑는다:

  1) 클래스별 정확도 (약한 순으로) — 40개 중 어느 행동을 자주 틀리나
  2) 혼동 쌍 (정답 → 오답) 상위 목록 — 무엇을 무엇으로 착각하나
  3) 가장 확신하며 틀린 예 — Grad-CAM으로 들여다볼 후보

ch3(분석과 개선)의 첫 단계: 증강·스케줄러를 아무거나 시도하기 전에,
'어디가 새는지'를 먼저 본다. 관련: notes/ch3/notes.md
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import Stanford40, build_transform, load_class_names  # noqa: E402
from train import build_model  # noqa: E402

CHECKPOINT_DIR = ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="exp02_full_finetune",
                        help="분석할 실험 이름 (checkpoints/<name>_best.pt 를 읽음)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--top", type=int, default=12,
                        help="약한 클래스·혼동 쌍을 몇 개까지 볼지")
    return parser.parse_args()


@torch.no_grad()
def collect(model, loader, device, num_classes):
    """test set 전체를 추론해 (혼동행렬, 확신하며 틀린 예 목록)을 만든다.

    confusion[t][p] = 정답이 t인데 p로 예측한 개수.
    대각선(t==p)이 맞힌 것, 나머지가 틀린 것.
    """
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    confident_wrong = []  # (확률, 정답라벨, 오답라벨, 이미지경로)
    idx = 0
    for images, labels in loader:
        images = images.to(device)
        probs = model(images).softmax(dim=1).cpu()  # 점수 → 확률
        conf, preds = probs.max(dim=1)               # 가장 확신한 클래스
        for t, p, c in zip(labels.tolist(), preds.tolist(), conf.tolist()):
            confusion[t][p] += 1
            if t != p:
                confident_wrong.append((c, t, p, loader.dataset.samples[idx][0]))
            idx += 1
    return confusion, confident_wrong


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = CHECKPOINT_DIR / f"{args.name}_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt} — 해당 실험을 먼저 학습하세요")

    classes = load_class_names()
    test_set = Stanford40("test", transform=build_transform(train=False))
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(freeze_backbone=False, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    confusion, confident_wrong = collect(model, loader, device, len(classes))

    per_class_total = confusion.sum(dim=1)                 # 클래스별 test 이미지 수
    per_class_correct = confusion.diag()
    overall = per_class_correct.sum().item() / per_class_total.sum().item()

    print(f"\n=== {args.name} — test set 진단 ({int(per_class_total.sum())}장) ===")
    print(f"전체 정확도: {overall * 100:.2f}%\n")

    # 1) 약한 클래스 (정확도 낮은 순)
    acc = (per_class_correct.float() / per_class_total.clamp(min=1)).tolist()
    order = sorted(range(len(classes)), key=lambda i: acc[i])
    print(f"[약한 클래스 {args.top}개] (정확도 · 맞힘/전체)")
    for i in order[:args.top]:
        print(f"  {acc[i] * 100:5.1f}%  {per_class_correct[i]:3d}/{per_class_total[i]:<3d}  {classes[i]}")

    # 2) 혼동 쌍 (정답 t → 오답 p), 많이 헷갈린 순
    pairs = [((t, p), int(confusion[t][p]))
             for t in range(len(classes)) for p in range(len(classes))
             if t != p and confusion[t][p] > 0]
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    print(f"\n[혼동 쌍 상위 {args.top}개]  정답 → 오답 (횟수)")
    for (t, p), n in pairs[:args.top]:
        print(f"  {n:3d}회   {classes[t]:<24s} → {classes[p]}")

    # 3) 가장 확신하며 틀린 예 (Grad-CAM 후보)
    confident_wrong.sort(reverse=True)  # 확률 높은 순
    print(f"\n[가장 확신하며 틀린 예 {args.top}개]  확신도 · 정답 → 오답 · 파일")
    for c, t, p, path in confident_wrong[:args.top]:
        print(f"  {c * 100:4.1f}%  {classes[t]:<22s} → {classes[p]:<22s}  {Path(path).name}")

    # 강한 클래스도 대비로 몇 개 (진단 균형용)
    print(f"\n[강한 클래스 5개]")
    for i in order[-5:][::-1]:
        print(f"  {acc[i] * 100:5.1f}%  {per_class_correct[i]:3d}/{per_class_total[i]:<3d}  {classes[i]}")


if __name__ == "__main__":
    main()
