"""ResNet18을 Stanford 40 Actions에 파인튜닝하는 학습 스크립트.

실험 1 — 베이스라인 (백본 얼리고 마지막 fc층만 학습):
    python src/train.py --name exp01_fc_only --freeze-backbone

실험 2 — 전체 파인튜닝 (모든 층을 낮은 학습률로 미세조정):
    python src/train.py --name exp02_full_finetune --lr 1e-4

동작 확인용 (일부 데이터로 1에폭만, 대시보드/체크포인트에 기록 안 함):
    python src/train.py --smoke

에폭마다 docs/metrics/<실험명>.json 이 갱신되고, git push 하면
https://hakhyun-kim.github.io/deep-learning-study/ 대시보드에 곡선이 나타난다.
테스트 정확도가 갱신될 때마다 checkpoints/<실험명>_best.pt 에 가중치를 저장한다.
"""
import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18

from dataset import Stanford40, build_transform, load_class_names
from metrics_logger import MetricsLogger

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="실험 이름 (대시보드/체크포인트 파일명)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률. 전체 파인튜닝 때는 1e-4처럼 낮게")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="백본(합성곱층) 가중치를 얼리고 fc층만 학습")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader 워커 수 (0이면 메인 프로세스에서 로드)")
    parser.add_argument("--smoke", action="store_true",
                        help="파이프라인 동작 확인: 일부 데이터로 1에폭만")
    args = parser.parse_args()
    if not args.smoke and not args.name:
        parser.error("--name 을 지정하세요 (예: --name exp01_fc_only)")
    return args


def build_model(freeze_backbone: bool, num_classes: int) -> nn.Module:
    # ImageNet(1000클래스, 120만 장)으로 사전학습된 가중치에서 시작한다.
    # 밑바닥부터 학습하면 4,000장으로는 어림도 없지만, 이미 "이미지 보는 법"을
    # 아는 모델을 가져와 우리 문제에 맞게 조정하는 것이 전이학습(transfer learning).
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        # requires_grad=False 면 역전파 때 기울기를 계산/갱신하지 않는다 → "얼림"
        for param in model.parameters():
            param.requires_grad = False
    # 마지막 분류층은 ImageNet 1000클래스용이므로 40클래스용 새 층으로 교체.
    # 새로 만든 층은 기본적으로 requires_grad=True라서 얼리더라도 이 층은 학습된다.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()  # 학습 모드: 드롭아웃/배치정규화가 학습용으로 동작
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # 이전 배치의 기울기 초기화
        outputs = model(images)          # 순전파: 예측값 계산
        loss = criterion(outputs, labels)  # 예측과 정답의 차이(손실)
        loss.backward()                  # 역전파: 손실에 대한 기울기 계산
        optimizer.step()                 # 기울기 방향으로 가중치 갱신

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()  # 평가 때는 기울기가 필요 없으니 계산을 꺼서 메모리/시간 절약
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()  # 평가 모드
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_set = Stanford40("train", transform=build_transform(train=True))
    test_set = Stanford40("test", transform=build_transform(train=False))
    if args.smoke:
        args.name, args.epochs, args.workers = "smoke", 1, 0
        train_set = Subset(train_set, range(0, len(train_set), 20))  # 200장
        test_set = Subset(test_set, range(0, len(test_set), 20))     # ~277장

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             persistent_workers=args.workers > 0)

    model = build_model(args.freeze_backbone, num_classes=len(load_class_names())).to(device)
    criterion = nn.CrossEntropyLoss()
    # 얼린 파라미터는 옵티마이저에 넣지 않는다 (넣어도 안 바뀌지만 명시적으로)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"학습 대상 파라미터: {sum(p.numel() for p in trainable):,}개 "
          f"/ 전체 {sum(p.numel() for p in model.parameters()):,}개")

    logger = None
    if not args.smoke:
        logger = MetricsLogger(args.name, config={
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "freeze_backbone": args.freeze_backbone, "device": str(device),
        })

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start

        print(f"[{epoch:2d}/{args.epochs}] "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"test_loss {test_loss:.4f} test_acc {test_acc:.4f} | {elapsed:.0f}초")
        if logger:
            logger.log(train_loss=train_loss, train_acc=train_acc,
                       test_loss=test_loss, test_acc=test_acc)

        if test_acc > best_acc and not args.smoke:
            best_acc = test_acc
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{args.name}_best.pt")

    print(f"완료. 최고 테스트 정확도: {max(best_acc, test_acc):.4f}")
    if not args.smoke:
        print(f"체크포인트: checkpoints/{args.name}_best.pt")
        print("대시보드 반영: git add docs/metrics && git commit && git push")


if __name__ == "__main__":
    # Windows에서 DataLoader 워커가 프로세스를 새로 띄울 때 이 파일을 다시
    # import 하므로, 실행 코드는 반드시 이 가드 안에 있어야 한다.
    main()
