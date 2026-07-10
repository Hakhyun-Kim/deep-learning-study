"""Stanford 40 Actions 데이터셋.

data/ImageSplits/<클래스명>_train.txt / _test.txt 에 적힌 공식 split을 읽어서
(이미지 텐서, 라벨 번호) 쌍을 돌려주는 PyTorch Dataset.

- train: 클래스당 100장 = 총 4,000장
- test:  나머지 5,532장
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ImageNet 사전학습 모델을 쓸 때는 사전학습 때와 같은 평균/표준편차로
# 정규화해야 모델이 기대하는 입력 분포와 맞는다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_class_names() -> list[str]:
    """actions.txt에서 40개 클래스 이름을 읽는다 (첫 줄은 헤더라서 건너뜀)."""
    lines = (DATA_DIR / "ImageSplits" / "actions.txt").read_text().splitlines()
    return [line.split()[0] for line in lines[1:] if line.strip()]


def build_transform(train: bool) -> transforms.Compose:
    """이미지 → 224x224 텐서 변환.

    학습용은 무작위 크롭 + 좌우반전으로 매 에폭 조금씩 다른 이미지를 보여줘서
    (데이터 증강) 과적합을 줄인다. 평가용은 항상 같은 결과가 나오도록
    결정적인 변환(가운데 크롭)만 쓴다.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Stanford40(Dataset):
    def __init__(self, split: str, transform=None):
        assert split in ("train", "test"), f"split은 train/test 중 하나: {split}"
        self.classes = load_class_names()
        self.transform = transform
        # (이미지 경로, 라벨 번호) 목록. 라벨 번호는 actions.txt 순서(0~39).
        self.samples: list[tuple[Path, int]] = []
        for label, name in enumerate(self.classes):
            split_file = DATA_DIR / "ImageSplits" / f"{name}_{split}.txt"
            for filename in split_file.read_text().split():
                self.samples.append((DATA_DIR / "JPEGImages" / filename, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        # 흑백 이미지가 섞여 있어서 전부 RGB 3채널로 통일한다.
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # 간단한 동작 확인: python src/dataset.py
    for split in ("train", "test"):
        ds = Stanford40(split)
        print(f"{split}: {len(ds)}장, 클래스 {len(ds.classes)}개")
    image, label = Stanford40("train", transform=build_transform(train=True))[0]
    print(f"샘플 텐서 크기: {tuple(image.shape)}, 라벨: {label}")
