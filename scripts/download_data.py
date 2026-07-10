"""Stanford 40 Actions 데이터셋 다운로드 + 압축 해제.

사용법:  python scripts/download_data.py
결과:    data/JPEGImages/   (이미지 9,532장)
         data/ImageSplits/  (train/test 목록 txt)
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "Stanford40_JPEGImages.zip": "http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip",
    "Stanford40_ImageSplits.zip": "http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download(url: str, dest: Path) -> None:
    def hook(blocks, block_size, total):
        done = blocks * block_size
        pct = min(100, done * 100 // total) if total > 0 else 0
        sys.stdout.write(f"\r  {dest.name}: {pct}% ({done // (1 << 20)}MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in URLS.items():
        dest = DATA_DIR / name
        if dest.exists():
            print(f"{name} 이미 존재 - 다운로드 건너뜀")
        else:
            print(f"다운로드 중: {url}")
            download(url, dest)
        print(f"압축 해제 중: {name}")
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(DATA_DIR)
    print("\n완료. data/ 내용:")
    for p in sorted(DATA_DIR.iterdir()):
        print("  -", p.name)


if __name__ == "__main__":
    main()
