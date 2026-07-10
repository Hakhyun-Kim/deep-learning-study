"""학습 지표를 docs/metrics/ 에 JSON으로 기록하는 로거.

에폭마다 log()를 부르면 파일이 갱신되고, git push 하면 GitHub Pages 대시보드
(https://hakhyun-kim.github.io/deep-learning-study/)에 그래프가 자동으로 나타난다.

사용법:
    from metrics_logger import MetricsLogger

    logger = MetricsLogger("exp01_baseline", config={"lr": 1e-3, "batch_size": 32})
    for epoch in range(num_epochs):
        ...학습...
        logger.log(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)

실험 삭제:
    python src/metrics_logger.py remove exp01_baseline
실험 목록:
    python src/metrics_logger.py list
"""
import datetime
import json
import sys
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parent.parent / "docs" / "metrics"
MANIFEST = METRICS_DIR / "manifest.json"


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"experiments": []}


def _save_manifest(manifest: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


class MetricsLogger:
    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.path = METRICS_DIR / f"{name}.json"
        self.data = {
            "name": name,
            "config": config or {},
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "epochs": [],
            "metrics": {},
        }
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        manifest = _load_manifest()
        if name not in manifest["experiments"]:
            manifest["experiments"].append(name)
            _save_manifest(manifest)
        self._save()

    def log(self, **metrics: float) -> None:
        """에폭 하나가 끝날 때마다 호출. 예: logger.log(train_loss=0.5, test_acc=0.71)"""
        self.data["epochs"].append(len(self.data["epochs"]) + 1)
        for key, value in metrics.items():
            self.data["metrics"].setdefault(key, []).append(round(float(value), 5))
        self._save()

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8"
        )


def remove(name: str) -> None:
    manifest = _load_manifest()
    if name in manifest["experiments"]:
        manifest["experiments"].remove(name)
        _save_manifest(manifest)
    target = METRICS_DIR / f"{name}.json"
    if target.exists():
        target.unlink()
        print(f"삭제됨: {name}")
    else:
        print(f"파일 없음: {target}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "remove":
        remove(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "list":
        for exp in _load_manifest()["experiments"]:
            print(exp)
    else:
        print(__doc__)
