"""학습 지표를 docs/metrics/ 에 JSON으로 기록하는 로거.

에폭마다 log()를 부르면 파일이 갱신되고, git push 하면 GitHub Pages 대시보드
(https://hakhyun-kim.github.io/deep-learning-study/)에 그래프가 자동으로 나타난다.

사용법:
    from metrics_logger import MetricsLogger

    logger = MetricsLogger("exp01_baseline", config={"lr": 1e-3, "batch_size": 32})
    for epoch in range(num_epochs):
        ...학습...
        logger.log(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)

에폭마다 docs/preview.svg (README에 들어가는 test_acc 곡선 미리보기)도 함께 갱신된다.

실험 삭제:
    python src/metrics_logger.py remove exp01_baseline
실험 목록:
    python src/metrics_logger.py list
미리보기만 다시 그리기:
    python src/metrics_logger.py preview
"""
import datetime
import json
import sys
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parent.parent / "docs" / "metrics"
MANIFEST = METRICS_DIR / "manifest.json"
PREVIEW = METRICS_DIR.parent / "preview.svg"

# README 미리보기 곡선 색 (실험 순서대로 배정)
_PALETTE = ["#6366F1", "#10B981", "#F59E0B", "#EF4444",
            "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"experiments": []}


def _save_manifest(manifest: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def render_preview() -> None:
    """모든 실험의 test_acc 곡선을 docs/preview.svg 로 그린다 (README 임베드용).

    matplotlib 없이 SVG를 직접 만든다. 회색 텍스트를 써서 GitHub의
    라이트/다크 모드 어디서도 읽히게 한다.
    """
    manifest = _load_manifest()
    series: list[tuple[str, list[float]]] = []
    for name in manifest["experiments"]:
        path = METRICS_DIR / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        acc = [v * 100 for v in data["metrics"].get("test_acc", [])]
        if acc:
            series.append((name, acc))

    width, height = 720, 360
    left, right, top, bottom = 52, 704, 18, 316  # 플롯 영역 경계
    gray = "#8b8b8b"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">'
    ]

    def y_of(acc_percent: float) -> float:
        return bottom - acc_percent / 100 * (bottom - top)

    for v in range(0, 101, 20):
        y = y_of(v)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="{gray}" stroke-opacity="0.25"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="12" fill="{gray}">{v}%</text>')

    # 목표선 80%
    ty = y_of(80)
    parts.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{right}" y2="{ty:.1f}" '
                 f'stroke="#F43F5E" stroke-dasharray="6 4" stroke-opacity="0.8"/>')
    parts.append(f'<text x="{right}" y="{ty - 6:.1f}" text-anchor="end" '
                 f'font-size="12" fill="#F43F5E">목표 80%</text>')

    if not series:
        parts.append(f'<text x="{(left + right) / 2}" y="{(top + bottom) / 2}" '
                     f'text-anchor="middle" font-size="14" fill="{gray}">아직 실험 기록이 없습니다</text>')
    else:
        max_epochs = max(len(acc) for _, acc in series)

        def x_of(epoch: int) -> float:
            if max_epochs == 1:
                return (left + right) / 2
            return left + (epoch - 1) / (max_epochs - 1) * (right - left)

        tick_step = max(1, (max_epochs + 11) // 12)
        for e in range(1, max_epochs + 1, tick_step):
            parts.append(f'<text x="{x_of(e):.1f}" y="{bottom + 20}" text-anchor="middle" '
                         f'font-size="12" fill="{gray}">{e}</text>')
        parts.append(f'<text x="{right}" y="{bottom + 38}" text-anchor="end" '
                     f'font-size="12" fill="{gray}">epoch</text>')

        for i, (name, acc) in enumerate(series):
            color = _PALETTE[i % len(_PALETTE)]
            points = " ".join(f"{x_of(e):.1f},{y_of(v):.1f}" for e, v in enumerate(acc, 1))
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                         f'stroke-width="2" stroke-linejoin="round"/>')
            parts.append(f'<circle cx="{x_of(len(acc)):.1f}" cy="{y_of(acc[-1]):.1f}" '
                         f'r="3.5" fill="{color}"/>')
            ly = top + 12 + i * 20
            parts.append(f'<rect x="{left + 12}" y="{ly - 9}" width="10" height="10" fill="{color}"/>')
            parts.append(f'<text x="{left + 28}" y="{ly}" font-size="12" fill="{gray}">'
                         f'{name} · 최고 {max(acc):.1f}%</text>')

    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
                 f'stroke="{gray}" stroke-opacity="0.6"/>')
    parts.append("</svg>")
    PREVIEW.write_text("\n".join(parts), encoding="utf-8")


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
        render_preview()


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
    render_preview()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "remove":
        remove(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "list":
        for exp in _load_manifest()["experiments"]:
            print(exp)
    elif len(sys.argv) >= 2 and sys.argv[1] == "preview":
        render_preview()
        print(f"생성됨: {PREVIEW}")
    else:
        print(__doc__)
