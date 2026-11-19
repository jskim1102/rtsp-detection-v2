"""추론 엔진의 가벼운 메타데이터 registry.

이 모듈은 FastAPI 메인 프로세스에서도 import된다. 따라서 torch, ultralytics,
rfdetr 같은 런타임 패키지를 import하지 않고 순수 메타데이터만 보관한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    presets: tuple[str, ...]
    default: str
    closed_set_coco: bool
    prompt: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ENGINES: dict[str, EngineSpec] = {
    "yolo26": EngineSpec(
        id="yolo26",
        label="YOLO26",
        presets=(
            "yolo26n.pt",
            "yolo26s.pt",
            "yolo26m.pt",
            "yolo26l.pt",
            "yolo26x.pt",
        ),
        default="yolo26n.pt",
        closed_set_coco=True,
        prompt=False,
    ),
    "rtdetr": EngineSpec(
        id="rtdetr",
        label="RT-DETR",
        presets=("rtdetr-l.pt", "rtdetr-x.pt"),
        default="rtdetr-l.pt",
        closed_set_coco=True,
        prompt=False,
    ),
    "rfdetr": EngineSpec(
        id="rfdetr",
        label="RF-DETR",
        presets=("rf-detr-base.pth",),
        default="rf-detr-base.pth",
        closed_set_coco=True,
        prompt=False,
    ),
    "yolo_world": EngineSpec(
        id="yolo_world",
        label="YOLO-World",
        presets=(
            "yolov8s-worldv2.pt",
            "yolov8m-worldv2.pt",
            "yolov8l-worldv2.pt",
            "yolov8x-worldv2.pt",
        ),
        default="yolov8s-worldv2.pt",
        closed_set_coco=False,
        prompt=True,
    ),
}

DEFAULT_ENGINE = "yolo26"

_ENGINE_BY_MODEL = {
    model_name: engine_id
    for engine_id, spec in ENGINES.items()
    for model_name in spec.presets
}


def engine_of(model_name: str) -> str | None:
    """허용된 preset 이름의 소유 엔진을 반환한다."""
    return _ENGINE_BY_MODEL.get(model_name)


def is_preset(name: str) -> bool:
    """모든 엔진 preset의 합집합 allowlist."""
    return name in _ENGINE_BY_MODEL
