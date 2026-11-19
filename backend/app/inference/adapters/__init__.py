"""DetectorAdapter factory.

무거운 엔진 모듈 import는 create() 호출 안에서만 일어난다. 정상 애플리케이션에서는
worker subprocess만 이 함수를 호출한다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.inference import engines
from app.inference.adapters.base import DetectorSession


def _create_yolo26(model_name: str, device: str) -> DetectorSession:
    from app.inference.adapters.ultralytics_engine import UltralyticsSession

    return UltralyticsSession("yolo26", model_name, device)


def _create_rtdetr(model_name: str, device: str) -> DetectorSession:
    from app.inference.adapters.ultralytics_engine import UltralyticsSession

    return UltralyticsSession("rtdetr", model_name, device)


def _create_rfdetr(model_name: str, device: str) -> DetectorSession:
    from app.inference.adapters.rf_detr import RFDETRSession

    return RFDETRSession(model_name, device)


def _create_yolo_world(model_name: str, device: str) -> DetectorSession:
    from app.inference.adapters.yolo_world import YOLOWorldSession

    return YOLOWorldSession(model_name, device)


@dataclass(frozen=True)
class _AdapterRegistration:
    factory: Callable[[str, str], DetectorSession]
    honors_imgsz: bool


_ADAPTERS: dict[str, _AdapterRegistration] = {
    "yolo26": _AdapterRegistration(_create_yolo26, honors_imgsz=True),
    "rtdetr": _AdapterRegistration(_create_rtdetr, honors_imgsz=True),
    # RF-DETR 1.9.3 preprocesses at its configured 560px resolution.
    "rfdetr": _AdapterRegistration(_create_rfdetr, honors_imgsz=False),
    "yolo_world": _AdapterRegistration(_create_yolo_world, honors_imgsz=True),
}


def implemented_engines() -> frozenset[str]:
    """실제로 세션을 만들 수 있는 엔진 ID 집합."""
    return frozenset(_ADAPTERS)


def is_implemented(engine: str) -> bool:
    return engine in _ADAPTERS


def honors_imgsz(engine: str) -> bool:
    """엔진이 요청별 imgsz를 실제 전처리에 반영하는지 반환한다."""
    registration = _ADAPTERS.get(engine)
    if registration is None:
        raise ValueError(f"아직 구현되지 않은 추론 엔진: {engine!r}")
    return registration.honors_imgsz


def create(engine: str, model_name: str, device: str) -> DetectorSession:
    spec = engines.ENGINES.get(engine)
    if spec is None:
        raise ValueError(f"알 수 없는 추론 엔진: {engine!r}")
    if model_name not in spec.presets:
        raise ValueError(
            f"엔진 {engine!r}이 소유하지 않는 preset 모델: {model_name!r}"
        )

    registration = _ADAPTERS.get(engine)
    if registration is None:
        raise ValueError(f"아직 구현되지 않은 추론 엔진: {engine!r}")
    return registration.factory(model_name, device)


__all__ = [
    "DetectorSession",
    "create",
    "honors_imgsz",
    "implemented_engines",
    "is_implemented",
]
