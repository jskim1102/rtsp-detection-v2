"""ultralytics 기반 DetectorSession.

YOLO26의 기존 로드·호출·파싱을 클래스 경계로만 감쌌다. OOM 복구와 batch 분할은
계속 worker.py가 소유한다.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pickle import UnpicklingError
from typing import Any
from zipfile import is_zipfile

import numpy as np

from app.inference.models_dir import resolve_model_path
from app.inference.worker import Detection, _is_cuda_oom


def _load_preset_model(model_class, model_name: str):
    path = resolve_model_path(model_name)
    try:
        return model_class(path)
    except (EOFError, UnpicklingError, RuntimeError) as exc:
        # 공식 .pt는 ZIP이다. 다운로드가 끊겨 ZIP을 열 수 없을 때만 다시 받는다.
        if _is_cuda_oom(exc) or not os.path.isfile(path) or is_zipfile(path):
            raise
        logging.getLogger(__name__).warning("Retrying incomplete preset download: %s", model_name)
        os.unlink(path)
    return model_class(path)


def _normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _parse_result(result: Any, names: object, model_name: str) -> list[Detection]:
    """ultralytics Results → 부모와 동일한 Detection."""
    detections: list[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    boxes = result.boxes
    xyxy_arr = boxes.xyxy.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    for index in range(len(boxes)):
        x1, y1, x2, y2 = xyxy_arr[index].tolist()
        class_id = int(cls_arr[index])
        if hasattr(names, "get"):
            class_name = str(names.get(class_id, str(class_id)))
        else:
            class_name = (
                str(names[class_id]) if class_id < len(names) else str(class_id)
            )
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(conf_arr[index]),
                xyxy=(x1, y1, x2, y2),
                model=model_name,
            )
        )
    return detections


class UltralyticsSession:
    def __init__(self, engine: str, model_name: str, device: str) -> None:
        from ultralytics import RTDETR, YOLO

        model_class = {
            "yolo26": YOLO,
            "rtdetr": RTDETR,
            "yolo_world": YOLO,
        }.get(engine)
        if model_class is None:
            raise ValueError(f"지원하지 않는 ultralytics 엔진: {engine!r}")

        self._model_name = model_name
        self._device = device
        self._model: Any | None = _load_preset_model(model_class, model_name)
        self._model.to(device)
        self.names = _normalize_names(self._model.names)

    def infer(
        self,
        frames: list[np.ndarray],
        *,
        conf: float,
        imgsz: int,
    ) -> list[list[Detection]]:
        if self._model is None:
            raise RuntimeError("종료된 detector session")
        predictions = self._model(
            frames,
            conf=conf,
            imgsz=imgsz,
            device=self._device,
            verbose=False,
        )
        if len(predictions) != len(frames):
            raise RuntimeError(
                "DetectorAdapter 결과 길이 불일치: "
                f"frames={len(frames)} results={len(predictions)}"
            )
        return [
            _parse_result(prediction, self._model.names, self._model_name)
            for prediction in predictions
        ]

    def to(self, device: str) -> None:
        if self._model is None:
            raise RuntimeError("종료된 detector session")
        self._model.to(device)
        self._device = device

    def set_classes(self, names: list[str]) -> None:
        # YOLO26은 closed-set이므로 prompt control을 무시한다.
        return None

    def reset_classes(self) -> None:
        return None

    def close(self) -> None:
        model, self._model = self._model, None
        close = getattr(model, "close", None)
        if callable(close):
            close()
