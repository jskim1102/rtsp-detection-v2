"""ultralytics YOLO-World prompt adapter."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from app.inference.adapters.ultralytics_engine import (
    UltralyticsSession,
    _normalize_names,
)

logger = logging.getLogger("rtsp-streaming.inference.adapters.yolo_world")
_MISSING = object()


def _copy_names(names: object) -> object:
    if isinstance(names, Mapping):
        return dict(names)
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return list(names)
    return names


class YOLOWorldSession(UltralyticsSession):
    """한 모델 세션 안에서 text embedding만 교체한다."""

    def __init__(self, model_name: str, device: str) -> None:
        super().__init__("yolo_world", model_name, device)
        if self._model is None:
            raise RuntimeError("YOLO-World model load failed")
        inner = getattr(self._model, "model", self._model)
        self._default_names = _copy_names(getattr(inner, "names", self._model.names))
        self._default_txt_feats: Any = getattr(inner, "txt_feats", _MISSING)
        layers = getattr(inner, "model", None)
        head = layers[-1] if layers else None
        self._default_nc = getattr(head, "nc", _MISSING)

    def set_classes(self, names: list[str]) -> None:
        if not names:
            return
        if self._model is None:
            raise RuntimeError("종료된 detector session")
        started = time.perf_counter()
        self._model.set_classes(list(names))
        self.names = _normalize_names(self._model.names)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "YOLO-World re-embedding: classes=%d elapsed=%.2fms",
            len(names),
            elapsed_ms,
        )

    def to(self, device: str) -> None:
        super().to(device)
        mover = getattr(self._default_txt_feats, "to", None)
        if callable(mover):
            self._default_txt_feats = mover(device)

    def reset_classes(self) -> None:
        """빈 prompt는 set_classes를 호출하지 않고 로드 시 embedding을 복원한다."""
        if self._model is None:
            raise RuntimeError("종료된 detector session")
        inner = getattr(self._model, "model", self._model)
        if self._default_txt_feats is not _MISSING:
            inner.txt_feats = self._default_txt_feats
        layers = getattr(inner, "model", None)
        if layers and self._default_nc is not _MISSING:
            layers[-1].nc = self._default_nc
        restored_names = _copy_names(self._default_names)
        inner.names = restored_names
        predictor = getattr(self._model, "predictor", None)
        predictor_model = getattr(predictor, "model", None)
        if predictor_model is not None:
            predictor_model.names = _copy_names(self._default_names)
        self.names = _normalize_names(restored_names)
        logger.info("YOLO-World default vocabulary restored: classes=%d", len(self.names))
