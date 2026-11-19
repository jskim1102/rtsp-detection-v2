"""rfdetr 1.9.3 기반 DetectorSession."""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from app.inference.models_dir import COCO_CLASS_NAMES, resolve_model_path
from app.inference.worker import Detection

_COCO_ID_BY_NAME = {
    class_name: class_id
    for class_id, class_name in enumerate(COCO_CLASS_NAMES)
}
_COCO_SPARSE_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)
_COCO_NAME_BY_SPARSE_ID = dict(zip(_COCO_SPARSE_IDS, COCO_CLASS_NAMES))


def _normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(value) for index, value in enumerate(names)}
    return {
        index: class_name for index, class_name in enumerate(COCO_CLASS_NAMES)
    }


class RFDETRSession:
    """RF-DETR Base를 한 번 로드하고 package batch API를 재사용한다."""

    def __init__(self, model_name: str, device: str) -> None:
        from rfdetr import RFDETRBase

        self._model_name = model_name
        self._device = device
        self._model: Any | None = RFDETRBase(
            pretrain_weights=resolve_model_path(model_name),
            device=device,
        )
        self.names = _normalize_names(getattr(self._model, "class_names", None))

    def infer(
        self,
        frames: list[np.ndarray],
        *,
        conf: float,
        imgsz: int,
    ) -> list[list[Detection]]:
        del imgsz  # package shape 제약과 worker adaptive stage가 충돌하므로 원본 크기 계약을 쓴다.
        if self._model is None:
            raise RuntimeError("종료된 detector session")
        # OpenCV capture는 BGR, RF-DETR의 NumPy 입력 계약은 RGB다.
        rgb_frames = [np.ascontiguousarray(frame[..., ::-1]) for frame in frames]
        raw = self._model.predict(
            rgb_frames,
            threshold=conf,
            include_source_image=False,
        )
        predictions = raw if isinstance(raw, list) else [raw]
        if len(predictions) != len(frames):
            raise RuntimeError(
                "DetectorAdapter 결과 길이 불일치: "
                f"frames={len(frames)} results={len(predictions)}"
            )
        return [self._parse_result(prediction) for prediction in predictions]

    def _parse_result(self, result: Any) -> list[Detection]:
        xyxy = np.asarray(getattr(result, "xyxy", ()))
        confidence = np.asarray(getattr(result, "confidence", ()))
        class_ids = np.asarray(getattr(result, "class_id", ()))
        data = getattr(result, "data", {})
        class_names = data.get("class_name") if isinstance(data, Mapping) else None
        detections: list[Detection] = []
        for index in range(len(xyxy)):
            raw_class_id = int(class_ids[index])
            class_name = ""
            if class_names is not None and index < len(class_names):
                class_name = str(class_names[index])
            if not class_name:
                class_name = _COCO_NAME_BY_SPARSE_ID.get(
                    raw_class_id,
                    str(raw_class_id),
                )
            # RF-DETR pretrained COCO 결과는 sparse category ID(1~90)를 쓴다.
            # 공통 UI/API의 연속 0~79 클래스 필터와 맞추기 위해 이름으로 정규화한다.
            class_id = _COCO_ID_BY_NAME.get(class_name, raw_class_id)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidence[index]),
                    xyxy=tuple(
                        int(round(float(value))) for value in xyxy[index]
                    ),
                    model=self._model_name,
                )
            )
        return detections

    def to(self, device: str) -> None:
        if self._model is None:
            raise RuntimeError("종료된 detector session")
        import torch

        target = torch.device(device)
        context = self._model.model
        context.device = target
        if getattr(context, "model", None) is not None:
            context.model = context.model.to(target)
        if getattr(context, "inference_model", None) is not None:
            context.inference_model = context.inference_model.to(target)
        self._device = str(target)

    def set_classes(self, names: list[str]) -> None:
        return None

    def reset_classes(self) -> None:
        return None

    def close(self) -> None:
        model, self._model = self._model, None
        if model is None:
            return
        del model
        gc.collect()
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
