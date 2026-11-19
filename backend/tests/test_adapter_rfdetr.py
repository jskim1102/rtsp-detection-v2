"""RF-DETR adapter normalization contract without GPU or weight downloads."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from app import config
from app.inference import adapters
from app.inference.worker import Detection


class _FakeInnerModel:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device):
        self.moves.append(str(device))
        return self


class _FakeDetections:
    def __init__(self, *, empty: bool = False) -> None:
        if empty:
            self.xyxy = np.empty((0, 4), dtype=float)
            self.confidence = np.empty((0,), dtype=float)
            self.class_id = np.empty((0,), dtype=int)
            self.data = {}
            return
        self.xyxy = np.asarray(
            [[1.4, 2.6, 30.5, 40.49], [4.6, 5.4, 20.1, 21.9]]
        )
        self.confidence = np.asarray([0.91, 0.72])
        self.class_id = np.asarray([0, 2])
        self.data = {"class_name": np.asarray(["person", ""], dtype=object)}


def _install_fake_rfdetr(monkeypatch):
    fake_module = ModuleType("rfdetr")

    class FakeRFDETRBase:
        instances: list["FakeRFDETRBase"] = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.class_names = ["person", "bicycle", "car"]
            self.model = SimpleNamespace(
                device=kwargs["device"], model=_FakeInnerModel()
            )
            self.predict_call = None
            self.__class__.instances.append(self)

        def predict(self, images, **kwargs):
            self.predict_call = (images, kwargs)
            return [
                _FakeDetections(empty=index == 1)
                for index, _image in enumerate(images)
            ]

    fake_module.RFDETRBase = FakeRFDETRBase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rfdetr", fake_module)
    sys.modules.pop("app.inference.adapters.rf_detr", None)
    return FakeRFDETRBase


def test_rfdetr_adapter_batches_and_normalizes_results(monkeypatch) -> None:
    fake_rfdetr = _install_fake_rfdetr(monkeypatch)
    first = np.zeros((12, 16, 3), dtype=np.uint8)
    first[0, 0] = [10, 20, 30]
    frames = [first, np.zeros((8, 10, 3), dtype=np.uint8)]

    session = adapters.create("rfdetr", "rf-detr-base.pth", "cuda:0")
    output = session.infer(frames, conf=0.5, imgsz=416)

    model = fake_rfdetr.instances[-1]
    assert model.kwargs == {
        "pretrain_weights": str(config.YOLO_WEIGHTS_DIR / "rf-detr-base.pth"),
        "device": "cuda:0",
    }
    images, predict_kwargs = model.predict_call
    assert images[0][0, 0].tolist() == [30, 20, 10]
    assert predict_kwargs == {"threshold": 0.5, "include_source_image": False}
    assert output == [
        [
            Detection(0, "person", pytest.approx(0.91), (1, 3, 30, 40), "rf-detr-base.pth"),
            Detection(1, "bicycle", pytest.approx(0.72), (5, 5, 20, 22), "rf-detr-base.pth"),
        ],
        [],
    ]

    session.to("cpu")
    assert str(model.model.device) == "cpu"
    assert model.model.model.moves == ["cpu"]
    session.close()


def test_rfdetr_adapter_rejects_result_length_mismatch(monkeypatch) -> None:
    fake_rfdetr = _install_fake_rfdetr(monkeypatch)
    session = adapters.create("rfdetr", "rf-detr-base.pth", "cpu")
    fake_rfdetr.instances[-1].predict = lambda images, **kwargs: []

    with pytest.raises(RuntimeError, match="결과 길이"):
        session.infer([np.zeros((8, 8, 3), dtype=np.uint8)], conf=0.5, imgsz=320)


def test_rfdetr_sparse_coco_ids_are_normalized_for_shared_filters(monkeypatch) -> None:
    fake_rfdetr = _install_fake_rfdetr(monkeypatch)
    session = adapters.create("rfdetr", "rf-detr-base.pth", "cpu")
    fake_rfdetr.instances[-1].predict = lambda images, **kwargs: [
        SimpleNamespace(
            xyxy=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]),
            confidence=np.asarray([0.9, 0.8]),
            class_id=np.asarray([1, 62]),
            data={"class_name": np.asarray(["person", "chair"])},
        )
    ]

    output = session.infer(
        [np.zeros((8, 8, 3), dtype=np.uint8)], conf=0.5, imgsz=320
    )

    assert [(item.class_id, item.class_name) for item in output[0]] == [
        (0, "person"),
        (56, "chair"),
    ]


def test_rfdetr_missing_class_name_uses_sparse_coco_id_map(monkeypatch) -> None:
    fake_rfdetr = _install_fake_rfdetr(monkeypatch)
    session = adapters.create("rfdetr", "rf-detr-base.pth", "cpu")
    fake_rfdetr.instances[-1].predict = lambda images, **kwargs: [
        SimpleNamespace(
            xyxy=np.asarray([[1, 2, 3, 4]]),
            confidence=np.asarray([0.9]),
            class_id=np.asarray([18]),
            data={"class_name": np.asarray([""])},
        )
    ]

    output = session.infer(
        [np.zeros((8, 8, 3), dtype=np.uint8)], conf=0.5, imgsz=320
    )

    assert [(item.class_id, item.class_name) for item in output[0]] == [
        (16, "dog"),
    ]
