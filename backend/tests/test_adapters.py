"""DetectorAdapter 계약과 YOLO26 파싱 동치성."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock
from zipfile import ZipFile

import numpy as np
import pytest

from app import config
from app.inference import adapters
from app.inference.adapters import base
from app.inference.worker import Detection


class _FakeTensor:
    def __init__(self, value) -> None:
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = _FakeTensor([[1.9, 2.1, 30.8, 40.2]])
        self.conf = _FakeTensor([0.81])
        self.cls = _FakeTensor([0])

    def __len__(self) -> int:
        return 1


class _FakeResult:
    def __init__(self, *, empty: bool = False) -> None:
        self.boxes = None if empty else _FakeBoxes()


def _install_fake_ultralytics(monkeypatch, *, short_result: bool = False):
    fake_module = ModuleType("ultralytics")

    class FakeYOLO:
        instances: list["FakeYOLO"] = []

        def __init__(self, path: str) -> None:
            self.path = path
            self.names = {0: "person"}
            self.device = ""
            self.call: dict[str, object] | None = None
            self.closed = False
            self.__class__.instances.append(self)

        def to(self, device: str) -> None:
            self.device = device

        def __call__(self, frames, **kwargs):
            self.call = {"frames": frames, **kwargs}
            if short_result:
                return []
            return [_FakeResult(empty=index == 1) for index, _frame in enumerate(frames)]

    class FakeRTDETR(FakeYOLO):
        pass

    fake_module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    fake_module.RTDETR = FakeRTDETR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    sys.modules.pop("app.inference.adapters.ultralytics_engine", None)
    return FakeYOLO, FakeRTDETR


def test_detector_session_protocol_exposes_required_contract() -> None:
    assert hasattr(base, "DetectorSession")
    for name in ("infer", "to", "set_classes", "reset_classes", "close"):
        assert hasattr(base.DetectorSession, name)


def test_yolo26_adapter_preserves_batch_order_and_detection_shape(monkeypatch) -> None:
    fake_yolo, _fake_rtdetr = _install_fake_ultralytics(monkeypatch)
    frames = [
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.zeros((12, 16, 3), dtype=np.uint8),
    ]

    session = adapters.create("yolo26", "yolo26n.pt", "cuda:0")
    output = session.infer(frames, conf=0.25, imgsz=640)

    assert len(output) == len(frames)
    assert output[1] == []
    assert output[0] == [
        Detection(
            class_id=0,
            class_name="person",
            confidence=pytest.approx(0.81),
            xyxy=(1, 2, 30, 40),
            model="yolo26n.pt",
        )
    ]
    model = fake_yolo.instances[-1]
    assert model.path == str(config.YOLO_WEIGHTS_DIR / "yolo26n.pt")
    assert model.device == "cuda:0"
    assert model.call is not None
    assert model.call["frames"] is frames
    assert model.call["conf"] == 0.25
    assert model.call["imgsz"] == 640
    assert model.call["device"] == "cuda:0"
    assert model.call["verbose"] is False

    session.to("cpu")
    assert model.device == "cpu"
    session.set_classes(["ignored"])
    session.close()


def test_adapter_rejects_result_length_mismatch(monkeypatch) -> None:
    _install_fake_ultralytics(monkeypatch, short_result=True)
    session = adapters.create("yolo26", "yolo26n.pt", "cpu")

    with pytest.raises(RuntimeError, match="결과 길이"):
        session.infer([np.zeros((8, 8, 3), dtype=np.uint8)], conf=0.5, imgsz=320)


def test_rtdetr_adapter_uses_shared_ultralytics_path(monkeypatch) -> None:
    _fake_yolo, fake_rtdetr = _install_fake_ultralytics(monkeypatch)

    session = adapters.create("rtdetr", "rtdetr-l.pt", "cpu")
    output = session.infer(
        [np.zeros((8, 8, 3), dtype=np.uint8)], conf=0.4, imgsz=640
    )

    assert len(output) == 1
    assert output[0][0].model == "rtdetr-l.pt"
    assert isinstance(fake_rtdetr.instances[-1], fake_rtdetr)


def test_adapter_registry_is_the_implementation_source_of_truth() -> None:
    assert adapters.implemented_engines() == frozenset(
        {"yolo26", "rtdetr", "rfdetr", "yolo_world"}
    )
    assert adapters.is_implemented("rtdetr") is True
    assert adapters.is_implemented("yolo_world") is True


def test_create_rejects_unknown_or_wrong_engine_model() -> None:
    with pytest.raises(ValueError, match="알 수 없는"):
        adapters.create("unknown", "yolo26n.pt", "cpu")
    with pytest.raises(ValueError, match="소유하지 않는"):
        adapters.create("yolo26", "rtdetr-l.pt", "cpu")


@pytest.mark.parametrize("download_succeeds", [True, False])
def test_adapter_retries_interrupted_preset_once(monkeypatch, tmp_path, download_succeeds):
    _install_fake_ultralytics(monkeypatch)
    monkeypatch.setattr(config, "YOLO_WEIGHTS_DIR", tmp_path)
    weights = tmp_path / "yolo26n.pt"
    weights.write_bytes(b"PK\x03\x04partial")
    sibling = tmp_path / "yolo26s.pt"
    sibling.write_bytes(b"another model")
    cached_at_load = []

    def load(path):
        assert path == str(weights)
        cached_at_load.append(weights.exists())
        if weights.exists():
            raise RuntimeError("PytorchStreamReader failed finding central directory")
        if not download_succeeds:
            raise ConnectionError("download interrupted")
        return Mock(names={})

    monkeypatch.setattr("ultralytics.YOLO", load)
    if download_succeeds:
        adapters.create("yolo26", weights.name, "cpu")
    else:
        with pytest.raises(ConnectionError):
            adapters.create("yolo26", weights.name, "cpu")
    assert cached_at_load == [True, False]
    assert sibling.read_bytes() == b"another model"


@pytest.mark.parametrize("cuda_oom", [False, True])
def test_adapter_preserves_complete_cache_or_cuda_oom(monkeypatch, tmp_path, cuda_oom):
    _install_fake_ultralytics(monkeypatch)
    monkeypatch.setattr(config, "YOLO_WEIGHTS_DIR", tmp_path)
    weights = tmp_path / "yolo26n.pt"
    if cuda_oom:
        weights.write_bytes(b"partial")
        error = "CUDA out of memory"
    else:
        with ZipFile(weights, "w") as checkpoint:
            checkpoint.writestr("model", "data")
        error = "CPU allocator out of memory"
    original = weights.read_bytes()
    load = Mock(side_effect=RuntimeError(error))
    monkeypatch.setattr("ultralytics.YOLO", load)

    with pytest.raises(RuntimeError, match=error):
        adapters.create("yolo26", weights.name, "cuda:0" if cuda_oom else "cpu")
    load.assert_called_once_with(str(weights))
    assert weights.read_bytes() == original
