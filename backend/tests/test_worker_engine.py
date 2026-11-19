"""worker 엔진 lifecycle과 control-message 경로."""

from __future__ import annotations

import queue
import sys
from types import ModuleType

import numpy as np
import pytest

from app import config
from app.inference import adapters
from app.inference.worker import (
    ControlMessage,
    FrameRequest,
    InferenceWorker,
    _ModelLane,
    _model_worker_main,
)


class _FakeLane:
    def __init__(self) -> None:
        self.stops = 0
        self.messages: list[ControlMessage] = []
        self.initial_control: ControlMessage | None = None

    def stop(self) -> None:
        self.stops += 1

    def send_control(self, message: ControlMessage) -> bool:
        self.messages.append(message)
        return True

    def set_initial_control(self, message: ControlMessage | None) -> None:
        self.initial_control = message


class _StopEvent:
    def __init__(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        return self.value

    def set(self) -> None:
        self.value = True


class _StartProcess:
    pid = 123

    def __init__(self) -> None:
        self.alive = False

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout=None) -> None:
        self.alive = False

    def terminate(self) -> None:
        self.alive = False


class _StartContext:
    @staticmethod
    def Queue(maxsize: int):
        return queue.Queue(maxsize=maxsize)

    @staticmethod
    def Event():
        return _StopEvent()

    @staticmethod
    def Process(**kwargs):
        return _StartProcess()


class _FakeSession:
    names = {0: "person"}

    def __init__(self) -> None:
        self.classes: list[list[str]] = []
        self.infer_calls = 0
        self.closed = False
        self.resets = 0

    def infer(self, frames, *, conf: float, imgsz: int):
        self.infer_calls += 1
        return [[] for _frame in frames]

    def to(self, device: str) -> None:
        return None

    def set_classes(self, names: list[str]) -> None:
        self.classes.append(names)

    def reset_classes(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


def test_model_worker_initializes_logging_once(monkeypatch) -> None:
    setup_calls: list[None] = []
    monkeypatch.setattr(config, "setup_logging", lambda: setup_calls.append(None))
    monkeypatch.setattr(adapters, "create", lambda *args: _FakeSession())
    fake_torch = ModuleType("torch")
    fake_torch.cuda = type(  # type: ignore[attr-defined]
        "Cuda",
        (),
        {"is_available": staticmethod(lambda: False), "empty_cache": staticmethod(lambda: None)},
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    in_q: queue.Queue = queue.Queue()
    in_q.put(None)

    _model_worker_main(
        "yolo26",
        "yolo26n.pt",
        in_q,
        queue.Queue(),
        _StopEvent(),
        "cpu",
        1,
        0.0,
        320,
    )

    assert setup_calls == [None]


def test_engine_switch_stops_all_lanes_and_selects_new_default() -> None:
    worker = InferenceWorker()
    lane = _FakeLane()
    worker._lanes = {"yolo26n.pt": lane}  # type: ignore[assignment]
    worker._desired_models = {"yolo26n.pt"}

    worker.set_engine("rtdetr")

    assert lane.stops == 1
    assert worker.get_status()["engine"] == "rtdetr"
    assert worker.get_status()["model"] == "rtdetr-l.pt"
    assert worker.get_pool_status()["models"] == []

    worker.set_engine("rtdetr")
    assert lane.stops == 1


def test_lane_and_pool_control_delivery_is_picklable() -> None:
    message = ControlMessage(kind="set_classes", payload=("person", "car"))
    lane = object.__new__(_ModelLane)
    lane.in_q = queue.Queue()

    assert lane.send_control(message) is True
    assert lane.in_q.get_nowait() == message

    worker = InferenceWorker()
    fake_lane = _FakeLane()
    worker._lanes = {"yolo26n.pt": fake_lane}  # type: ignore[assignment]
    assert worker.send_control(message) == 1
    assert fake_lane.messages == [message]


def test_prompt_is_retained_for_future_lanes_and_clear_resets_live_lanes() -> None:
    worker = InferenceWorker(engine="yolo_world")

    assert worker.set_prompt(["person", "car"]) == 0
    future = worker._make_lane("yolov8s-worldv2.pt")
    assert future._initial_control == ControlMessage(  # type: ignore[attr-defined]
        "set_classes",
        ("person", "car"),
    )

    live = _FakeLane()
    worker._lanes = {"yolov8s-worldv2.pt": live}  # type: ignore[assignment]
    assert worker.set_prompt([]) == 1
    assert live.initial_control is None
    assert live.messages == [ControlMessage("reset_classes", ())]


def test_stopped_lane_does_not_report_control_delivery() -> None:
    lane = object.__new__(_ModelLane)
    lane.in_q = None

    assert lane.send_control(ControlMessage("set_classes", ("person",))) is False


def test_lane_start_injects_prompt_before_frames_and_stop_clears_delivery_queue() -> None:
    prompt = ControlMessage("set_classes", ("helmet",))
    lane = _ModelLane(
        _StartContext(),
        "yolo_world",
        "yolov8s-worldv2.pt",
        "cpu",
        batch_max=2,
        batch_timeout_sec=0.01,
        min_imgsz=320,
        initial_control=prompt,
    )

    lane.start()
    assert lane.in_q.get_nowait() == prompt

    lane.stop()
    assert lane.in_q is None
    assert lane.send_control(prompt) is False


@pytest.mark.parametrize("control_first", [True, False])
def test_subprocess_consumes_control_without_treating_it_as_frame(
    monkeypatch,
    control_first: bool,
) -> None:
    session = _FakeSession()
    created: list[tuple[str, str, str]] = []

    def fake_create(engine: str, model_name: str, device: str):
        created.append((engine, model_name, device))
        return session

    monkeypatch.setattr(adapters, "create", fake_create)
    fake_torch = ModuleType("torch")
    fake_torch.cuda = type(  # type: ignore[attr-defined]
        "Cuda",
        (),
        {"is_available": staticmethod(lambda: False), "empty_cache": staticmethod(lambda: None)},
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    control = ControlMessage("set_classes", ("person", "car"))
    request = FrameRequest(
        source_id="cam",
        frame=np.zeros((12, 16, 3), dtype=np.uint8),
        timestamp=1.0,
        request_id=1,
        imgsz=320,
    )
    if control_first:
        in_q.put(control)
        in_q.put(request)
    else:
        in_q.put(request)
        in_q.put(control)
    in_q.put(None)

    _model_worker_main(
        "yolo26",
        "yolo26n.pt",
        in_q,
        out_q,
        _StopEvent(),
        "cpu",
        8,
        0.1,
        320,
    )

    assert created == [("yolo26", "yolo26n.pt", "cpu")]
    assert session.classes == [["person", "car"]]
    assert session.infer_calls == 1
    assert session.closed is True
    result = out_q.get_nowait()
    assert result.source_id == "cam"
    assert result.detections == []


def test_non_imgsz_engine_oom_skips_futile_resize_ladder(monkeypatch) -> None:
    class OomThenSuccessSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[tuple[int, ...], int]] = []
            self.moves: list[str] = []

        def infer(self, frames, *, conf: float, imgsz: int):
            self.calls.append((frames[0].shape, imgsz))
            if len(self.calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return [[] for _frame in frames]

        def to(self, device: str) -> None:
            self.moves.append(device)

    session = OomThenSuccessSession()
    monkeypatch.setattr(adapters, "create", lambda *args: session)
    fake_torch = ModuleType("torch")
    fake_torch.cuda = type(  # type: ignore[attr-defined]
        "Cuda",
        (),
        {"is_available": staticmethod(lambda: True), "empty_cache": staticmethod(lambda: None)},
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    in_q.put(
        FrameRequest(
            source_id="cam",
            frame=np.zeros((360, 640, 3), dtype=np.uint8),
            timestamp=1.0,
            request_id=1,
            imgsz=640,
        )
    )
    in_q.put(None)

    _model_worker_main(
        "rfdetr",
        "rf-detr-base.pth",
        in_q,
        out_q,
        _StopEvent(),
        "cuda:0",
        1,
        0.0,
        320,
    )

    assert session.calls == [((360, 640, 3), 640), ((360, 640, 3), 640)]
    assert session.moves == ["cpu"]
    result = out_q.get_nowait()
    assert (result.frame_w, result.frame_h) == (640, 360)


def test_worker_logs_failed_cpu_reload_after_cuda_oom(monkeypatch, caplog):
    name = "yolo26n.pt"
    devices = []
    cpu_error = OSError("CPU reload failed")

    def create(engine, model, device):
        assert (engine, model) == ("yolo26", name)
        devices.append(device)
        if device == "cuda:0":
            raise RuntimeError("CUDA out of memory")
        raise cpu_error

    monkeypatch.setattr(adapters, "create", create)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    _model_worker_main("yolo26", name, None, None, _StopEvent(), "cuda:0", 1, 0.0, 320)
    assert devices == ["cuda:0", "cpu"]
    assert any(
        record.getMessage() == f"Model load CPU fallback failed: {name}"
        and record.exc_info[1] is cpu_error
        for record in caplog.records
    )
