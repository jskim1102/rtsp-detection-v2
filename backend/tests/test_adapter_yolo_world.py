"""YOLO-World prompt adapter contract without weights or GPU."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from app.inference import adapters


def test_yolo_world_info_log_reaches_fresh_process_stdout_once() -> None:
    probe = """
from app import config
from app.inference.adapters.yolo_world import logger

config.setup_logging()
config.setup_logging()
logger.info("YOLO-World re-embedding: classes=2 elapsed=12.34ms")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [
        line for line in completed.stdout.splitlines()
        if "YOLO-World re-embedding" in line
    ]
    assert len(lines) == 1
    assert "rtsp-streaming.inference.adapters.yolo_world" in lines[0]


def _install_fake_ultralytics(monkeypatch):
    fake_module = ModuleType("ultralytics")

    class FakeYOLO:
        instances: list["FakeYOLO"] = []

        def __init__(self, path: str) -> None:
            self.path = path
            self.names: dict[int, str] | list[str] = {
                0: "person",
                1: "bicycle",
                2: "car",
            }
            self.device = ""
            self.set_classes_calls: list[list[str]] = []
            self.__class__.instances.append(self)

        def to(self, device: str) -> None:
            self.device = device

        def set_classes(self, names: list[str]) -> None:
            self.set_classes_calls.append(list(names))
            self.names = list(names)

        def __call__(self, frames, **kwargs):
            return [type("Result", (), {"boxes": None})() for _frame in frames]

    class FakeRTDETR(FakeYOLO):
        pass

    fake_module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    fake_module.RTDETR = FakeRTDETR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    sys.modules.pop("app.inference.adapters.ultralytics_engine", None)
    sys.modules.pop("app.inference.adapters.yolo_world", None)
    return FakeYOLO


def test_yolo_world_empty_prompt_is_noop_and_nonempty_reuses_session(
    monkeypatch,
    caplog,
) -> None:
    fake_yolo = _install_fake_ultralytics(monkeypatch)
    with caplog.at_level(logging.INFO):
        session = adapters.create(
            "yolo_world",
            "yolov8s-worldv2.pt",
            "cuda:0",
        )
        session_id = id(session)
        model = fake_yolo.instances[-1]
        model_id = id(model)

        session.set_classes([])
        assert model.set_classes_calls == []
        assert session.names == {0: "person", 1: "bicycle", 2: "car"}

        session.set_classes(["helmet", "forklift"])

    assert id(session) == session_id
    assert id(fake_yolo.instances[-1]) == model_id
    assert model.set_classes_calls == [["helmet", "forklift"]]
    assert session.names == {0: "helmet", 1: "forklift"}
    assert any(
        "re-embedding" in record.message and "ms" in record.message
        for record in caplog.records
    )


def test_yolo_world_reset_restores_default_without_set_classes_call(monkeypatch) -> None:
    fake_yolo = _install_fake_ultralytics(monkeypatch)
    session = adapters.create("yolo_world", "yolov8s-worldv2.pt", "cpu")
    model = fake_yolo.instances[-1]

    session.set_classes(["helmet"])
    session.reset_classes()  # type: ignore[attr-defined]

    assert model.set_classes_calls == [["helmet"]]
    assert session.names == {0: "person", 1: "bicycle", 2: "car"}
    assert model.names == {0: "person", 1: "bicycle", 2: "car"}


def test_adapter_registry_declares_imgsz_capability() -> None:
    assert adapters.implemented_engines() == frozenset(
        {"yolo26", "rtdetr", "rfdetr", "yolo_world"}
    )
    assert adapters.honors_imgsz("yolo26") is True
    assert adapters.honors_imgsz("rtdetr") is True
    assert adapters.honors_imgsz("yolo_world") is True
    assert adapters.honors_imgsz("rfdetr") is False
