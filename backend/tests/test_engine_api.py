"""엔진 상태 API와 manager의 엔진별 모델 격리 계약."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.inference_api as inference_api
import app.ipcam as ipcam
from app.inference.worker import ControlMessage, InferenceResult
from app.streaming.manager import StreamManager


BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "configured_model",
    ["rtdetr-l.pt", "yolov8n.pt", "evil.pt"],
)
def test_invalid_yolo_default_model_falls_back_without_startup_crash(
    configured_model: str,
) -> None:
    env = {**os.environ, "YOLO_DEFAULT_MODEL": configured_model}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fastapi.testclient import TestClient; "
            "import app.main; "
            "assert app.main.stream_manager.get_inference_config()['model'] "
            "== 'yolo26n.pt'; "
            "assert TestClient(app.main.app).get('/api/health').status_code == 200",
        ],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_manager_keeps_per_source_models_separate_by_engine() -> None:
    manager = StreamManager()
    manager.set_source_models("cam", ["yolo26x.pt"])
    assert manager.get_source_models("cam") == ["yolo26x.pt"]

    manager.set_engine("rtdetr")
    assert manager.get_source_models("cam") is None
    assert manager.get_inference_config()["model"] == "rtdetr-l.pt"
    manager.set_source_models("cam", ["rtdetr-x.pt"])
    assert manager.get_source_models("cam") == ["rtdetr-x.pt"]

    manager.set_engine("yolo26")
    assert manager.get_source_models("cam") == ["yolo26x.pt"]

    manager.set_engine("rtdetr")
    assert manager.get_source_models("cam") == ["rtdetr-x.pt"]


def test_engine_switch_uses_global_default_without_seeding_source_key() -> None:
    manager = StreamManager()
    manager._captures["cam"] = type("Capture", (), {"is_running": True})()
    manager._latest_results["cam"] = InferenceResult("cam", 1.0)
    manager.set_source_models("cam", ["yolo26x.pt"])

    manager.set_engine("rtdetr")

    assert manager.get_source_models("cam") is None
    assert manager._worker.get_pool_status()["models"] == ["rtdetr-l.pt"]
    assert manager._latest_results == {}


def test_global_model_is_used_for_unconfigured_sources() -> None:
    manager = StreamManager()
    manager._captures["cam"] = type("Capture", (), {"is_running": True})()

    manager.set_inference_model("yolo26s.pt")

    assert manager._worker.get_pool_status()["models"] == ["yolo26s.pt"]


def test_submit_leaves_unconfigured_model_for_worker_fallback(monkeypatch) -> None:
    manager = StreamManager()
    submitted = []
    monkeypatch.setattr(
        manager._worker,
        "submit",
        lambda request: submitted.append(request) or True,
    )
    manager.set_inference_model("yolo26s.pt")

    manager._submit_frame("cam", __import__("numpy").zeros((8, 8, 3), dtype="uint8"))

    assert len(submitted) == 1
    assert submitted[0].model_names is None


def test_engine_switch_preserves_explicit_empty_model_opt_out() -> None:
    manager = StreamManager()
    manager.set_source_models("cam", [])

    manager.set_engine("rtdetr")

    assert manager.get_source_models("cam") == []
    assert manager._worker.get_pool_status()["models"] == []


def test_manager_rejects_unimplemented_engine_without_mutation(monkeypatch) -> None:
    manager = StreamManager()
    before = manager.get_inference_config()
    monkeypatch.setattr(inference_api.adapters, "is_implemented", lambda engine: False)

    with pytest.raises(ValueError, match="구현되지 않은"):
        manager.set_engine("rtdetr")

    assert manager.get_inference_config() == before


def test_manager_prompt_is_trimmed_and_sent_as_control(monkeypatch) -> None:
    manager = StreamManager()
    messages: list[ControlMessage] = []
    monkeypatch.setattr(
        manager._worker,
        "set_prompt",
        lambda names: messages.append(
            ControlMessage(
                kind="set_classes" if names else "reset_classes",
                payload=tuple(names),
            )
        )
        or 1,
    )

    manager.set_prompt(" person, car, person ,, ")

    assert manager.get_inference_config()["prompt"] == "person, car, person ,,"
    assert messages == [
        ControlMessage(kind="set_classes", payload=("person", "car"))
    ]

    manager._latest_results["cam"] = InferenceResult("cam", 1.0)
    manager.set_prompt("person, car, person ,,")
    assert len(messages) == 1
    assert "cam" in manager._latest_results

    manager.set_prompt("")
    assert manager.get_inference_config()["prompt"] == ""
    assert messages[-1] == ControlMessage(kind="reset_classes", payload=())
    assert manager._latest_results == {}


def test_combined_engine_model_put_leaves_camera_on_configured_global_model(
    monkeypatch,
) -> None:
    manager = StreamManager()
    manager._captures["cam"] = type("Capture", (), {"is_running": True})()
    monkeypatch.setattr(inference_api, "stream_manager", manager)
    app = FastAPI()
    app.include_router(inference_api.inference_router)
    client = TestClient(app)

    response = client.put(
        "/api/inference/config",
        json={"engine": "rtdetr", "model": "rtdetr-x.pt"},
    )
    manager._sync_worker_models()

    assert response.status_code == 200
    assert response.json()["model"] == "rtdetr-x.pt"
    assert manager.get_source_models("cam") is None
    assert manager._worker.get_pool_status()["models"] == ["rtdetr-x.pt"]


def test_engine_api_roundtrips_state_and_scopes_models(monkeypatch) -> None:
    manager = StreamManager()
    monkeypatch.setattr(inference_api, "stream_manager", manager)
    app = FastAPI()
    app.include_router(inference_api.inference_router)
    client = TestClient(app)

    engines_response = client.get("/api/inference/engines")
    assert engines_response.status_code == 200
    specs = engines_response.json()
    assert [spec["id"] for spec in specs] == [
        "yolo26",
        "rtdetr",
        "rfdetr",
        "yolo_world",
    ]
    assert specs[-1]["prompt"] is True
    assert specs[-1]["closed_set_coco"] is False
    assert [spec["implemented"] for spec in specs] == [True, True, True, True]

    response = client.put(
        "/api/inference/config",
        json={"engine": "rtdetr", "prompt": "person, car"},
    )
    assert response.status_code == 200
    assert response.json()["engine"] == "rtdetr"
    assert response.json()["model"] == "rtdetr-l.pt"
    assert response.json()["prompt"] == "person, car"

    models = client.get("/api/inference/models?engine=rtdetr")
    assert models.status_code == 200
    assert [item["name"] for item in models.json()] == [
        "rtdetr-l.pt",
        "rtdetr-x.pt",
    ]
    assert client.get(
        "/api/inference/models/yolov8s-worldv2.pt/classes"
    ).json() == []


def test_engine_api_rejects_unknown_engine(monkeypatch) -> None:
    manager = StreamManager()
    monkeypatch.setattr(inference_api, "stream_manager", manager)
    app = FastAPI()
    app.include_router(inference_api.inference_router)
    client = TestClient(app)

    assert client.put(
        "/api/inference/config", json={"engine": "unknown"}
    ).status_code == 400
    assert client.get(
        "/api/inference/models?engine=unknown"
    ).status_code == 400


def test_engine_api_rejects_unimplemented_engine_without_mutation(monkeypatch) -> None:
    manager = StreamManager()
    monkeypatch.setattr(inference_api, "stream_manager", manager)
    monkeypatch.setattr(inference_api.adapters, "is_implemented", lambda engine: False)
    app = FastAPI()
    app.include_router(inference_api.inference_router)
    client = TestClient(app)
    before = client.get("/api/inference/config").json()

    response = client.put(
        "/api/inference/config", json={"engine": "yolo_world"}
    )

    assert response.status_code == 400
    assert client.get("/api/inference/config").json() == before


def test_engine_and_model_are_prevalidated_before_state_mutation(monkeypatch) -> None:
    manager = StreamManager()
    monkeypatch.setattr(inference_api, "stream_manager", manager)
    app = FastAPI()
    app.include_router(inference_api.inference_router)
    client = TestClient(app)
    before = client.get("/api/inference/config").json()

    response = client.put(
        "/api/inference/config",
        json={"engine": "rtdetr", "model": "yolo26n.pt"},
    )

    assert response.status_code == 400
    assert client.get("/api/inference/config").json() == before


def test_per_camera_inference_response_shape_is_unchanged(monkeypatch) -> None:
    manager = StreamManager()
    monkeypatch.setattr(ipcam, "stream_manager", manager)
    app = FastAPI()
    app.include_router(ipcam.router)
    client = TestClient(app)

    response = client.put(
        "/api/ipcams/demo/inference",
        json={"enabled": True, "conf_threshold": 0.4, "models": ["yolo26n.pt"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "enabled": True,
        "conf_threshold": 0.4,
        "models": ["yolo26n.pt"],
    }
