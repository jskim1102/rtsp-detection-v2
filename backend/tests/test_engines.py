"""엔진 registry와 preset-only 신뢰경계 계약."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.inference import engines, models_dir


BACKEND = Path(__file__).resolve().parents[1]


def test_registry_exposes_four_engine_specs_and_capabilities() -> None:
    assert tuple(engines.ENGINES) == (
        "yolo26",
        "rtdetr",
        "rfdetr",
        "yolo_world",
    )
    assert engines.DEFAULT_ENGINE == "yolo26"
    assert engines.ENGINES["yolo26"].default == "yolo26n.pt"
    assert engines.ENGINES["rtdetr"].default == "rtdetr-l.pt"
    assert engines.ENGINES["rfdetr"].default == "rf-detr-base.pth"
    assert engines.ENGINES["yolo_world"].prompt is True
    assert engines.ENGINES["yolo_world"].closed_set_coco is False
    assert all(
        not spec.prompt and spec.closed_set_coco
        for key, spec in engines.ENGINES.items()
        if key != "yolo_world"
    )


@pytest.mark.parametrize(
    ("model_name", "engine"),
    [
        ("yolo26n.pt", "yolo26"),
        ("rtdetr-l.pt", "rtdetr"),
        ("rf-detr-base.pth", "rfdetr"),
        ("yolov8s-worldv2.pt", "yolo_world"),
    ],
)
def test_engine_of_and_preset_union(model_name: str, engine: str) -> None:
    assert engines.engine_of(model_name) == engine
    assert engines.is_preset(model_name) is True
    assert models_dir.is_preset(model_name) is True


def test_models_are_scoped_and_open_vocabulary_has_no_static_classes() -> None:
    assert [item["name"] for item in models_dir.list_all_models("rtdetr")] == [
        "rtdetr-l.pt",
        "rtdetr-x.pt",
    ]
    assert [item["name"] for item in models_dir.list_all_models()] == list(
        engines.ENGINES[engines.DEFAULT_ENGINE].presets
    )
    assert len(models_dir.list_model_classes("rf-detr-base.pth")) == 80
    assert models_dir.list_model_classes("yolov8s-worldv2.pt") == []

    with pytest.raises(ValueError):
        models_dir.list_all_models("unknown")
    with pytest.raises(ValueError):
        models_dir.resolve_model_path("../../evil.pt")


def test_main_import_keeps_heavy_inference_packages_out_of_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.main; "
            "assert 'torch' not in sys.modules; "
            "assert 'ultralytics' not in sys.modules; "
            "assert 'rfdetr' not in sys.modules",
        ],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
