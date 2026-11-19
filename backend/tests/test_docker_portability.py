"""Docker clone portability and startup contracts."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_shell_entrypoint_is_normalized_and_invoked_through_posix_shell() -> None:
    dockerfile = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY --chmod=0755 docker-entrypoint.sh /app/docker-entrypoint.sh" in dockerfile
    assert "replace(b'\\r\\n', b'\\n')" in dockerfile
    assert "replace(b'\\r', b'\\n')" in dockerfile
    assert "removeprefix(b'\\xef\\xbb\\xbf')" in dockerfile
    assert 'ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]' in dockerfile


def test_compose_uses_public_backend_and_builds_frontend_with_client_settings() -> None:
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    backend, frontend = services["backend"], services["frontend"]
    assert backend["image"] == "ghcr.io/jskim1102/rtsp-detection-v2-backend:latest"
    assert "pull_policy" not in backend
    assert frontend["pull_policy"] == "build"
    assert "http://127.0.0.1:8000/api/health" in backend["healthcheck"]["test"][-1]
    assert "http://127.0.0.1/" in frontend["healthcheck"]["test"]
    assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"


def test_backend_weights_directory_uses_persistent_volume() -> None:
    backend = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["backend"]
    assert backend["environment"]["YOLO_WEIGHTS_DIR"] == "/app/models"
    assert "yolo_weights:/app/models" in backend["volumes"]
    assert "clip_cache:/root/.cache/clip" in backend["volumes"]


def test_only_gpu_overlay_reserves_devices() -> None:
    base = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    overlay = yaml.safe_load((ROOT / "docker-compose.gpu.yml").read_text())
    assert "deploy" not in base["services"]["backend"]
    resources = overlay["services"]["backend"]["deploy"]["resources"]
    assert resources["reservations"]["devices"][0]["capabilities"] == ["gpu"]


def test_git_forces_lf_for_shell_scripts() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "* text=auto eol=lf" in attributes
    assert "*.sh text eol=lf" in attributes
