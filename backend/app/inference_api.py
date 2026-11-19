"""추론 제어 + 모델 가중치 관리 라우터 (/api/inference/*).

deepeye-lite ipcam.py 의 inference_router 차용. main.py 에서 include_router(inference_router).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.inference import adapters, engines, models_dir
from app.streaming.manager import manager as stream_manager


# ─── 추론 제어 (모델 토글, ON/OFF, conf threshold) ───


class InferenceConfig(BaseModel):
    enabled: bool
    model: str
    conf_threshold: float
    device: str
    gpu_util_target: float
    gpu_util_duty: float
    engine: str
    prompt: str


class InferenceConfigUpdate(BaseModel):
    enabled: bool | None = None
    model: str | None = None
    conf_threshold: float | None = None
    gpu_util_target: float | None = None
    engine: str | None = None
    prompt: str | None = None


inference_router = APIRouter(prefix="/api/inference", tags=["inference"])


@inference_router.get("/config", response_model=InferenceConfig)
def get_inference_config() -> dict:
    """현재 추론 워커 상태."""
    return stream_manager.get_inference_config()


@inference_router.put("/config", response_model=InferenceConfig)
def update_inference_config(body: InferenceConfigUpdate) -> dict:
    """추론 ON/OFF · 모델 · conf threshold 변경. 부분 업데이트 지원."""
    try:
        current_engine = stream_manager.get_inference_config()["engine"]
        target_engine = body.engine or current_engine
        if target_engine not in engines.ENGINES:
            raise ValueError(f"알 수 없는 추론 엔진: {target_engine!r}")
        if not adapters.is_implemented(target_engine):
            raise ValueError(f"아직 구현되지 않은 추론 엔진: {target_engine!r}")
        if body.model is not None and engines.engine_of(body.model) != target_engine:
            raise ValueError(
                f"엔진 {target_engine!r}이 소유하지 않는 preset 모델: {body.model!r}"
            )
        if body.engine is not None:
            stream_manager.set_engine(body.engine)
        if body.enabled is not None:
            stream_manager.set_inference_enabled(body.enabled)
        if body.model is not None:
            stream_manager.set_inference_model(body.model)
        if body.conf_threshold is not None:
            stream_manager.set_inference_conf_threshold(body.conf_threshold)
        if body.gpu_util_target is not None:
            stream_manager.set_gpu_util_target(body.gpu_util_target)
        if body.prompt is not None:
            stream_manager.set_prompt(body.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return stream_manager.get_inference_config()


# ─── 모델 목록 + 클래스 메타 (preset-only — custom 업로드 제거, pickle RCE 차단) ───


class ModelInfo(BaseModel):
    name: str
    type: str  # 항상 "preset"
    size_mb: float | None = None


class EngineInfo(BaseModel):
    id: str
    label: str
    presets: tuple[str, ...]
    default: str
    closed_set_coco: bool
    prompt: bool
    implemented: bool


@inference_router.get("/engines", response_model=list[EngineInfo])
def list_engines() -> list[dict[str, object]]:
    """사용 가능한 엔진과 UI capability를 무거운 import 없이 반환한다."""
    return [
        {**spec.to_dict(), "implemented": adapters.is_implemented(spec.id)}
        for spec in engines.ENGINES.values()
    ]


@inference_router.get("/models", response_model=list[ModelInfo])
def list_models(engine: str | None = None) -> list[dict]:
    """선택 엔진의 preset 목록. custom 업로드는 제거됨."""
    try:
        return models_dir.list_all_models(engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@inference_router.get("/models/{name}/classes")
def get_model_classes(name: str) -> list[dict]:
    """주어진 preset 모델의 클래스 ID→이름 목록.

    허용된 YOLO26 detection preset은 모두 COCO 80-class 모델이다. 클래스 이름 조회는
    정적 메타데이터만 반환하고 가중치 다운로드·모델 로드를 시작하지 않는다.
    """
    if not models_dir.is_preset(name):
        raise HTTPException(status_code=404, detail="알 수 없는 모델 (preset 만 가능)")

    return models_dir.list_model_classes(name)
