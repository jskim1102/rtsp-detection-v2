"""서브프로세스 DetectorAdapter 계약."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.inference.worker import Detection


class DetectorSession(Protocol):
    """엔진별 세션이 worker에 제공하는 최소 계약."""

    names: dict[int, str]

    def infer(
        self,
        frames: list[np.ndarray],
        *,
        conf: float,
        imgsz: int,
    ) -> list[list[Detection]]: ...

    def to(self, device: str) -> None: ...

    def set_classes(self, names: list[str]) -> None:
        """closed-set 엔진의 기본 구현은 no-op이다."""

    def reset_classes(self) -> None:
        """prompt 엔진을 세션 로드 시 기본 어휘로 복원한다."""

    def close(self) -> None: ...
