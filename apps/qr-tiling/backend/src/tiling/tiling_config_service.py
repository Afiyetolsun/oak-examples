from pydantic import BaseModel, Field

from base_service import BaseService
from fps_control import FPSCalculator
from tiling import DynamicTiling


class TilingConfigPayload(BaseModel):
    rows: int = Field(..., ge=1, le=8)
    cols: int = Field(..., ge=1, le=8)
    overlap: float = Field(0.2, ge=0.0, lt=1.0)
    global_detection: bool = False
    grid_matrix: list[list[int]] | None = None


class TilingConfigService(BaseService[TilingConfigPayload]):
    NAME = "Tiling Config Service"
    PAYLOAD_MODEL = TilingConfigPayload

    def __init__(
        self,
        dynamic_tiling: DynamicTiling,
        fps_calculator: FPSCalculator,
    ):
        self._dynamic_tiling = dynamic_tiling
        self._fps_calculator = fps_calculator
        self._old_tile_count = dynamic_tiling.tile_count

    def handle_typed(self, payload: TilingConfigPayload) -> dict:
        grid_size = (payload.cols, payload.rows)

        self._dynamic_tiling.updateConfig(
            grid_size=grid_size,
            overlap=payload.overlap,
            global_detection=payload.global_detection,
            grid_matrix=payload.grid_matrix,
        )

        new_tile_count = self._dynamic_tiling.tile_count
        if new_tile_count != self._old_tile_count:
            self._fps_calculator.adjust_fps_from_tile_count(new_tile_count)
            self._old_tile_count = new_tile_count

        return {"ok": True}
