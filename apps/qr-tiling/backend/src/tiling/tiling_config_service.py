from typing import List, Optional

from pydantic import BaseModel, Field

from base_service import BaseService
from tiling.dynamic_tiling import DynamicTiling
from tiling.tile_grid_overlay import TileGridOverlay


class TilingConfigPayload(BaseModel):
    rows: int = Field(..., ge=1, le=8)
    cols: int = Field(..., ge=1, le=8)
    overlap: float = Field(0.2, ge=0.0, lt=1.0)
    global_detection: bool = False
    grid_matrix: Optional[List[List[int]]] = None


class TilingConfigService(BaseService[TilingConfigPayload]):
    NAME = "Tiling Config Service"
    PAYLOAD_MODEL = TilingConfigPayload

    def __init__(self, dynamic_tiling: DynamicTiling, grid_visualizer: TileGridOverlay):
        self._dynamic_tiling = dynamic_tiling
        self._grid_visualizer = grid_visualizer

    def handle_typed(self, payload: TilingConfigPayload) -> dict:
        grid_size = (payload.cols, payload.rows)

        self._dynamic_tiling.updateConfig(
            grid_size=grid_size,
            overlap=payload.overlap,
            global_detection=payload.global_detection,
            grid_matrix=payload.grid_matrix,
        )

        self._grid_visualizer.tile_positions = self._dynamic_tiling.tile_positions

        return {"ok": True}
