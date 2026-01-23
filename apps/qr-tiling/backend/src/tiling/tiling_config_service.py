from typing import List, Optional, Any

from pydantic import BaseModel, Field

from base_service import BaseService
from tiling.dynamic_tiling import DynamicTiling
from tiling.tile_grid_visualizer import TileGridVisualizer


class TilingConfigPayload(BaseModel):
    rows: int = Field(..., ge=1, le=8)
    cols: int = Field(..., ge=1, le=8)
    overlap: float = Field(0.2, ge=0.0, lt=1.0)
    global_detection: bool = False
    grid_matrix: Optional[List[List[int]]] = None


class TilingConfigService(BaseService[TilingConfigPayload]):
    NAME = "Tiling Config Service"
    FETCH = "Get Current Params Service"
    PAYLOAD_MODEL = TilingConfigPayload

    def __init__(
        self, tile_manager: DynamicTiling, grid_visualizer: TileGridVisualizer
    ):
        self._tile_manager = tile_manager
        self._grid_visualizer = grid_visualizer

    def handle_typed(self, payload: TilingConfigPayload) -> dict:
        grid_size = (payload.cols, payload.rows)

        print(f"Got payload {payload}")
        self._tile_manager.updateConfig(
            grid_size=grid_size,
            overlap=payload.overlap,
            global_detection=payload.global_detection,
            grid_matrix=payload.grid_matrix,
        )

        self._grid_visualizer.tile_positions = self._tile_manager.tile_positions

        return {"ok": True}

    def get_current_params(self, __req: dict = None) -> dict[str, Any]:
        return self._tile_manager.current_params
