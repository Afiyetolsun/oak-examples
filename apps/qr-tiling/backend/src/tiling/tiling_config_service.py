import depthai as dai
from pydantic import BaseModel, Field

from base_service import BaseService
from fps_control import FPSController
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

    SAFETY_MARGIN = 0.8
    TILE_DECREASE_FPS_BOOST = 2

    NN_NODE_NAME = "NeuralNetwork"

    def __init__(
        self,
        dynamic_tiling: DynamicTiling,
        pipeline: dai.Pipeline,
        fps_controller: FPSController,
    ):
        self._dynamic_tiling = dynamic_tiling
        self._pipeline = pipeline
        self._fps_controller = fps_controller

        self._old_tile_count = dynamic_tiling.tile_count

    def handle_typed(self, payload: TilingConfigPayload) -> dict:
        grid_size = (payload.cols, payload.rows)

        self._dynamic_tiling.updateConfig(
            grid_size=grid_size,
            overlap=payload.overlap,
            global_detection=payload.global_detection,
            grid_matrix=payload.grid_matrix,
        )

        # Feed-forward: pre-adjust FPS target based on new tile count
        self._adjust_fps_for_new_tiles()

        return {"ok": True}

    def _adjust_fps_for_new_tiles(self) -> None:
        tile_count = self._dynamic_tiling.tile_count

        if tile_count == self._old_tile_count:
            return

        if tile_count > self._old_tile_count:
            # Number of tiles increased: FPS needs to drop -> new safe estimate
            est_fps = self._estimate_max_fps(tile_count)
            if est_fps is not None:
                print(
                    f"[TILES_INCREASE] {self._old_tile_count}\u2192{tile_count}, est={est_fps}"
                )
                self._fps_controller.set_target(est_fps)
        else:
            # Number of tiles decreased: FPS can increase -> let the feedback loop rise
            print(
                f"[TILES_DECREASE] {self._old_tile_count}\u2192{tile_count}, allowing rise"
            )
            self._fps_controller.set_target(
                self._fps_controller.current_fps + self.TILE_DECREASE_FPS_BOOST
            )
        self._old_tile_count = tile_count

    def _estimate_max_fps(self, tile_count: int) -> int | None:
        # Estimate safe FPS assuming the NN is the bottleneck.
        # Uses medianMicrosRecent (per-tile NN processing time) from pipeline state.
        try:
            pipeline_state = self._pipeline.getPipelineState().nodes().detailed()
            for node_id, ns in pipeline_state.nodeStates.items():
                if self._pipeline.getNode(node_id).getName() == self.NN_NODE_NAME:
                    nn_tile_us = ns.mainLoopTiming.durationStats.medianMicrosRecent
                    est_max_fps = 1_000_000 / (nn_tile_us * tile_count)
                    est_fps_target = int(est_max_fps * self.SAFETY_MARGIN)
                    print(
                        f"[PIPELINE_STATE] tiles={tile_count}, nn_us={nn_tile_us}, est_target={est_fps_target}"
                    )
                    return est_fps_target
        except Exception as e:
            print("Error getting pipeline state:", e)
        return None
