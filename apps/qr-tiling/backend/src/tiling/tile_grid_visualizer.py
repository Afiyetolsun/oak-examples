from typing import List, Tuple

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.node import BaseHostNode


class TileGridOverlay(BaseHostNode):
    """Draws tile grid overlay directly on frames."""

    def __init__(self) -> None:
        super().__init__()
        self._tile_positions: List[Tuple[int, int, int, int]] = []
        self._source_size: Tuple[int, int] | None = None
        self._colors: List[Tuple[int, int, int]] = []

    def build(
        self,
        preview: dai.Node.Output,
        tile_positions: List[Tuple[int, int, int, int]],
        source_size: Tuple[int, int] | None = None,
    ) -> "TileGridOverlay":
        self.link_args(preview)
        self._tile_positions = tile_positions
        self._source_size = source_size
        self._generate_colors()
        return self

    @property
    def tile_positions(self) -> List[Tuple[int, int, int, int]]:
        return self._tile_positions

    @tile_positions.setter
    def tile_positions(self, value: List[Tuple[int, int, int, int]]) -> None:
        self._tile_positions = value
        self._generate_colors()

    def _generate_colors(self) -> None:
        """Generate random colors for tiles."""
        np.random.seed(432)
        self._colors = [
            (
                int(np.random.random() * 255),
                int(np.random.random() * 255),
                int(np.random.random() * 255),
            )
            for _ in range(max(len(self._tile_positions), 1))
        ]

    def process(self, preview: dai.ImgFrame) -> None:
        frame = preview.getCvFrame()
        frame_with_grid = self._draw_grid(frame)

        out_frame = dai.ImgFrame()
        out_frame.setCvFrame(frame_with_grid, dai.ImgFrame.Type.BGR888p)
        out_frame.setTimestamp(preview.getTimestamp())
        out_frame.setSequenceNum(preview.getSequenceNum())

        self.out.send(out_frame)

    def _scale_positions(
        self, frame_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Scale tile positions from source size to frame size."""
        if not self._source_size:
            return self._tile_positions

        src_w, src_h = self._source_size
        dst_w, dst_h = frame_size

        scale_x = dst_w / src_w
        scale_y = dst_h / src_h

        return [
            (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            )
            for x1, y1, x2, y2 in self._tile_positions
        ]

    def _draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """Draw tile grid overlay on the frame."""
        frame_h, frame_w = frame.shape[:2]
        scaled_positions = self._scale_positions((frame_w, frame_h))

        overlay = frame.copy()

        for idx, (x1, y1, x2, y2) in enumerate(scaled_positions):
            color = self._colors[idx % len(self._colors)]

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        text = f"Tiles: {len(self._tile_positions)}"
        cv2.putText(
            frame,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return frame
