from typing import List, Tuple

import depthai as dai
import numpy as np

from depthai_nodes.node import BaseHostNode


class TileGridVisualizer(BaseHostNode):
    """Visualizes tile grid overlay on frames."""

    def __init__(self) -> None:
        super().__init__()
        self._tile_positions: List[Tuple[int, int, int, int]] = []

        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

    def build(
        self,
        preview: dai.Node.Output,
        tile_positions: List[Tuple[int, int, int, int]],
    ) -> "TileGridVisualizer":
        self.link_args(preview)
        self._tile_positions = tile_positions
        return self

    @property
    def tile_positions(self) -> List[Tuple[int, int, int, int]]:
        return self._tile_positions

    @tile_positions.setter
    def tile_positions(self, value: List[Tuple[int, int, int, int]]) -> None:
        self._tile_positions = value

    def process(self, preview: dai.ImgFrame) -> None:
        frame = preview.getCvFrame()
        annotations = self._create_grid_annotations(
            frame, preview.getTimestamp(), preview.getSequenceNum()
        )
        self._out.send(annotations)

    def _create_grid_annotations(
        self, frame: np.ndarray, timestamp, sequence_num: int
    ) -> dai.ImgAnnotations:
        """Create a tile grid overlay annotations."""
        print(
            f"[TileGridVisualizer] Creating annotations with {len(self._tile_positions)} tiles"
        )
        print(f"[TileGridVisualizer] Frame shape: {frame.shape}")
        img_annots = dai.ImgAnnotations()
        img_annot = dai.ImgAnnotation()

        if self._tile_positions:
            img_height, img_width = frame.shape[:2]

            np.random.seed(432)
            colors = [
                dai.Color(
                    np.random.random(), np.random.random(), np.random.random(), 0.3
                )
                for _ in range(len(self._tile_positions))
            ]

            for idx, (x1, y1, x2, y2) in enumerate(self._tile_positions):
                rect = dai.PointsAnnotation()
                rect.fillColor = colors[idx]
                rect.points.extend(
                    [
                        dai.Point2f(x1 / img_width, y1 / img_height),
                        dai.Point2f(x1 / img_width, y2 / img_height),
                        dai.Point2f(x2 / img_width, y2 / img_height),
                        dai.Point2f(x2 / img_width, y1 / img_height),
                    ]
                )
                rect.type = dai.PointsAnnotationType.LINE_LOOP
                img_annot.points.append(rect)

        tile_count_text = dai.TextAnnotation()
        tile_count_text.fontSize = 25
        tile_count_text.text = f"Tiles: {len(self._tile_positions)}"
        tile_count_text.position = dai.Point2f(0.05, 0.05)
        tile_count_text.textColor = dai.Color(0.0, 0.0, 0.0)
        tile_count_text.backgroundColor = dai.Color(0.0, 1.0, 0.0)
        img_annot.texts.append(tile_count_text)

        img_annots.annotations.append(img_annot)
        img_annots.setTimestamp(timestamp)
        img_annots.setSequenceNum(sequence_num)

        return img_annots

    @property
    def out(self) -> dai.Node.Output:
        return self._out
