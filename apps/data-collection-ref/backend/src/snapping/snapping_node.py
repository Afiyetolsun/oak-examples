import depthai as dai
from box import Box
from typing import Optional

from depthai_nodes.node import SnapsUploader

from snapping.snaps_producer import SnapsProducer
from snapping.conditions import Condition, ConditionKey, build_conditions
from snapping.snapping_service import SnappingService


class SnappingNode(dai.node.ThreadedHostNode):
    """
    High-level node for automatic data snapping/collection.

    Internal structure:
        frame + detections + tracklets
          -> SnapsProducer (evaluates conditions, creates SnapData)
          -> SnapsUploader (uploads snaps)

    Exposes:
      - service: SnappingService for runtime configuration via RemoteConnection
      - conditions: condition instances (used by ExportService)
    """

    def __init__(self) -> None:
        super().__init__()

        self._producer: SnapsProducer = self.createSubnode(SnapsProducer)
        self._uploader: SnapsUploader = self.createSubnode(SnapsUploader)

        self._conditions: dict[ConditionKey, Condition] = {}

        self.service: Optional[SnappingService] = None

    def build(
        self,
        image_source: dai.Node.Output,
        detections: dai.Node.Output,
        tracklets: dai.Node.Output,
        cfg: Box,
    ) -> "SnappingNode":
        """
        @param image_source: BGR frames from camera.
        @param detections: dai.ImgDetections from detection node.
        @param tracklets: Tracklets output from tracking node.
        @param cfg: Snapping configuration (conditions, cooldown, etc.)
        """
        self._conditions = build_conditions(cfg)

        self._producer.build(
            frame=image_source,
            conditions=self._conditions,
            detections=detections,
            tracklets=tracklets,
        )

        self._uploader.build(self._producer.out)

        self.service = SnappingService(self._conditions)

        return self

    def register_service(self, visualizer: dai.RemoteConnection) -> None:
        """
        Register snapping service with the visualizer.

        @param visualizer: RemoteConnection to register services with.
        """
        if self.service is None:
            raise RuntimeError("SnappingNode.build() must be called before register_service()")
        visualizer.registerService(self.service.name, self.service.handle)

    @property
    def conditions(self) -> dict[ConditionKey, Condition]:
        """Get conditions dict (for ExportService)."""
        return self._conditions

    def run(self) -> None:
        # High-level node: no host-side processing here. Processing happens in the composed subnodes.
        pass
