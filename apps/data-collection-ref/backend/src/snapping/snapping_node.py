import depthai as dai
from box import Box

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
          -> SnapsUploader (handles storage/transmission)

    Components:
        - Conditions: Configurable triggers (timed, no_detections, etc.)
        - SnappingService: Frontend API for configuration

    Usage:
        snapping_node = pipeline.create(SnappingNode).build(...)
        visualizer.registerService(snapping_node.service.name, snapping_node.service.handle)
    """

    def __init__(self):
        super().__init__()

        self._producer: SnapsProducer = self.createSubnode(SnapsProducer)
        self._uploader: SnapsUploader = self.createSubnode(SnapsUploader)

        self._conditions: dict[ConditionKey, Condition] = {}

        self.service: SnappingService = None

    def build(
        self,
        image_source: dai.Node.Output,
        detections: dai.Node.Output,
        tracklets: dai.Node.Output,
        cfg: Box,
    ) -> "SnappingNode":
        """
        Build the snapping node.

        @param image_source: BGR frames from camera.
        @param detections: ImgDetections from detection node.
        @param tracklets: Tracklets output from tracking node.
        @param cfg: Snapping configuration (conditions, cooldown, etc.)
        """
        # Build conditions from config
        self._conditions = build_conditions(cfg)

        # Configure producer
        self._producer.build(
            frame=image_source,
            conditions=self._conditions,
            detections=detections,
            tracklets=tracklets,
        )

        # Link producer -> uploader
        self._uploader.build(self._producer.out)

        # Create service
        self.service = SnappingService(self._conditions)

        return self

    def register_service(self, visualizer: dai.RemoteConnection) -> None:
        """
        Register snapping service with the visualizer.

        @param visualizer: RemoteConnection to register services with.
        """
        visualizer.registerService(self.service.name, self.service.handle)

    @property
    def conditions(self) -> dict[ConditionKey, Condition]:
        """Get conditions dict (for ExportService)."""
        return self._conditions

    def run(self) -> None:
        # High-level node: subnodes handle processing.
        pass
