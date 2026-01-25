import depthai as dai
from typing import Optional
from config.config_data_classes import TrackingConfig


class TrackingNode(dai.node.ThreadedHostNode):
    """
    High-level node grouping the object tracking block.

    Exposes:
      - tracklets: dai.Tracklets with tracking IDs.
    """

    def __init__(self) -> None:
        super().__init__()

        self._tracker: dai.node.ObjectTracker = self.createSubnode(dai.node.ObjectTracker)
        self.tracklets: Optional[dai.Node.Output] = None

    def build(
        self,
        image_source: dai.Node.Output,
        detections: dai.Node.Output,
        cfg: TrackingConfig,
    ) -> "TrackingNode":
        """
        @param image_source: Frame stream used by the tracker (dai.ImgFrame).
        @param detections: Detection stream to track (dai.ImgDetections).
        @param cfg: Tracker configuration.
        """
        self._tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
        self._tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
        self._tracker.setTrackingPerClass(cfg.track_per_class)
        self._tracker.setTrackletBirthThreshold(cfg.birth_threshold)
        self._tracker.setTrackletMaxLifespan(cfg.max_lifespan)
        self._tracker.setOcclusionRatioThreshold(cfg.occlusion_ratio_threshold)
        self._tracker.setTrackerThreshold(cfg.tracker_threshold)

        image_source.link(self._tracker.inputTrackerFrame)
        image_source.link(self._tracker.inputDetectionFrame)
        detections.link(self._tracker.inputDetections)

        # Outputs
        self.tracklets = self._tracker.out
        return self

    def run(self) -> None:
        # High-level node: no host-side processing here. Processing happens in the composed subnodes.
        pass
