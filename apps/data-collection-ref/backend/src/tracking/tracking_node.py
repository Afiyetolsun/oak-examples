import depthai as dai


class TrackingNode(dai.node.ThreadedHostNode):
    """
    High-level node grouping the tracking block.

    Exposes:
      - tracklets: dai.Tracklets with tracking IDs.
    """

    def __init__(self) -> None:
        super().__init__()

        self._tracker: dai.node.Tracker = self.createSubnode(dai.node.ObjectTracker)

        # Outputs
        self.tracklets: dai.Node.Output = None

    def build(
        self,
        image_source: dai.Node.Output,
        detections: dai.Node.Output,
        cfg,
    ) -> "TrackingNode":

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
        # High-level node: no host-side processing, subnodes run on device.
        pass
