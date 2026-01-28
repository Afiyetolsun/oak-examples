import depthai as dai
from box import Box
from typing import Dict, Any

from depthai_nodes.node import SnapsUploader

from snapping.snaps_producer import SnapsProducer
from snapping.conditions import Condition, ConditionKey, ConditionConfig, build_conditions
from pydantic import RootModel, ValidationError


class SnapPayload(RootModel[Dict[ConditionKey, ConditionConfig]]):
    """Payload for updating multiple conditions at once."""
    pass


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

        return self

    def fe_update_conditions(self, payload: dict) -> Dict[str, Any]:
        try:
            validated = SnapPayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        any_active = False
        for key, params in validated.root.items():
            cond = self._conditions.get(key)
            if cond is None:
                continue
            cond.apply_config(params)
            any_active = any_active or cond.enabled

        return {"ok": True, "active": any_active}

    def export_snap_conditions_config(self) -> dict[str, Any]:
        """Export current snapping state in a FE-friendly dict."""
        return {
            "running": any(c.enabled for c in self._conditions.values()),
            **{k.value: c.export_config() for k, c in self._conditions.items()},
        }

    def run(self) -> None:
        # High-level node: no host-side processing here. Processing happens in the composed subnodes.
        pass
