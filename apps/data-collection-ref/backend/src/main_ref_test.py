"""
Hybrid test script: NEW nodes + OLD managers.

Tests the refactored nodes (CameraSourceNode, NNDetectionNode, TrackingNode)
while using the old prompts and snapping managers until those are refactored.

This allows incremental testing of the refactor.
"""
import depthai as dai
import logging as log

# Config (unchanged)
from config.system_configuration import SystemConfiguration

# NEW refactored nodes
from camera.camera_source_node import CameraSourceNode
from nn.nn_detection_node import NNDetectionNode
from tracking.tracking_node import TrackingNode

# OLD managers (still using until refactored)
from core.neural_network.prompts.nn_prompts_manager import NNPromptsManager
from core.snapping.snaps_manager import SnappingServiceManager
from core.export_service import ExportService

log.basicConfig(level=log.INFO)
logger = log.getLogger(__name__)


class ControllerAdapter:
    """
    Adapter to make new PromptController compatible with old NNPromptsManager.

    Old manager calls: send_prompts_pair(), set_confidence_threshold(), get_model_state()
    New controller has: apply_prompts(), set_confidence_threshold(), get_model_state()
    """

    def __init__(self, new_controller):
        self._controller = new_controller

    def send_prompts_pair(self, visual_prompt, text_prompt, class_names, offset):
        """Adapter method - maps old interface to new."""
        self._controller.apply_prompts(visual_prompt, text_prompt, class_names, offset)

    def set_confidence_threshold(self, threshold):
        self._controller.set_confidence_threshold(threshold)

    def get_model_state(self):
        return self._controller.get_model_state()


def main():
    device = dai.Device()
    visualizer = dai.RemoteConnection(serveFrontend=False)

    platform = device.getPlatformAsString()
    logger.info(f"Platform: {platform}")

    if platform != "RVC4":
        raise ValueError("This example is supported only on RVC4 platform")

    config = SystemConfiguration(platform)
    config.build()

    with dai.Pipeline(device) as pipeline:
        logger.info("Creating pipeline with NEW nodes...")

        # ========================================
        # NEW: CameraSourceNode (replaces VideoFactory)
        # ========================================
        camera_source = pipeline.create(CameraSourceNode).build(
            cfg=config.get_video_config()
        )
        cam_out = camera_source.preview
        encoded_out = camera_source.encoded

        visualizer.addTopic("Video", encoded_out)
        logger.info("CameraSourceNode created")

        # ========================================
        # NEW: NNDetectionNode (replaces NNPipelineBuilder partially)
        # ========================================
        nn_node = pipeline.create(NNDetectionNode).build(
            image_source=cam_out,
            cfg=config.get_neural_network_config(),
        )

        visualizer.addTopic("Annotations", nn_node.detections_extended)
        logger.info("NNDetectionNode created")

        # ========================================
        # NEW: TrackingNode (replaces TrackerFactory)
        # ========================================
        tracking_node = pipeline.create(TrackingNode).build(
            image_source=cam_out,
            detections=nn_node.detections,
            cfg=config.get_neural_network_config().tracker,
        )
        logger.info("TrackingNode created")

        # ========================================
        # OLD: NNPromptsManager (with adapter for new controller)
        # ========================================
        # Wrap new controller to be compatible with old manager
        controller_adapter = ControllerAdapter(nn_node.controller)

        prompts_manager = NNPromptsManager(
            pipeline,
            cam_out,
            config.get_prompts_config(),
            controller_adapter,
        )
        prompts_manager.build()
        prompts_manager.register_services(visualizer)
        logger.info("NNPromptsManager (OLD) connected to new controller")

        # ========================================
        # OLD: SnappingServiceManager
        # ========================================
        # Note: SnappingServiceManager expects the tracker subnode and a bridge node
        # We need to pass the internal tracker and create a compatible bridge reference
        snaps_manager = SnappingServiceManager(
            pipeline,
            cam_out,
            tracking_node._tracker,  # Access internal tracker subnode
            nn_node._annotation,     # The annotated ImgDetections output node
            config.get_snaps_config(),
        )
        snaps_manager.build()
        snaps_manager.register_service(visualizer)
        logger.info("SnappingServiceManager (OLD) connected")

        # ========================================
        # OLD: ExportService
        # ========================================
        export_service = ExportService(
            nn_node.controller.get_model_state(),
            snaps_manager.get_conditions()
        )
        visualizer.registerService(export_service.name, export_service.handle)
        logger.info("ExportService registered")

        # ========================================
        # Start pipeline
        # ========================================
        logger.info("Pipeline created. Starting...")
        pipeline.start()
        visualizer.registerPipeline(pipeline)
        logger.info("Pipeline running!")

        print("\n" + "="*50)
        print("TEST SUMMARY:")
        print("  CameraSourceNode (NEW)")
        print("  NNDetectionNode (NEW)")
        print("  TrackingNode (NEW)")
        print("  NNPromptsManager (OLD) - via adapter")
        print("  SnappingServiceManager (OLD)")
        print("  ExportService (OLD)")
        print("="*50)
        print("Press 'q' to quit\n")

        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            pipeline.processTasks()
            if key == ord("q"):
                logger.info("Got 'q' key. Exiting...")
                break


if __name__ == "__main__":
    main()
