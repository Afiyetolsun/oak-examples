"""
Test script: All NEW refactored nodes.

Tests:
- CameraSourceNode
- NNDetectionNode
- TrackingNode
- PromptsNode
- SnappingNode
- ExportService
"""
import depthai as dai
import logging as log

from config.system_configuration import SystemConfiguration

from camera.camera_source_node import CameraSourceNode
from nn.nn_detection_node import NNDetectionNode
from tracking.tracking_node import TrackingNode
from prompting.prompts_node import PromptsNode
from snapping.snapping_node import SnappingNode

from export_service import ExportService

log.basicConfig(level=log.INFO)
logger = log.getLogger(__name__)


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

        camera_source = pipeline.create(CameraSourceNode).build(
            cfg=config.get_video_config()
        )
        cam_out = camera_source.preview
        encoded_out = camera_source.encoded

        visualizer.addTopic("Video", encoded_out)
        logger.info("CameraSourceNode created")

        nn_node = pipeline.create(NNDetectionNode).build(
            image_source=cam_out,
            cfg=config.get_neural_network_config(),
        )

        visualizer.addTopic("Annotations", nn_node.detections_extended)
        logger.info("NNDetectionNode created")

        tracking_node = pipeline.create(TrackingNode).build(
            image_source=cam_out,
            detections=nn_node.detections,
            cfg=config.get_neural_network_config().nn_yaml.tracker,
        )
        logger.info("TrackingNode created")

        prompts_node = pipeline.create(PromptsNode).build(
            image_source=cam_out,
            controller=nn_node.controller,
            cfg=config.get_prompts_config(),
        )
        prompts_node.register_services(visualizer)
        logger.info("PromptsNode created!")

        snapping_node = pipeline.create(SnappingNode).build(
            image_source=cam_out,
            detections=nn_node.detections,
            tracklets=tracking_node.tracklets,
            cfg=config.get_snaps_config(),
        )
        snapping_node.register_service(visualizer)
        logger.info("SnappingNode created!")

        export_service = ExportService(
            nn_node.controller.get_model_state(),
            snapping_node.conditions,
        )
        visualizer.registerService(export_service.name, export_service.handle)
        logger.info("ExportService registered")

        logger.info("Pipeline created. Starting...")
        pipeline.start()
        visualizer.registerPipeline(pipeline)
        logger.info("Pipeline running!")

        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            pipeline.processTasks()
            if key == ord("q"):
                logger.info("Got 'q' key. Exiting...")
                break


if __name__ == "__main__":
    main()
