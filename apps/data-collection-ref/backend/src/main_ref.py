import depthai as dai
from config.system_configuration import SystemConfiguration
from core.export_service import ExportService
from core.neural_network.prompts.nn_prompts_manager import NNPromptsManager
from core.neural_network.pipeline.nn_pipeline_setup import NNPipelineBuilder
from core.snapping.snaps_manager import SnappingServiceManager
from core.video.video_factory import VideoFactory
import logging as log

from camera.camera_source_node import CameraSourceNode
from nn.nn_detection_node import NNDetectionNode
from tracking.tracking_node import TrackingNode

log.basicConfig(level=log.INFO)
logger = log.getLogger(__name__)


def main():
    log.basicConfig(level=log.INFO)
    device = dai.Device()
    visualizer = dai.RemoteConnection(serveFrontend=False)

    platform = device.getPlatformAsString()

    if platform != "RVC4":
        raise ValueError("This example is supported only on RVC4 platform")

    config = SystemConfiguration(platform)
    config.build()

    with dai.Pipeline(device) as pipeline:
        logger.info("Creating pipeline...")

        camera_source = pipeline.create(CameraSourceNode).build(config.get_video_config())
        cam_out = camera_source.preview
        encoded_out = camera_source.encoded

        nn_node = pipeline.create(NNDetectionNode).build(
            image_source=cam_out,
            cfg=config.nn,
        )

        tracking_node = pipeline.create(TrackingNode).build(
            image_source=cam_out,
            detections=nn_node.detections,
            cfg=config.nn.tracker,
        )

        prompts_node = pipeline.create(PromptsNode).build(
            image_source=cam_out,
            controller=nn_node.controller,
            cfg=config.prompts,
        )

        snapping_node = pipeline.create(SnappingNode).build(
            image_source=cam_out,
            tracker=tracking_node,
            detections=nn_node.detections,
            cfg=config.get_snaps_config(),
        )

        visualizer.addTopic("Video", encoded_out)
        visualizer.addTopic("Annotations", nn_node.detections_extended)

        logger.info("Pipeline created.")
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            pipeline.processTasks()
            if key == ord("q"):
                print("Got q key. Exiting...")
                break


if __name__ == "__main__":
    main()
