import depthai as dai
import logging as log

from config.arguments import parse_args
from config.system_configuration import build_configuration

from camera.camera_source_node import CameraSourceNode

from nn.nn_detection_node import NNDetectionNode

from tracking.tracking_node import TrackingNode

from snapping.snapping_node import SnappingNode

from prompting.frame_cache_node import FrameCacheNode
from prompting.encoders.textual_prompt_encoder import TextualPromptEncoder
from prompting.encoders.visual_prompt_encoder import VisualPromptEncoder
from prompting.fe_services import PromptingFEServices

from app_config_service import GetAppConfigService

log.basicConfig(level=log.INFO)
logger = log.getLogger(__name__)


def main():
    device = dai.Device()
    visualizer = dai.RemoteConnection(serveFrontend=False)

    platform = device.getPlatformAsString()
    logger.info(f"Platform: {platform}")

    if platform != "RVC4":
        raise ValueError("This example is supported only on RVC4 platform")

    args = parse_args()
    config = build_configuration(platform, args)

    with dai.Pipeline(device) as pipeline:
        logger.info("Creating pipeline with NEW nodes...")

        camera_source = pipeline.create(CameraSourceNode).build(cfg=config.video)
        logger.info("CameraSourceNode created")

        # Encoders used both by the NN controller (initial prompts) and by FE services (runtime updates)
        text_encoder = TextualPromptEncoder(config.prompts)
        visual_encoder = VisualPromptEncoder(config.prompts)

        nn_node = pipeline.create(NNDetectionNode).build(
            image_source=camera_source.bgr,
            text_encoder=text_encoder,
            visual_encoder=visual_encoder,
            cfg_nn=config.nn,
            cfg_prompts=config.prompts,
        )
        logger.info("NNDetectionNode created")

        tracking_node = pipeline.create(TrackingNode).build(
            image_source=camera_source.bgr,
            detections=nn_node.detections,
            cfg=config.tracker,
        )
        logger.info("TrackingNode created")

        snapping_node = pipeline.create(SnappingNode).build(
            image_source=camera_source.bgr,
            detections=nn_node.detections,
            tracklets=tracking_node.tracklets,
            cfg=config.snaps,
        )
        logger.info("SnappingNode created!")

        frame_cache_node = pipeline.create(FrameCacheNode).build(
            frame=camera_source.bgr,
        )
        logger.info("FrameCacheNode created")

        prompting_services = PromptingFEServices(
            controller=nn_node.controller,
            text_encoder=text_encoder,
            visual_encoder=visual_encoder,
            get_last_frame=frame_cache_node.get_last_frame,
        )

        get_config_service = GetAppConfigService(
            model_state=nn_node.controller.state,
            get_snap_conditions_config=snapping_node.export_snap_conditions_config,
        )

        # Visualizer topics
        visualizer.addTopic("Video", camera_source.encoded)
        visualizer.addTopic("Annotations", nn_node.detections_extended)

        # Register FE services
        visualizer.registerService("Class Update Service", prompting_services.fe_class_update)
        visualizer.registerService("Threshold Update Service", prompting_services.fe_threshold_update)
        visualizer.registerService("Image Upload Service", prompting_services.fe_image_upload)
        visualizer.registerService("BBox Prompt Service", prompting_services.fe_bbox_prompt)
        visualizer.registerService("Snap Collection Service", snapping_node.fe_update_conditions)
        visualizer.registerService("Get App Config Service", get_config_service.handle)
        logger.info("FE services registered!")

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
