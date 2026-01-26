from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, TilesPatcher

from tiling.dynamic_tiling import DynamicTiling
from qr_scan.host_qr_scanner import QRScanner
from tiling.tile_grid_visualizer import TileGridVisualizer
from tiling.tiling_config_service import TilingConfigService

IMG_SHAPE = (1920, 1080)

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    platform = device.getPlatform()
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            dai.NNModelDescription.fromYamlFile(
                Path(f"qrdet_nano.{platform.name}.yaml")
            )
        )
    )

    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput(IMG_SHAPE, type=dai.ImgFrame.Type.NV12)

    tile_manager = pipeline.create(DynamicTiling).build(
        img_output=cam_out,
        img_shape=IMG_SHAPE,
        nn_shape=nn_archive.getInputSize(),
        resize_mode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )

    interleaved_manip = pipeline.create(dai.node.ImageManip)
    interleaved_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
    interleaved_manip.setMaxOutputFrameSize(
        nn_archive.getInputHeight() * nn_archive.getInputWidth() * 3
    )
    tile_manager.out.link(interleaved_manip.inputImage)

    nn_input = interleaved_manip.out

    nn = pipeline.create(ParsingNeuralNetwork).build(
        input=nn_input, nn_source=nn_archive
    )

    patcher = pipeline.create(TilesPatcher).build(
        img_frames=cam_out, nn=nn.out, conf_thresh=0.3, iou_thresh=0.2
    )

    scanner = pipeline.create(QRScanner).build(
        preview=cam_out,
        detections=patcher.out,
    )

    grid_visualizer = pipeline.create(TileGridVisualizer).build(
        preview=cam_out, tile_positions=tile_manager.tile_positions
    )

    encoder = pipeline.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(
        fps=30,
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
    )
    cam_out.link(encoder.input)

    visualizer.addTopic("Video", encoder.out, "images")
    visualizer.addTopic("Visualizations", scanner.out, "images")
    visualizer.addTopic("Tiling grid", grid_visualizer.out, "images")

    tiling_service = TilingConfigService(
        tile_manager=tile_manager, grid_visualizer=grid_visualizer
    )
    visualizer.registerService(tiling_service.NAME, tiling_service)
    visualizer.registerService(tiling_service.FETCH, tiling_service.get_current_params)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
