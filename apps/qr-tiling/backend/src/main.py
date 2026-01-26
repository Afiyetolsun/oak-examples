from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, TilesPatcher

from params_service import CurrentParamsService
from qr_scan.qr_service import QRConfigService
from tiling.dynamic_tiling import DynamicTiling
from qr_scan.host_qr_scanner import QRScanner
from tiling.tile_grid_visualizer import TileGridOverlay
from tiling.tiling_config_service import TilingConfigService

TILING_SIZE = (3840, 2160)
OUT_SIZE = (1920, 1080)

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

    camera = pipeline.create(dai.node.Camera).build()

    rbg_tiling = camera.requestOutput(TILING_SIZE, type=dai.ImgFrame.Type.BGR888i)
    rgb_out = camera.requestOutput(OUT_SIZE, type=dai.ImgFrame.Type.NV12)

    dynamic_tiling = pipeline.create(DynamicTiling).build(
        img_output=rbg_tiling,
        img_shape=TILING_SIZE,
        nn_shape=nn_archive.getInputSize(),
        resize_mode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )

    interleaved_manip = pipeline.create(dai.node.ImageManip)
    interleaved_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
    interleaved_manip.setMaxOutputFrameSize(
        nn_archive.getInputHeight() * nn_archive.getInputWidth() * 3
    )
    dynamic_tiling.out.link(interleaved_manip.inputImage)

    nn_input = interleaved_manip.out

    nn = pipeline.create(ParsingNeuralNetwork).build(
        input=nn_input, nn_source=nn_archive
    )

    patcher = pipeline.create(TilesPatcher).build(
        img_frames=rbg_tiling, nn=nn.out, conf_thresh=0.3, iou_thresh=0.2
    )

    scanner = pipeline.create(QRScanner).build(
        preview=rbg_tiling,
        detections=patcher.out,
    )

    qr_service = QRConfigService(scanner=scanner)
    visualizer.registerService(qr_service.NAME, qr_service)

    grid_overlay = pipeline.create(TileGridOverlay).build(
        preview=rgb_out,
        tile_positions=dynamic_tiling.tile_positions,
        source_size=TILING_SIZE,
    )

    grid_manip = pipeline.create(dai.node.ImageManip)
    grid_manip.initialConfig.setOutputSize(OUT_SIZE[0], OUT_SIZE[1])
    grid_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    grid_manip.setMaxOutputFrameSize(int(OUT_SIZE[0] * OUT_SIZE[1] * 3))

    grid_overlay.out.link(grid_manip.inputImage)

    encoder = pipeline.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(
        fps=30,
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
    )
    grid_manip.out.link(encoder.input)

    visualizer.addTopic("Video", encoder.out, "images")
    visualizer.addTopic("Visualizations", scanner.out, "images")

    tiling_service = TilingConfigService(
        dynamic_tiling=dynamic_tiling, grid_visualizer=grid_overlay
    )

    visualizer.registerService(tiling_service.NAME, tiling_service)

    params_service = CurrentParamsService(
        dynamic_tiling=dynamic_tiling, qr_scanner=scanner
    )

    visualizer.registerService(params_service.NAME, params_service)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
