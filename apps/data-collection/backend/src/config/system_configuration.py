
from dataclasses import dataclass
from pathlib import Path
from argparse import Namespace

from box import Box
import depthai as dai

from .config_data_classes import ModelInfo, VideoConfig, NeuralNetworkConfig, TrackingConfig


@dataclass
class SystemConfig:
    """All configuration for the pipeline."""
    video: VideoConfig
    nn: NeuralNetworkConfig
    tracker: TrackingConfig
    snaps: Box


def build_configuration(platform: str, args: Namespace) -> SystemConfig:
    """Build all configuration from CLI args and YAML files."""
    yaml = _load_yamls(Path(__file__).parent / "yaml_configs")
    model = _load_model(platform, yaml.prompts.precision)

    # Video config
    fps = args.fps_limit or yaml.video.default_fps
    video = VideoConfig(
        resolution=yaml.video.video_resolution,
        fps=fps,
        media_path=args.media_path,
        width=model.width,
        height=model.height,
    )

    # NN config
    b = yaml.nn.nn_backend
    nn = NeuralNetworkConfig(
        model=model,
        backend_type=b.type,
        runtime=b.runtime,
        performance_profile=b.performance_profile,
        num_inference_threads=b.inference_threads,
        prompts=yaml.prompts,
    )

    # Tracking config
    t = yaml.nn.tracker
    tracker = TrackingConfig(
        track_per_class=t.track_per_class,
        birth_threshold=t.birth_threshold,
        max_lifespan=t.max_lifespan,
        occlusion_ratio_threshold=t.occlusion_ratio_threshold,
        tracker_threshold=t.tracker_threshold,
    )

    return SystemConfig(
        video=video,
        nn=nn,
        tracker=tracker,
        snaps=yaml.conditions,
    )


def _load_yamls(base: Path) -> Box:
    def safe_load(file: str) -> Box:
        path = base / file
        if not path.exists():
            raise FileNotFoundError(f"Missing YAML: {path}")
        return Box.from_yaml(filename=path)

    return Box({
        "nn": safe_load("nn_config.yaml"),
        "video": safe_load("visual_constants.yaml"),
        "conditions": safe_load("conditions.yaml"),
        "prompts": safe_load("prompts_config.yaml"),
    })


def _load_model(platform: str, precision: str) -> ModelInfo:
    models_dir = Path(__file__).parent.parent / "depthai_models"
    yaml_path = models_dir / f"yoloe_v8_l_fp16.{platform}.yaml"

    if not yaml_path.exists():
        raise SystemExit(f"Model YAML not found: {yaml_path}")

    desc = dai.NNModelDescription.fromYamlFile(str(yaml_path))
    desc.platform = platform
    archive = dai.NNArchive(dai.getModelFromZoo(desc))
    w, h = archive.getInputSize()

    return ModelInfo(yaml_path, w, h, desc, archive, precision)
