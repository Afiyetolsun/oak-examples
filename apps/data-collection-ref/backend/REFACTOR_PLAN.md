# Refactoring Plan: Data-Collection App

## Goal
Transform from over-factored enterprise architecture (40+ files) to clean, readable structure like people-demographics (~15 files) with max 2 levels deep.

---

## Current vs Target Structure

### CURRENT (Over-factored)
```
src/
├── config/                          # 6 files
├── core/
│   ├── base_service.py
│   ├── export_service.py
│   ├── service_name.py
│   ├── neural_network/
│   │   ├── pipeline/               # 7 files
│   │   └── prompts/                # 15+ files across 4 subdirs
│   ├── snapping/                   # 10+ files across 3 subdirs
│   └── video/                      # 4 files across 2 subdirs
└── main.py
```

### TARGET (Clean, like people-demographics)
```
src/
├── config/
│   ├── arguments.py                # CLI parsing (keep)
│   └── system_configuration.py     # Simplified, merge config_data_classes
├── video/
│   └── video_source.py             # ThreadedHostNode (merge providers)
├── nn/
│   └── detection_node.py           # ThreadedHostNode (NN + filter + annotation)
├── tracking/
│   └── tracking_node.py            # ThreadedHostNode (tracker only)
├── prompts/
│   ├── prompts_node.py             # ThreadedHostNode (frame cache + encoding)
│   ├── encoders.py                 # Both encoders in one file
│   └── services.py                 # All prompt services + handlers merged
├── snapping/
│   ├── snapping_node.py            # ThreadedHostNode (producer + uploader)
│   ├── conditions.py               # All conditions in one file
│   └── snapping_service.py         # Keep separate (frontend API)
├── services/
│   └── export_service.py           # Keep (or move to main)
└── main.py                         # Clear pipeline with ThreadedHostNodes
```

---

## Detailed Refactoring Plan

### 1. VIDEO SOURCE NODE

**Current:** `VideoFactory` + `BaseVideoProvider` + `CameraVideoProvider` + `ReplayVideoProvider` (4 files)

**Target:** Single `VideoSourceNode` (ThreadedHostNode)

```python
# video/video_source.py
class VideoSourceNode(dai.node.ThreadedHostNode):
    """
    High-level node for video capture (camera or replay).

    Subnodes:
      - Camera or ReplayManip (based on config)
      - VideoEncoder (H.264)

    Outputs:
      - preview: BGR frames for NN
      - encoded: H.264 stream for visualization
    """

    def __init__(self, use_replay: bool = False):
        self._use_replay = use_replay
        # Create subnodes based on mode
        if use_replay:
            self._source = self.createSubnode(dai.node.ImageManip)
        else:
            self._camera = self.createSubnode(dai.node.Camera)
        self._encoder = self.createSubnode(dai.node.VideoEncoder)

    def build(self, config) -> "VideoSourceNode":
        # Configure camera or replay, encoder
        return self
```

**Files to delete:** `video_factory.py`, `video_providers/` folder

---

### 2. DETECTION NODE (NN Pipeline)

**Current:** `NNPipelineBuilder` + `NnNodeFactory` + `DetectionGraphFactory` + `AnnotationNode` + `LabelManager` + `PromptControllerFactory` (6+ files)

**Target:** Single `DetectionNode` (ThreadedHostNode) with internal classes

```python
# nn/detection_node.py

class PromptController:
    """Handles sending prompts to NN. Internal class."""
    def __init__(self, nn_node, det_filter, annotation_nodes, precision):
        # Setup queues, parser reference
        pass

    def send_prompts_pair(self, visual, text, classes, offset):
        # Send to queues + update labels directly (no LabelManager)
        pass

    def update_labels(self, label_names, offset):
        # Was LabelManager.update_labels - now inline
        self._det_filter.setLabels([...])
        self._annotation_extended.set_label_encoding({...})
        self._annotation_standard.set_label_encoding({...})


class DetectionNode(dai.node.ThreadedHostNode):
    """
    High-level node for object detection with YOLO-E.

    Subnodes:
      - ParsingNeuralNetwork (YOLO-E)
      - ImgDetectionsFilter
      - ImgDetectionsBridge
      - AnnotationNode (2x: extended + standard format)

    Outputs:
      - detections_extended: ImgDetectionsExtended with labels
      - detections_standard: ImgDetections with labels

    Exposes:
      - controller: PromptController for dynamic prompts
    """

    def __init__(self):
        # Subnodes
        self._nn = self.createSubnode(ParsingNeuralNetwork)
        self._det_filter = self.createSubnode(ImgDetectionsFilter)
        self._bridge = self.createSubnode(ImgDetectionsBridge)
        self._annotation_extended = self.createSubnode(AnnotationNode)
        self._annotation_standard = self.createSubnode(AnnotationNode)

        # Internal controller (not a subnode, just a helper class)
        self.controller: PromptController = None

        # Outputs
        self.detections_extended: dai.Node.Output = None
        self.detections_standard: dai.Node.Output = None

    def build(self, video_source, nn_config) -> "DetectionNode":
        # Wire up all subnodes
        # Create controller
        self.controller = PromptController(
            self._nn, self._det_filter,
            self._annotation_extended, self._annotation_standard,
            nn_config.model.precision
        )
        return self
```

**Why PromptController is internal, not separate file:**
- Like `PersonFaceAssociator` and `ReIdManager` in people-demographics
- It's tightly coupled to DetectionNode internals
- No reason for it to exist independently

**Files to delete:**
- `nn_node_factory.py`
- `detection_graph_factory.py`
- `prompt_controller_factory.py`
- `label_manager.py`
- `model_state.py` (merge into PromptController)

**Files to move:** `annotation_node.py` stays but as subnode used internally

---

### 3. TRACKING NODE - SEPARATE FROM DETECTION

**Question you raised:** Should tracker be inside DetectionNode or separate?

**Recommendation: SEPARATE TrackingNode**

**Reasons FOR separate:**
1. **People-demographics does this** - `PeopleTrackingNode` is its own ThreadedHostNode
2. **Single responsibility** - Detection and tracking are distinct concerns
3. **Reusability** - Tracking can work with different detection sources
4. **Clarity in main.py** - You see: Detection → Tracking → Snapping

**Reasons AGAINST (if you wanted it inside):**
- Fewer nodes in main.py
- But: makes DetectionNode do too much

```python
# tracking/tracking_node.py
class TrackingNode(dai.node.ThreadedHostNode):
    """
    Object tracking node.

    Subnodes:
      - ObjectTracker

    Inputs:
      - detections
      - video frame

    Output:
      - tracklets
    """
    def __init__(self, tracker_config):
        self._tracker = self.createSubnode(dai.node.ObjectTracker)
        self.out = self._tracker.out

    def build(self, detections, video_source, config) -> "TrackingNode":
        # Configure tracker
        return self
```

**Files to delete:** `tracker_factory.py`

---

### 4. PROMPTS NODE

**Current:** `NNPromptsManager` + `PromptEncodersManager` + `HandlersFactory` + `PromptServiceFactory` + `FrameCacheNode` + 4 services + 4 payloads + 3 handlers + 3 encoders (15+ files!)

**Target:** 3 files max

```python
# prompts/encoders.py
class TextualEncoder:
    """CLIP text encoder."""
    def __init__(self, config): ...
    def extract_embeddings(self, class_names) -> np.ndarray: ...
    def make_dummy(self) -> np.ndarray: ...

class VisualEncoder:
    """SAM visual encoder."""
    def __init__(self, config): ...
    def extract_embeddings(self, image, mask=None) -> np.ndarray: ...
    def make_dummy(self) -> np.ndarray: ...


# prompts/services.py
# All services + their handlers merged into one file
# Each service is a simple class, handlers become methods

class ClassUpdateService:
    def __init__(self, controller, text_encoder):
        self._controller = controller
        self._encoder = text_encoder

    def handle(self, payload) -> dict:
        # Validate with pydantic inline
        embeddings = self._encoder.extract_embeddings(payload["classes"])
        dummy = self._encoder.make_dummy()
        self._controller.send_prompts_pair(dummy, embeddings, payload["classes"], offset)
        return {"ok": True, "classes": payload["classes"]}

class ThresholdUpdateService:
    def handle(self, payload) -> dict:
        self._controller.set_confidence_threshold(payload["threshold"])
        return {"ok": True}

class ImageUploadService:
    # Similar pattern...

class BBoxPromptService:
    # Similar pattern, uses frame_cache


# prompts/prompts_node.py
class PromptsNode(dai.node.ThreadedHostNode):
    """
    High-level node for prompt handling.

    Subnode:
      - FrameCacheNode (caches frames for bbox prompts)

    Exposes:
      - services: List of prompt services to register
    """

    def __init__(self):
        self._frame_cache = self.createSubnode(FrameCacheNode)
        self._text_encoder: TextualEncoder = None
        self._visual_encoder: VisualEncoder = None
        self.services: list = []

    def build(self, video_source, controller, config) -> "PromptsNode":
        self._text_encoder = TextualEncoder(config)
        self._visual_encoder = VisualEncoder(config)

        # Create services (handlers are now internal to services)
        self.services = [
            ClassUpdateService(controller, self._text_encoder),
            ThresholdUpdateService(controller),
            ImageUploadService(controller, self._visual_encoder),
            BBoxPromptService(controller, self._visual_encoder, self._frame_cache),
        ]

        # Send initial prompts
        text_prompt = self._text_encoder.extract_embeddings(config.class_names)
        image_prompt = self._text_encoder.make_dummy()
        controller.send_prompts_pair(image_prompt, text_prompt, config.class_names, config.text_offset)

        return self

    def register_services(self, visualizer):
        for service in self.services:
            visualizer.registerService(service.name, service.handle)
```

**Why HandlersFactory is useless:**
- It just returns 3 handler instances
- The handlers can be instantiated directly in services
- Or become methods on the services themselves

**Why PromptServiceFactory is useless:**
- It just returns a list of 4 services
- Can be done inline in PromptsNode.build()

**Files to delete:**
- `handlers_factory.py`
- `prompt_service_factory.py`
- `prompt_encoders_manager.py`
- `handlers/` folder (merge into services)
- `front_end_prompt_services/` folder (merge into services.py)
- `payloads/` folder (inline pydantic models in services.py)

---

### 5. SNAPPING NODE

**Current:** `SnappingServiceManager` + `SnapsProducer` + `ConditionsFactory` + `SnappingService` + conditions folder (10+ files)

**Target:** 3 files

```python
# snapping/conditions.py
# All conditions in one file

class Condition(ABC):
    """Base condition."""
    # ... keep existing logic

class TimedCondition(Condition):
    """Triggers at regular intervals."""

class NoDetectionsCondition(Condition):
    """Triggers when no detections."""

class LowConfidenceCondition(Condition):
    """Triggers on low confidence detections."""

class LostMidCondition(Condition):
    """Triggers when tracklet lost in center."""

class TrackletAnalyzer:
    """Helper for tracklet analysis."""

def build_conditions(config) -> dict[str, Condition]:
    """Factory function (not a class)."""
    # No dynamic import needed - all conditions are in this file
    conditions = {}
    for entry in config.conditions:
        if entry.type == "timed":
            conditions["timed"] = TimedCondition(...)
        elif entry.type == "no_detections":
            conditions["no_detections"] = NoDetectionsCondition(...)
        # etc.
    return conditions


# snapping/snapping_service.py (keep separate - it's a frontend API)
class SnappingService:
    """Handles frontend snapping configuration."""
    # Keep as-is


# snapping/snapping_node.py
class SnappingNode(dai.node.ThreadedHostNode):
    """
    High-level node for data snapping/collection.

    Subnodes:
      - SnapsProducer (HostNode)
      - SnapsUploader (HostNode)

    Exposes:
      - service: SnappingService for frontend
      - conditions: dict for export
    """

    def __init__(self):
        self._producer = self.createSubnode(SnapsProducer)
        self._uploader = self.createSubnode(SnapsUploader)
        self._conditions: dict = {}
        self.service: SnappingService = None

    def build(self, video_source, tracker, detections, config) -> "SnappingNode":
        self._conditions = build_conditions(config)
        # Wire producer and uploader
        self.service = SnappingService(self._conditions)
        return self

    @property
    def conditions(self):
        return self._conditions
```

**Why ConditionsFactory should be a function, not a class:**
- It's stateless
- Has only one static method
- A simple function is clearer

**Files to delete:**
- `conditions_factory.py` (becomes function in conditions.py)
- `conditions/` subfolder (merge all into conditions.py)
- `front_end_config_service/` folder (snapping_service.py moves up)
- `snaps_manager.py` (becomes SnappingNode)

---

### 6. EXPORT SERVICE - KEEP AS-IS OR MERGE

**Options:**

**Option A: Keep as separate service (visible in main)**
```python
# In main.py
export_service = ExportService(detection_node.controller.model_state, snapping_node.conditions)
visualizer.registerService(export_service.name, export_service.handle)
```

**Option B: Make it part of a "state" object**
- Create a simple function that gathers state from nodes
- Register inline in main

**Recommendation: Option A** - It's actually consistent with how MonitorFacesNode exposes `visualizer_get_payload` in people-demographics. The export service aggregates state from multiple nodes, so it makes sense as a standalone in main.py.

---

### 7. CONFIG SIMPLIFICATION

**Current:** 6 files in config/

**Target:** 2-3 files

```python
# config/arguments.py (keep)
def initialize_argparser(): ...

# config/system_configuration.py (simplified)
@dataclass
class VideoConfig:
    fps: int
    width: int
    height: int
    media_path: Optional[str]

@dataclass
class NNConfig:
    model_path: str
    precision: str
    # ...

@dataclass
class SystemConfig:
    video: VideoConfig
    nn: NNConfig
    prompts: Box  # Keep Box for complex yaml
    snapping: Box

def build_configuration(platform: str, args) -> SystemConfig:
    # Load yamls, merge CLI args, return config
    pass
```

**Files to delete:**
- `cli_env_loader.py` (merge into system_configuration.py)
- `config_data_classes.py` (merge into system_configuration.py)
- `yaml_config_manager.py` (inline yaml loading)
- `model_loader.py` (inline model loading)

---

## Target main.py

```python
import depthai as dai
from config.system_configuration import build_configuration
from config.arguments import initialize_argparser

from video.video_source import VideoSourceNode
from nn.detection_node import DetectionNode
from tracking.tracking_node import TrackingNode
from prompts.prompts_node import PromptsNode
from snapping.snapping_node import SnappingNode
from services.export_service import ExportService


def main():
    _, args = initialize_argparser()
    device = dai.Device()
    visualizer = dai.RemoteConnection(serveFrontend=False)

    platform = device.getPlatformAsString()
    if platform != "RVC4":
        raise ValueError("This example is supported only on RVC4 platform")

    config = build_configuration(platform, args)

    with dai.Pipeline(device) as pipeline:
        # Video source
        video_node = pipeline.create(VideoSourceNode).build(config.video)
        visualizer.addTopic("Video", video_node.encoded)

        # Detection (NN + filter + annotation)
        detection_node = pipeline.create(DetectionNode).build(
            video_source=video_node.preview,
            nn_config=config.nn,
        )
        visualizer.addTopic("Annotations", detection_node.detections_extended)

        # Tracking
        tracking_node = pipeline.create(TrackingNode).build(
            detections=detection_node.detections_standard,
            video_source=video_node.preview,
            config=config.nn.tracker,
        )

        # Prompts (encoders + services)
        prompts_node = pipeline.create(PromptsNode).build(
            video_source=video_node.preview,
            controller=detection_node.controller,
            config=config.prompts,
        )
        prompts_node.register_services(visualizer)

        # Snapping
        snapping_node = pipeline.create(SnappingNode).build(
            video_source=video_node.preview,
            tracker=tracking_node.out,
            detections=detection_node.detections_standard,
            config=config.snapping,
        )
        visualizer.registerService(snapping_node.service.name, snapping_node.service.handle)

        # Export service
        export_service = ExportService(
            detection_node.controller.model_state,
            snapping_node.conditions
        )
        visualizer.registerService(export_service.name, export_service.handle)

        # Run
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            pipeline.processTasks()
            visualizer.waitKey(1)


if __name__ == "__main__":
    main()
```

---

## File Count Comparison

| Area | Current | Target |
|------|---------|--------|
| config/ | 6 | 2 |
| video/ | 4 | 1 |
| nn/ | 7 | 1-2 |
| prompts/ | 15+ | 3 |
| snapping/ | 10+ | 3 |
| services/ | 3 | 1 |
| **Total** | **45+** | **~12** |

---

## Summary of Key Decisions

1. **Tracker: SEPARATE NODE** - Follows people-demographics pattern, single responsibility

2. **LabelManager: DELETE** - Becomes a simple method in PromptController

3. **HandlersFactory: DELETE** - Handlers merge into services

4. **PromptServiceFactory: DELETE** - Service creation moves to PromptsNode.build()

5. **ConditionsFactory: CONVERT TO FUNCTION** - No need for class with single static method

6. **All conditions: ONE FILE** - No dynamic import needed, all defined together

7. **Services visible in main: YES for SnappingService and ExportService** - They're frontend APIs, should be visible

8. **Prompt services: HIDDEN** - Registered via prompts_node.register_services()

---

## Implementation Order

1. Start with `VideoSourceNode` (simplest)
2. Then `TrackingNode` (also simple)
3. Then `DetectionNode` with `PromptController` (moderate complexity)
4. Then `SnappingNode` with conditions (moderate)
5. Then `PromptsNode` with services (most complex)
6. Finally, simplify config/ and update main.py
