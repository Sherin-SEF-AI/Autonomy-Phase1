# Advanced ADAS Features Guide

Version 1.1.0 introduces 8 powerful new modules that transform this system into a comprehensive ADAS platform.

## Table of Contents

- [Quick Start](#quick-start)
- [Module Overview](#module-overview)
- [Detailed Usage](#detailed-usage)
- [Integration Guide](#integration-guide)
- [Dependencies](#dependencies)
- [Performance](#performance)

---

## Quick Start

### Installation

```bash
# Required dependencies (already installed)
pip install opencv-python numpy PyQt6

# Optional dependencies for deep learning features
pip install torch torchvision

# Optional for better performance
pip install numba
```

### Run the Demo

```bash
python examples/advanced_features_demo.py
```

---

## Module Overview

### 1. 📹 Multi-Camera Recording (`recording/video_recorder.py`)

**Purpose**: Professional-grade multi-camera DVR system

**Key Features**:
- Synchronized multi-camera recording
- Circular buffer (keeps last N minutes in RAM)
- Event-triggered clip saving
- Automatic metadata logging

**Quick Example**:
```python
from recording.video_recorder import MultiCameraRecorder, RecordingEvent

recorder = MultiCameraRecorder()
recorder.start_session_recording("my_drive")

# Add frames
recorder.add_frame(camera_id=0, frame=image, timestamp=time.time())

# Trigger event to save clip
event = RecordingEvent(
    timestamp=time.time(),
    event_type="collision_warning",
    severity="high",
    description="Near collision"
)
recorder.trigger_event(event)

# Stop and save
recorder.stop_session_recording()
```

---

### 2. 🚦 Traffic Light Detection (`perception/traffic_light_detection.py`)

**Purpose**: Detect and classify traffic light states

**Key Features**:
- Red, Yellow, Green detection
- Confidence scoring
- Red light violation warnings

**Quick Example**:
```python
from perception.traffic_light_detection import TrafficLightDetector

detector = TrafficLightDetector()
lights = detector.detect(image)

for light in lights:
    print(f"{light.state.value} light at {light.position}")

if detector.is_red_light(lights):
    print("RED LIGHT - STOP!")
```

---

### 3. 🌊 Depth Estimation (`perception/depth_estimation.py`)

**Purpose**: Estimate depth from monocular images

**Key Features**:
- MiDaS/DPT model support (optional)
- Simple fallback (no ML required)
- Distance estimation
- Colored depth visualization

**Quick Example**:
```python
from perception.depth_estimation import DepthEstimator, DepthModel

# With deep learning (if torch available)
estimator = DepthEstimator(model_type=DepthModel.MIDAS_SMALL)

# Or simple mode (no dependencies)
estimator = DepthEstimator(model_type=DepthModel.SIMPLE)

depth_map = estimator.estimate_depth(image)
distance_m = estimator.estimate_distance(depth_map[y, x])
```

---

### 4. 🎨 Semantic Segmentation (`perception/semantic_segmentation.py`)

**Purpose**: Pixel-wise scene classification

**Key Features**:
- 19-class Cityscapes segmentation
- Drivable area extraction
- DeepLabV3/FCN support (optional)

**Quick Example**:
```python
from perception.semantic_segmentation import SemanticSegmenter

segmenter = SemanticSegmenter(model_type="deeplabv3")
mask = segmenter.segment(image)

# Extract drivable area
drivable = segmenter.extract_drivable_area(mask)

# Get percentages
percentages = segmenter.get_class_percentages(mask)
print(f"Road: {percentages['road']:.1f}%")
```

---

### 5. 🌤️ Scene Recognition (`perception/scene_recognition.py`)

**Purpose**: Understand environmental context

**Key Features**:
- Weather classification (Sunny, Rainy, Foggy, etc.)
- Road type detection (Highway, Urban, etc.)
- Time of day classification
- Visibility scoring

**Quick Example**:
```python
from perception.scene_recognition import SceneRecognizer

recognizer = SceneRecognizer()
context = recognizer.recognize(image)

print(f"Weather: {context.weather.value}")
print(f"Road: {context.road_type.value}")
print(f"Visibility: {context.visibility_score:.2f}")

if recognizer.is_poor_visibility(context):
    print("WARNING: Poor visibility!")
```

---

### 6. 🔍 Object Re-Identification (`perception/object_reidentification.py`)

**Purpose**: Track objects across cameras and occlusions

**Key Features**:
- Appearance-based matching
- Cross-camera tracking
- Occlusion handling

**Quick Example**:
```python
from perception.object_reidentification import ObjectReIdentifier

reidentifier = ObjectReIdentifier()

# Update appearance
reidentifier.update_appearance(tracked_obj, image, camera_id)

# Re-identify after occlusion
reidentifier.mark_lost(track_id=5)
new_id = reidentifier.try_reidentify(detection, image, camera_id)
```

---

### 7. 🛡️ Safety Scoring (`safety/safety_scorer.py`)

**Purpose**: Real-time driving safety assessment

**Key Features**:
- 0-100 safety score
- 5 component scores (collision, lane, following, speed, environment)
- Event tracking (near misses, hard braking, etc.)
- Trend analysis

**Quick Example**:
```python
from safety.safety_scorer import SafetyScorer

scorer = SafetyScorer()
score = scorer.calculate_score(
    tracked_objects=tracked,
    lane_result=lane_detection,
    scene_context=context,
    warnings=warnings,
    ego_speed=15.0  # m/s
)

print(f"Safety: {score.overall_score:.0f}/100")
print(f"Level: {score.safety_level.value}")
print(f"Trend: {scorer.get_trend()}")
```

**Component Weights**:
- Collision Avoidance: 35%
- Lane Keeping: 20%
- Following Distance: 20%
- Speed Appropriateness: 15%
- Environmental Awareness: 10%

---

### 8. 📊 Trip Analytics (`analytics/trip_analyzer.py`)

**Purpose**: Comprehensive trip statistics and reporting

**Key Features**:
- Detection counting by class
- Scene/weather distribution
- Safety score averaging
- JSON report generation

**Quick Example**:
```python
from analytics.trip_analyzer import TripAnalyzer

analyzer = TripAnalyzer()
trip_id = analyzer.start_trip("morning_commute")

# During trip
analyzer.update_trip(
    detections={"car": 10, "person": 3},
    scene_type="urban",
    weather="sunny",
    safety_score=85.0
)

# End and get stats
trip_stats = analyzer.end_trip()
print(f"Duration: {trip_stats.duration_seconds/60:.1f} min")
print(f"Avg safety: {trip_stats.average_safety_score:.1f}")
```

---

## Detailed Usage

### Recording Sessions

```python
from recording.video_recorder import MultiCameraRecorder, RecordingConfig
from pathlib import Path

# Configure recorder
config = RecordingConfig(
    output_dir=Path("recordings"),
    codec="mp4v",  # or "avc1" for H.264
    fps=30,
    buffer_duration_sec=300,  # 5 min buffer
    event_pre_buffer_sec=10,  # Save 10s before event
    event_post_buffer_sec=10  # Save 10s after event
)

recorder = MultiCameraRecorder(config)

# Add cameras
for camera_id in [0, 1, 2, 3]:
    recorder.add_camera(camera_id)

# Start session
session_id = recorder.start_session_recording("trip_001")

# Process frames
while recording:
    for camera_id, frame in frames.items():
        recorder.add_frame(camera_id, frame, timestamp)

# Trigger events
if collision_warning:
    event = RecordingEvent(
        timestamp=time.time(),
        event_type="collision_warning",
        severity="critical",
        description="Forward collision imminent"
    )
    recorder.trigger_event(event)

# Stop
session_info = recorder.stop_session_recording()
```

**Output Structure**:
```
data/recordings/
├── trip_001_20250118_143022/
│   ├── camera_0.mp4
│   ├── camera_1.mp4
│   ├── camera_2.mp4
│   ├── camera_3.mp4
│   └── metadata.json
└── events/
    └── 20250118_143045_collision_warning/
        ├── camera_0.mp4
        ├── camera_1.mp4
        ├── camera_2.mp4
        ├── camera_3.mp4
        └── event_info.json
```

---

### Safety Monitoring

```python
from safety.safety_scorer import SafetyScorer, SafetyLevel

scorer = SafetyScorer()

# Continuous scoring
while driving:
    score = scorer.calculate_score(
        tracked_objects=tracked,
        lane_result=lane_detection,
        scene_analysis=scene_analysis,
        scene_context=context,
        warnings=active_warnings,
        ego_speed=ego_speed_ms
    )

    # Check safety level
    if score.safety_level == SafetyLevel.CRITICAL:
        print("🚨 CRITICAL SAFETY ISSUE!")
        trigger_alert()

    if score.safety_level == SafetyLevel.POOR:
        print("⚠️ Safety degraded - exercise caution")

    # Monitor specific components
    if score.collision_avoidance_score < 50:
        print("⚠️ Collision risk high!")

    if score.following_distance_score < 60:
        print("⚠️ Following too close!")

# Get statistics
stats = scorer.get_statistics()
print(f"Session average: {stats['average_score']:.1f}")
print(f"Near misses: {stats['total_near_misses']}")
print(f"Lane departures: {stats['total_lane_departures']}")
```

---

### Trip Analytics

```python
from analytics.trip_analyzer import TripAnalyzer

analyzer = TripAnalyzer()

# Start trip
trip_id = analyzer.start_trip("commute_home")

# Update each frame
analyzer.update_trip(
    detections={"car": 5, "person": 2, "bicycle": 1},
    tracked_object_ids=[1, 2, 3, 4, 5],
    scene_type="urban_street",
    weather="cloudy",
    safety_score=82.0,
    speed_kmh=45.0,
    fps=28.5,
    processing_time_ms=35.2,
    collision_warnings=0,
    lane_departures=1,
    near_misses=0
)

# End trip
trip_stats = analyzer.end_trip()

# Print summary
print(f"Trip: {trip_stats.trip_id}")
print(f"Duration: {trip_stats.duration_seconds/60:.1f} minutes")
print(f"Total detections: {trip_stats.total_detections}")
print(f"Unique objects: {trip_stats.unique_objects_tracked}")
print(f"Average safety: {trip_stats.average_safety_score:.1f}")
print(f"Max speed: {trip_stats.max_speed_kmh:.1f} km/h")
print(f"Average FPS: {trip_stats.avg_fps:.1f}")

# Scene distribution
for scene_type, percentage in trip_stats.scene_types_distribution.items():
    print(f"  {scene_type}: {percentage:.1f}%")

# Generate multi-trip summary
summary = analyzer.generate_summary_report()
analyzer.save_summary_report("weekly_summary.json")
```

---

## Integration Guide

### Into Perception Processor

```python
# perception/perception_processor.py

from perception.traffic_light_detection import TrafficLightDetector
from perception.depth_estimation import DepthEstimator, DepthModel
from perception.scene_recognition import SceneRecognizer
from safety.safety_scorer import SafetyScorer

class PerceptionProcessor(QThread):
    def __init__(self, ...):
        # ... existing initialization ...

        # Add new modules
        self.traffic_detector = TrafficLightDetector()
        self.depth_estimator = DepthEstimator(model_type=DepthModel.SIMPLE)
        self.scene_recognizer = SceneRecognizer()
        self.safety_scorer = SafetyScorer()

    def _process_single_frame(self, frame):
        # ... existing processing ...

        # Traffic light detection
        traffic_lights = self.traffic_detector.detect(frame.image)
        result.traffic_lights = traffic_lights

        # Depth estimation
        depth_map = self.depth_estimator.estimate_depth(frame.image)
        result.depth_map = depth_map

        # Scene recognition
        scene_context = self.scene_recognizer.recognize(frame.image)
        result.scene_context = scene_context

        return result

    def _process_frames(self):
        # ... existing processing ...

        # Calculate safety score after tracking
        if tracked_objects:
            safety_score = self.safety_scorer.calculate_score(
                tracked_objects=tracked_objects,
                lane_result=lane_result,
                scene_context=scene_context,
                warnings=warnings
            )
            result.safety_score = safety_score
```

---

### Into Main Application

```python
# In your main application

from recording.video_recorder import MultiCameraRecorder
from analytics.trip_analyzer import TripAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        # ... existing initialization ...

        # Add recorder
        self.recorder = MultiCameraRecorder()

        # Add trip analyzer
        self.trip_analyzer = TripAnalyzer()

    def start_recording(self):
        # Start both recording and analytics
        session_id = self.recorder.start_session_recording()
        trip_id = self.trip_analyzer.start_trip()
        self.statusBar().showMessage(f"Recording: {session_id}")

    def stop_recording(self):
        # Stop both
        session_info = self.recorder.stop_session_recording()
        trip_stats = self.trip_analyzer.end_trip()

        # Show summary
        self.show_trip_summary(trip_stats)
```

---

## Dependencies

### Required (Always)
```
opencv-python >= 4.5.0
numpy >= 1.19.0
PyQt6 >= 6.0.0
```

### Optional (For Deep Learning)
```
torch >= 1.9.0
torchvision >= 0.10.0
```

### Installation Commands

**Minimal (no deep learning)**:
```bash
# Already installed if you installed requirements.txt
```

**Full features with deep learning**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU acceleration** (NVIDIA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance

### Benchmarks (on i7-10700K, RTX 3070)

| Module | Mode | FPS | Notes |
|--------|------|-----|-------|
| Traffic Light | Color-based | 120+ | CPU only |
| Depth | Simple | 90+ | CPU only |
| Depth | MiDaS Small | 25-30 | GPU recommended |
| Segmentation | Simple | 100+ | CPU only |
| Segmentation | DeepLabV3 | 15-20 | GPU required |
| Scene Recognition | Full | 150+ | CPU only |
| Re-ID | Histogram | 200+ | CPU only |
| Safety Scorer | Full | 300+ | CPU only |
| Recording | 4 cameras | 120+ | Async I/O |

### Optimization Tips

1. **Use simple modes** for real-time performance:
   ```python
   depth_estimator = DepthEstimator(model_type=DepthModel.SIMPLE)
   segmenter = SemanticSegmenter(model_type="simple")
   ```

2. **Enable GPU** for deep learning models:
   ```python
   depth_estimator = DepthEstimator(
       model_type=DepthModel.MIDAS_SMALL,
       enable_gpu=True
   )
   ```

3. **Selective processing**:
   ```python
   # Only process every Nth frame for heavy modules
   if frame_count % 5 == 0:
       depth_map = depth_estimator.estimate_depth(image)
   ```

4. **Async recording**:
   ```python
   # Recording is already asynchronous
   # Frames are added to queue without blocking
   recorder.add_frame(camera_id, frame, timestamp)
   ```

---

## Troubleshooting

### Q: "torch not found" error
**A**: Either install PyTorch or use simple mode:
```python
estimator = DepthEstimator(model_type=DepthModel.SIMPLE)
segmenter = SemanticSegmenter(model_type="simple")
```

### Q: Low FPS with deep learning models
**A**: Use GPU or reduce processing frequency:
```python
# GPU
estimator = DepthEstimator(model_type=DepthModel.MIDAS_SMALL, enable_gpu=True)

# Or process every 3rd frame
if frame_count % 3 == 0:
    depth = estimator.estimate_depth(image)
```

### Q: Recording files are too large
**A**: Adjust quality or resolution:
```python
config = RecordingConfig(
    codec="avc1",  # Better compression
    quality=70,  # Lower quality (0-100)
    # Or reduce resolution before recording
)
```

### Q: Safety score always low
**A**: Ensure you're passing valid data:
```python
score = scorer.calculate_score(
    tracked_objects=tracked,  # Must be list of TrackedObject
    lane_result=lane_result,  # Must be LaneDetectionResult
    scene_context=context,  # Must be SceneContext
    warnings=warnings,  # List of SafetyWarning
    ego_speed=speed_ms  # Speed in m/s (not km/h!)
)
```

---

## Examples

See `examples/advanced_features_demo.py` for a complete working example.

Run it:
```bash
python examples/advanced_features_demo.py
```

---

## API Reference

For detailed API documentation, see inline docstrings in each module:

- `recording/video_recorder.py`
- `perception/traffic_light_detection.py`
- `perception/depth_estimation.py`
- `perception/semantic_segmentation.py`
- `perception/scene_recognition.py`
- `perception/object_reidentification.py`
- `safety/safety_scorer.py`
- `analytics/trip_analyzer.py`

All modules have comprehensive docstrings with type hints.

---

## License

Same as main project.

## Support

For issues or questions, see:
- GitHub Issues: https://github.com/Sherin-SEF-AI/Autonomy-Phase1/issues
- Documentation: `docs/`
- Examples: `examples/`
