# Changelog

All notable changes to the Autonomous Vehicle Perception System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0] - 2025-01-19 - Complex Features & Advanced Planning

### Added

#### 🗺️ Path Planning & Navigation
- **Path Planner** (`planning/path_planner.py` - 650+ lines)
  - Multiple planning algorithms: Polynomial, Quintic, A*, RRT, Frenet
  - Lane keeping, lane change, and overtake maneuvers
  - Emergency stop trajectory planning
  - Obstacle avoidance with safety margins
  - Speed profile optimization for comfort
  - Path smoothing and validation
  - Dynamic replanning when deviations occur
  - Waypoint-based trajectory representation
  - Real-time path visualization

#### 🚦 Traffic Sign Recognition (TSR)
- **Traffic Sign Recognizer** (`perception/traffic_sign_recognition.py` - 750+ lines)
  - 40+ traffic sign types supported
  - Speed limits (20-120 km/h) with OCR capability
  - Regulatory signs: Stop, Yield, No Entry, No Parking, etc.
  - Warning signs: Curves, Crossings, Road Work, etc.
  - Informational signs: Parking, Highway, Roundabout, etc.
  - Color-based and shape-based detection
  - Multi-frame tracking for stability
  - Distance estimation to signs
  - Importance-based sign prioritization
  - Temporal smoothing to reduce jitter

#### 👁️ Driver Monitoring System (DMS)
- **Driver Monitor** (`safety/driver_monitoring.py` - 650+ lines)
  - Real-time face and eye detection
  - Eye state analysis (open, closed, drowsy)
  - Head pose estimation (pitch, yaw, roll)
  - Gaze direction tracking (forward, left, right, down, up)
  - Drowsiness detection via prolonged eye closure
  - Distraction detection via head angle and gaze
  - Yawning detection
  - Phone usage detection capability
  - Attention scoring (0-100)
  - Multi-level alerting (Low, Medium, High, Critical)
  - Real-time visualization dashboard

#### 📦 3D Object Detection
- **3D Object Detector** (`perception/object_detection_3d.py` - 700+ lines)
  - 3D bounding box generation from 2D detections + depth
  - Full 6-DOF object pose (position + rotation)
  - Standard object dimension templates
  - Geometric depth estimation fallback
  - 3D orientation estimation (8 directions)
  - Bird's eye view projection
  - 3D IoU calculation
  - Corner point computation for 3D boxes
  - Multi-view 3D visualization
  - Depth integration from depth estimation module

#### 🅿️ Parking Assist
- **Parking Assistant** (`planning/parking_assist.py` - 700+ lines)
  - Automatic parking space detection
  - Multiple parking modes: Parallel, Perpendicular, Angled (45°, 60°)
  - Gap-based spot detection between vehicles
  - Spot quality assessment (Excellent → Unsuitable)
  - Multi-point turn trajectory planning
  - Collision-free path generation
  - Steering angle calculation at each waypoint
  - Forward/Reverse gear sequencing
  - Real-time clearance monitoring
  - Trajectory visualization

#### ⚠️ Predictive Collision Warning (PCW)
- **Predictive Collision System** (`safety/predictive_collision_warning.py` - 750+ lines)
  - Multi-object trajectory prediction (5s ahead)
  - Time-to-collision (TTC) estimation
  - Collision probability calculation (0-100%)
  - 9 collision scenario types: Frontal, Rear, Side, Pedestrian, Cyclist, etc.
  - 6 risk levels: None → Imminent
  - Critical zone monitoring around vehicle
  - Vulnerable road user prioritization (pedestrians, cyclists)
  - Recommended action generation (Monitor → Emergency Brake)
  - Multi-level warning system
  - Collision point prediction

#### 🎥 Visual Odometry
- **Visual Odometry System** (`perception/visual_odometry.py` - 650+ lines)
  - Real-time camera ego-motion estimation
  - Multiple feature detectors: ORB, SIFT, FAST, AKAZE
  - Essential matrix estimation and decomposition
  - Camera pose recovery (rotation + translation)
  - Feature matching with ratio test
  - RANSAC outlier rejection
  - Trajectory reconstruction and visualization
  - Scale estimation from known object sizes
  - Speed estimation from trajectory
  - 2D/3D trajectory export
  - Drift monitoring

### Technical Highlights

- **4,850+ lines** of production-ready code across 7 modules
- Comprehensive dataclass-based architecture
- Type hints throughout all modules
- Multi-algorithm support with fallbacks
- Real-time performance optimized
- Statistics tracking in all systems
- Professional visualization capabilities
- Extensive configuration options

### Integration

All modules designed for seamless integration:
- Shared data structures (Point3D, Waypoint, etc.)
- Common coordinate systems
- Compatible with existing v1.1.0 modules
- Modular architecture for easy adoption

## [1.1.0] - 2025-01-18 - Advanced Features Release

### Added

#### 🎥 Recording & Playback System
- **Multi-Camera Video Recorder** (`recording/video_recorder.py`)
  - Synchronized multi-camera recording with H.264 encoding
  - Circular buffer for continuous recording (last N minutes in memory)
  - Event-triggered clip saving (collision warnings, hard braking)
  - Configurable pre/post-event buffers (10s before + 10s after events)
  - Automatic metadata logging (detections, tracking, warnings)
  - Background event processing thread
  - Storage management with auto-cleanup
  - Session-based recording with unique IDs

#### 🚦 Traffic Light Detection
- **Traffic Light Detector** (`perception/traffic_light_detection.py`)
  - Color-based detection using HSV segmentation
  - State classification: Red, Yellow, Green, Off
  - Confidence scoring for each detection
  - Configurable detection parameters
  - Red light violation prevention
  - ROI-based detection (focuses on upper image region)
  - Morphological filtering for noise reduction

#### 🌊 Depth Estimation
- **Monocular Depth Estimator** (`perception/depth_estimation.py`)
  - Support for MiDaS (Small, V2.1) and DPT Hybrid models
  - GPU acceleration when available
  - Fallback to simple depth estimation (no ML required)
  - Depth-at-point queries for specific coordinates
  - Depth statistics for bounding boxes (min, max, mean, median)
  - Colored depth map visualization (multiple colormaps)
  - Depth overlay on original images
  - Distance estimation with calibration support

#### 🎨 Semantic Segmentation
- **Semantic Segmenter** (`perception/semantic_segmentation.py`)
  - Support for DeepLabV3 and FCN models
  - 19-class Cityscapes-based segmentation
  - Classes: road, sidewalk, building, vehicle, person, sky, vegetation, etc.
  - Colored mask visualization
  - Drivable area extraction
  - Class percentage calculation
  - Obstacle-on-road detection
  - GPU acceleration support

#### 🌤️ Scene Recognition
- **Environmental Context Recognizer** (`perception/scene_recognition.py`)
  - **Weather Classification**: Sunny, Cloudy, Rainy, Foggy, Snowy
  - **Road Type Detection**: Highway, Urban, Residential, Rural, Parking Lot
  - **Time of Day**: Day, Night, Dusk, Dawn
  - **Lighting Conditions**: Bright, Normal, Dim, Dark
  - Visibility score calculation (0-1)
  - Temporal smoothing for stable classifications
  - Feature extraction: brightness, contrast, saturation, edge density
  - Poor visibility detection
  - Caution requirement assessment

#### 🛡️ Safety Scoring System
- **Real-time Safety Scorer** (`safety/safety_scorer.py`)
  - Overall safety score (0-100) with 5 levels: Excellent, Good, Fair, Poor, Critical
  - **Component Scores**:
    - Collision avoidance (35% weight)
    - Lane keeping (20% weight)
    - Following distance (20% weight)
    - Speed appropriateness (15% weight)
    - Environmental awareness (10% weight)
  - Event tracking: near misses, hard braking, lane departures, dangerous maneuvers
  - 2-second rule enforcement for following distance
  - Weather-adjusted speed recommendations
  - Traffic density awareness
  - Score history and trend analysis (improving/stable/worsening)
  - Event logging with severity and impact scoring

#### 📊 Analytics & Reporting
- **Trip Analyzer** (`analytics/trip_analyzer.py`)
  - Comprehensive trip statistics
  - Detection counts by class
  - Unique object tracking across trip
  - Scene type distribution percentages
  - Weather condition distribution
  - Safety score averaging and trends
  - Performance metrics (FPS, processing time)
  - JSON report generation
  - Multi-trip summary reports
  - Top detected classes analysis
  - Safety trend visualization data

### Features Summary

**Total New Files**: 7 major modules
- `recording/video_recorder.py` (~550 lines)
- `perception/traffic_light_detection.py` (~380 lines)
- `perception/depth_estimation.py` (~470 lines)
- `perception/semantic_segmentation.py` (~480 lines)
- `perception/scene_recognition.py` (~510 lines)
- `safety/safety_scorer.py` (~530 lines)
- `analytics/trip_analyzer.py` (~450 lines)

**Total New Code**: ~3,370 lines

### Capabilities Added

1. **Recording**: Multi-camera DVR with event-triggered clip saving
2. **Safety**: Traffic light detection, real-time safety scoring, near-miss detection
3. **Environment**: Weather/lighting/road type recognition, visibility assessment
4. **Depth**: 3D scene understanding, distance estimation, drivable area detection
5. **Segmentation**: Pixel-wise scene classification, obstacle detection
6. **Analytics**: Trip statistics, safety reports, performance tracking

### Integration Ready

All modules are standalone and ready for integration into the main perception pipeline.
Optional dependencies (PyTorch, torchvision) allow graceful fallback to simpler methods.

## [1.0.2] - 2025-01-18

### Added
- **Advanced Trajectory Prediction System** (`perception/trajectory_prediction.py`)
  - Linear extrapolation and polynomial fitting for future position prediction
  - Predicts object trajectories up to 3 seconds ahead with 10 discrete time steps
  - Collision risk assessment (0-1 scale) based on predicted paths
  - Multi-object collision pair detection
  - Automatic cleanup of old trajectory histories
  - Integration with perception processor for real-time predictions

- **Object Behavior Classification** (`perception/behavior_classification.py`)
  - Motion state classification: stationary, moving, stopping, starting
  - Turning behavior detection: straight, turning left/right
  - Speed change analysis: accelerating, decelerating, constant speed
  - Advanced maneuver detection: lane changes, parking, U-turns, reversing
  - Behavior confidence scoring based on tracking quality
  - 30-frame history window for robust classification

- **Scene Understanding Module** (`perception/scene_understanding.py`)
  - Traffic density classification: empty, light, moderate, heavy, congested
  - Scene type detection: highway, urban street, residential, parking lot, intersection
  - Traffic flow analysis: same direction, opposite directions, multi-directional
  - Situational complexity assessment: simple, moderate, complex, critical
  - Danger level evaluation: safe, cautious, warning, danger
  - Erratic behavior detection (hard braking, sharp turns, unusual maneuvers)
  - Pedestrian and vulnerable road user detection
  - Lane keeping quality assessment
  - Intersection detection based on traffic flow patterns
  - Trend analysis for key metrics

- **Enhanced Visualization Features**
  - Trajectory prediction arrows on camera overlays (showing 1s ahead prediction)
  - Time-to-collision (TTC) warnings with color-coded severity
  - Collision warning circles for critical situations (TTC < 1.5s)
  - Predicted trajectory visualization in bird's eye view (cyan arrows)
  - Collision risk visualization with warning circles in BEV
  - Enhanced object labels with TTC information

### Changed
- Perception processor now integrates trajectory prediction, behavior classification, and scene understanding
- Tracked objects now include predicted positions and time-to-collision estimates
- BEV generator enhanced with predicted trajectory and collision warning overlays
- Overlay renderer extended with trajectory prediction visualization toggle
- Camera overlays now show predicted movement direction for tracked objects

### Improved
- Camera reconnection logic with exponential backoff (1s → 2s → 4s → 8s → 16s → 30s max)
- Maximum reconnection attempts limit (5 attempts before giving up)
- Error log rate limiting to prevent console spam (max every 5 seconds)
- Better error messages for camera failures
- More robust camera failure handling

## [1.0.1] - 2025-01-18

### Fixed
- **PyQt6 Compatibility**: Removed deprecated `AA_EnableHighDpiScaling` and `AA_UseHighDpiPixmaps` attributes
  - These attributes were removed in PyQt6 as high DPI scaling is enabled by default
  - Application now starts successfully on all platforms
- Added system check utility to diagnose environment issues on startup
- Added `pytest-benchmark` to requirements.txt for performance testing

### Added
- `utils/system_check.py`: New utility to check system requirements
  - Verifies Python version (3.10+)
  - Checks all required packages are installed
  - Tests camera access
  - Validates directory structure
  - Checks for YOLOv8 model
- CHANGELOG.md: This file to track changes between versions

### Changed
- `main.py`: Now runs system check on startup (non-blocking)
- `requirements.txt`: Added pytest-benchmark==4.0.0

## [1.0.0] - 2025-01-18

### Initial Release - All 5 Phases Complete!

#### Phase 1: Core Infrastructure
- Multi-camera capture engine (4 cameras)
- Professional PyQt6 UI with 2x2 camera grid
- Thread-safe architecture (QThread per camera)
- Camera management and configuration
- Frame synchronization across cameras
- Performance monitoring

#### Phase 2: Basic Perception
- Lane detection (Canny, Hough, polynomial fitting)
- YOLOv8 object detection (8 classes)
- Centroid-based object tracking
- Rich overlay rendering
- Perception processing pipeline

#### Phase 3: Multi-Camera Integration
- Sensor fusion (object deduplication)
- Bird's eye view generation
- Minimap visualization widget
- Multi-camera object association

#### Phase 4: Advanced Features
- Recording and playback system
- Safety warning systems (FCW, LDW, BSW)
- Telemetry dashboard with real-time graphs
- Data export tools (CSV, JSON, Video)

#### Phase 5: Professional Polish
- Advanced camera calibration wizard
- Comprehensive testing suite
- Complete documentation (50+ page user manual)
- System polish and finalization

### Statistics
- **Total Modules**: 46 Python files
- **Total Code**: ~12,000 lines
- **Documentation**: 100+ pages
- **Tests**: 30+ unit tests + 8 performance benchmarks
- **Status**: Production Ready ✅
