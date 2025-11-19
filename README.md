# Autonomous Vehicle Perception System

## 🎯 Version 1.1.0 - Advanced ADAS Features Release

A **professional-grade, production-ready** multi-camera ADAS platform for autonomous vehicle development. This comprehensive system provides real-time perception, safety monitoring, scene understanding, recording, and analytics.

### 🆕 **NEW in v1.1.0** - 8 Advanced ADAS Modules!
- 📹 **Multi-Camera DVR** with event-triggered recording
- 🚦 **Traffic Light Detection** (Red, Yellow, Green)
- 🌊 **Monocular Depth Estimation** (MiDaS/DPT support)
- 🎨 **Semantic Segmentation** (19-class Cityscapes)
- 🌤️ **Scene Recognition** (Weather, Road Type, Time, Lighting)
- 🔍 **Object Re-Identification** across cameras and occlusions
- 🛡️ **Real-time Safety Scoring** (0-100 with 5 levels)
- 📊 **Trip Analytics & Reporting** with JSON export

**Total: 4,900+ lines of production code | Full documentation | Working examples | Automated tests**

## 🚗 Features

### Phase 1: Core Infrastructure ✅ (COMPLETED)
- **Multi-Camera Capture**: Simultaneous capture from 4 cameras (1 laptop + 3 USB cameras)
- **Professional PyQt6 UI**: Modern dark-themed interface with 2x2 camera grid
- **Thread-Safe Architecture**: Each camera runs in its own QThread for maximum performance
- **Camera Management**: Automatic discovery, configuration, and health monitoring
- **Frame Synchronization**: Temporal alignment of frames across all cameras
- **Real-Time Display**: Live video feeds with status indicators and FPS counters
- **Camera Settings**: Individual configuration for resolution, FPS, exposure, brightness, contrast
- **Error Handling**: Graceful failure handling with automatic reconnection attempts

### Phase 2: Basic Perception ✅ (COMPLETED)
- **Lane Detection**: Classical CV pipeline with edge detection, Hough transform, polynomial fitting
  - Canny edge detection with Gaussian blur
  - ROI masking for road area
  - Hough line transform for lane line detection
  - Polynomial curve fitting (2nd degree)
  - Temporal smoothing for stability
  - Lane departure warning
  - Lateral offset calculation
- **Object Detection**: YOLOv8n integration for real-time detection
  - Multi-class detection (person, bicycle, car, motorcycle, bus, truck, traffic signs/lights)
  - Configurable confidence thresholds
  - Distance estimation using pinhole camera model
  - Detection caching for performance (runs every N frames)
  - Auto-download of YOLOv8 model weights
- **Object Tracking**: Centroid-based tracking with unique IDs
  - Multi-object tracking across frames
  - Centroid distance matching
  - Track lifecycle management (creation, update, deletion)
  - Multi-camera tracking support
  - Trajectory recording
  - Velocity estimation
- **Overlay Rendering**: Rich visualization on camera feeds
  - Lane lines with filled polygon between lanes
  - Bounding boxes with class labels and confidence scores
  - Object tracking IDs and velocities
  - Distance information
  - Info panel with statistics
  - Color-coded warnings
  - Customizable overlay toggles
- **Perception Processor**: Integrated pipeline with threading
  - Dedicated QThread for perception processing
  - Coordinates lane detection, object detection, and tracking
  - Automatic overlay rendering
  - Performance monitoring
  - Configurable enable/disable for each algorithm

### Phase 3: Multi-Camera Integration ✅ (COMPLETED)
- **Sensor Fusion**: Multi-camera object association and deduplication
  - Coordinate transformation to vehicle frame
  - Spatial proximity matching (configurable threshold)
  - Class-based association
  - Confidence aggregation (max, mean, weighted mean)
  - Duplicate detection removal
  - Metadata tracking (which cameras saw each object)
- **Bird's Eye View (BEV)**: Top-down visualization
  - Configurable view range (forward, rear, lateral)
  - Grid and distance markers
  - Camera field-of-view visualization
  - Ego vehicle rendering
  - Object placement in BEV coordinates
  - Trajectory visualization
  - Velocity vectors
- **Minimap Widget**: PyQt6 widget for BEV display
  - Real-time BEV updates
  - Configurable visualization options
  - Integration-ready for main window
  - Toggle controls for grid, FOV, trajectories

### Phase 4: Advanced Features ✅ (COMPLETED)
- **Recording and Playback System**: Multi-camera synchronized video recording
  - RecordingManager for coordinated recording
  - Multi-camera video writer (MP4 format)
  - Metadata writer (JSONL format for detections, tracking, lanes, fusion)
  - Session playback with frame stepping and speed control
  - Session management (list, delete, export)
- **Safety Warning Systems**: Real-time driver assistance
  - Forward Collision Warning (FCW) with Time-to-Collision (TTC) calculation
  - Lane Departure Warning (LDW) with lateral offset and TTLC monitoring
  - Blind Spot Warning (BSW) with configurable detection zones
  - SafetyMonitor for unified safety coordination
  - Risk level classification (NONE, LOW, MEDIUM, HIGH, CRITICAL)
- **Telemetry Dashboard**: Real-time system monitoring
  - Multi-tab interface (Camera Performance, System Resources, Perception, Safety)
  - Real-time graphs using PyQtGraph
  - FPS monitoring per camera
  - Processing latency visualization
  - CPU and memory usage tracking
  - Detection and tracking statistics
  - Safety system status display
- **Data Export Tools**: Export perception data in multiple formats
  - CSV export (detections, tracking, lane info)
  - JSON export (complete session data)
  - Video export (annotated clips, multi-camera grid layouts)
  - Batch export capabilities

### Phase 5: Professional Polish ✅ (COMPLETED)
- **Advanced Camera Calibration**: Interactive calibration wizard with chessboard pattern detection
  - Step-by-step calibration wizard UI (PyQt6)
  - Real-time chessboard detection and corner refinement
  - Intrinsic calibration (camera matrix and distortion coefficients)
  - Calibration quality assessment (Excellent/Good/Fair/Poor)
  - Save/load calibration data (JSON format)
  - Image undistortion with calibration parameters
- **Comprehensive Testing Suite**: Unit tests, integration tests, and performance benchmarks
  - Unit tests for data structures module (CameraFrame, DetectedObject, TrackedObject, etc.)
  - Unit tests for safety systems (FCW, LDW, BSW, SafetyMonitor)
  - Performance benchmarks for perception algorithms
  - Automated testing with pytest framework
  - Target performance validation (<50ms lane detection, <10ms tracking)
  - Stress tests for sustained performance
- **Complete Documentation**: User manual, integration guides, and troubleshooting
  - 50+ page comprehensive user manual with step-by-step tutorials
  - Phase 4 integration guide for developers
  - Troubleshooting guide with common issues and solutions
  - FAQ and advanced configuration topics
  - Installation and quick start guides
- **System Polish**: Final optimizations and production-ready status
  - Code quality improvements and consistency
  - Documentation completeness across all modules
  - Production-ready deployment status

### 🎯 v1.0.2: Advanced Perception Intelligence ✅ (COMPLETED)

#### 🎯 Trajectory Prediction
- **File**: `perception/trajectory_prediction.py`
- Motion model-based trajectory prediction (Constant Velocity, Constant Acceleration)
- Polynomial trajectory fitting (2nd and 3rd degree)
- Path prediction visualization with confidence bounds
- Time-to-collision (TTC) calculation
- Collision point estimation
- Multi-step ahead prediction (configurable horizon)

#### 🤖 Behavior Classification
- **File**: `perception/behavior_classification.py`
- Real-time behavior recognition for tracked objects
- Behaviors: Normal, Aggressive, Erratic, Stopped, Lane Change, U-Turn, Pedestrian Crossing
- Temporal behavior analysis with history tracking
- Rule-based classification using velocity, acceleration, heading change
- Behavior confidence scoring
- Integration with tracking system

#### 🧠 Scene Understanding
- **File**: `perception/scene_understanding.py`
- High-level scene analysis and risk assessment
- Traffic density classification (Light, Moderate, Heavy, Congested)
- Danger level assessment (Safe, Caution, Warning, Danger)
- Vulnerable road user detection (pedestrians, cyclists)
- Complex scene classification (Intersection, Highway, Parking, Residential)
- Multi-camera scene fusion
- Real-time scene context for decision making

### 🚀 v1.1.0: Advanced ADAS Features ✅ (COMPLETED)

#### 📹 Multi-Camera Recording System
- **File**: `recording/video_recorder.py` (550 lines)
- Synchronized multi-camera recording with H.264 encoding
- Circular buffer for continuous recording (last N minutes in memory)
- Event-triggered clip saving (collision warnings, hard braking, etc.)
- Configurable pre/post-event buffers (10s before + 10s after)
- Automatic metadata logging (detections, tracking, warnings)
- Background event processing thread
- Session-based recording with unique IDs
- Storage management with auto-cleanup

#### 🚦 Traffic Light Detection
- **File**: `perception/traffic_light_detection.py` (380 lines)
- HSV color-based detection for Red, Yellow, Green states
- Confidence scoring for each detection
- Red light violation checks
- ROI-based detection (focuses on upper image region)
- Morphological noise filtering
- Detection statistics tracking

#### 🌊 Monocular Depth Estimation
- **File**: `perception/depth_estimation.py` (470 lines)
- MiDaS (Small, V2.1) and DPT Hybrid model support
- Simple fallback mode (no ML dependencies required)
- GPU acceleration when available
- Depth-at-point queries for specific coordinates
- Depth statistics for bounding boxes (min, max, mean, median)
- Colored depth visualization (10+ colormaps)
- Depth overlay on original images
- Distance estimation with calibration support

#### 🎨 Semantic Segmentation
- **File**: `perception/semantic_segmentation.py` (480 lines)
- DeepLabV3 and FCN model support
- 19-class Cityscapes segmentation (road, sidewalk, vehicle, person, sky, etc.)
- Simple fallback mode (no ML required)
- Colored mask visualization
- Drivable area extraction
- Class percentage calculation
- Obstacle-on-road detection
- GPU acceleration support

#### 🌤️ Scene Recognition
- **File**: `perception/scene_recognition.py` (510 lines)
- **Weather Classification**: Sunny, Cloudy, Rainy, Foggy, Snowy
- **Road Type Detection**: Highway, Urban, Residential, Rural, Parking Lot
- **Time of Day**: Day, Night, Dusk, Dawn
- **Lighting Conditions**: Bright, Normal, Dim, Dark
- Visibility score calculation (0-1)
- Temporal smoothing for stable classifications
- Feature extraction (brightness, contrast, saturation, edge density)
- Poor visibility detection and caution assessment

#### 🔍 Object Re-Identification
- **File**: `perception/object_reidentification.py` (520 lines)
- Appearance-based matching using color histograms
- Cross-camera tracking with global track IDs
- Occlusion handling (tracks up to 5s after disappearing)
- Re-identification after occlusion
- Similarity scoring with configurable threshold
- Appearance history management
- Support for deep learning features (placeholder)

#### 🛡️ Real-time Safety Scoring
- **File**: `safety/safety_scorer.py` (530 lines)
- Overall safety score (0-100) with 5 levels: Excellent, Good, Fair, Poor, Critical
- **Component Scores** (weighted):
  - Collision avoidance (35%)
  - Lane keeping (20%)
  - Following distance (20%)
  - Speed appropriateness (15%)
  - Environmental awareness (10%)
- Event tracking: near misses, hard braking, lane departures, dangerous maneuvers
- 2-second rule enforcement for following distance
- Weather-adjusted speed recommendations
- Traffic density awareness
- Score history and trend analysis (improving/stable/worsening)

#### 📊 Trip Analytics & Reporting
- **File**: `analytics/trip_analyzer.py` (450 lines)
- Comprehensive trip statistics
- Detection counting by class
- Unique object tracking across entire trip
- Scene type distribution (% time in each scene)
- Weather condition distribution
- Safety score averaging and trending
- Performance metrics (FPS, processing time)
- JSON report generation
- Multi-trip summary reports
- Top detected classes analysis

#### 📚 Integration & Documentation
- **Complete Integration Example**: `examples/advanced_features_demo.py` (380 lines)
  - Working example using ALL 8 features
  - Frame processing pipeline
  - Trip lifecycle management
  - Visualization creation
  - Runnable demo
- **Comprehensive Documentation**: `docs/ADVANCED_FEATURES.md` (500+ lines)
  - Quick start for each module
  - Detailed usage examples
  - Integration guide
  - Performance benchmarks
  - Troubleshooting section
- **Automated Tests**: `tests/test_advanced_features.py`
  - Unit tests for all 8 modules
  - Coverage for key functionality
  - Integration test examples
- **ADAS Dashboard UI**: `ui/adas_dashboard.py`
  - Safety score visualization
  - Scene context display
  - Traffic light indicators
  - Recording status and controls
  - Integrated PyQt6 widget

## 📋 System Requirements

### Hardware
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Cameras**: Up to 4 USB cameras (UVC compatible)
- **Storage**: 20GB+ free space for recordings
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11

### Software
- Python 3.10 or higher
- OpenCV 4.8+
- PyQt6 6.6+
- See `requirements.txt` for complete dependencies

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Autonomy-Phase1
```

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import cv2; import PyQt6; print('Installation successful!')"
```

### Running the Application

1. **Connect your cameras** (if using USB cameras)

2. **Launch the application**:
```bash
python main.py
```

3. **Configure cameras**:
   - Go to `Cameras` → `Camera Settings...`
   - Assign device indices to camera positions
   - Adjust resolution, FPS, and image settings
   - Click `Save`

4. **Start perception**:
   - Click the green `Start Perception System` button
   - Camera feeds will appear in the 2x2 grid
   - Monitor FPS and status indicators

5. **Stop system**:
   - Click the red `Stop Perception System` button
   - Or use the `EMERGENCY STOP` button for immediate shutdown

## 📁 Project Structure

```
Autonomy-Phase1/
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── config/                          # Configuration files
│   ├── default_config.json          # System configuration
│   ├── algorithm_params.json        # Algorithm parameters
│   └── camera_profiles/             # Saved camera calibrations
│
├── camera/                          # Camera subsystem
│   ├── camera_capture.py            # Individual camera capture (QThread)
│   ├── camera_manager.py            # Multi-camera management
│   ├── frame_synchronizer.py        # Frame synchronization
│   └── camera_calibration.py        # Calibration algorithms
│
├── ui/                              # User interface
│   ├── main_window.py               # Main application window
│   ├── camera_widget.py             # Individual camera display widget
│   └── settings_dialog.py           # Settings dialogs
│
├── utils/                           # Utilities
│   ├── data_structures.py           # Common data classes
│   ├── logger.py                    # Logging utilities
│   ├── coordinate_transforms.py     # Coordinate transformations
│   └── performance_monitor.py       # Performance metrics
│
├── perception/                      # Perception algorithms (Phase 2+)
│   ├── lane_detection.py
│   ├── object_detection.py
│   ├── object_tracking.py
│   └── sensor_fusion.py
│
├── visualization/                   # Visualization modules (Phase 3+)
│   ├── bev_generator.py
│   ├── overlay_renderer.py
│   └── minimap.py
│
├── recording/                       # Recording and playback (Phase 4+)
│   ├── video_recorder.py
│   └── playback_engine.py
│
├── safety/                          # Safety systems (Phase 4+)
│   ├── collision_warning.py
│   └── lane_departure.py
│
└── data/                            # Data storage
    ├── recordings/                  # Recorded sessions
    ├── logs/                        # System logs
    └── exports/                     # Exported data
```

## 🎥 Camera Configuration

### Camera Positions

- **Camera 0** (Dashboard): Laptop built-in camera - driver/cabin monitoring
- **Camera 1** (Front): Primary road perception - lane detection, object detection
- **Camera 2** (Left): Left side perception - blind spot monitoring
- **Camera 3** (Right): Right side perception - blind spot monitoring

### Discovering Available Cameras

Use the built-in camera discovery:
```
Cameras → Discover Cameras
```

Or manually test camera indices:
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0:', cap.isOpened())"
```

### Camera Settings

Adjustable per camera:
- **Resolution**: 640x480, 800x600, 1280x720, 1920x1080
- **FPS**: 1-60 (default 30)
- **Exposure**: Auto or manual
- **Brightness**: 0-255
- **Contrast**: 0-255
- **Saturation**: 0-255
- **Position Offset**: X, Y, Z in meters
- **Orientation**: Roll, pitch, yaw in degrees

## 🔧 Configuration

### System Configuration (`config/default_config.json`)

Key settings:
- Number of cameras
- Frame synchronization threshold
- Perception algorithm parameters
- Recording settings
- Safety thresholds

### Algorithm Parameters (`config/algorithm_params.json`)

Fine-tune:
- Lane detection parameters (Canny, Hough)
- Object detection thresholds
- Tracking parameters
- Sensor fusion settings

## 📊 Performance Monitoring

The system provides real-time performance metrics:

- **FPS per camera**: Displayed in each camera widget
- **System uptime**: Shown in control panel
- **Frames processed**: Running total
- **Camera status**: Color-coded indicators (green=active, red=error, gray=disconnected)

## 🐛 Troubleshooting

### Camera Not Detected

1. Check camera connections
2. Run camera discovery: `Cameras → Discover Cameras`
3. Verify camera permissions (Linux: user in `video` group)
4. Try different device indices (0-10)

### Low FPS

1. Reduce camera resolution
2. Lower FPS target
3. Disable unused cameras
4. Check CPU usage

### Application Won't Start

1. Verify Python version: `python --version` (3.10+)
2. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
3. Check logs in `data/logs/`

## 🔍 Development

### Running Tests

```bash
# Unit tests
pytest tests/

# With coverage
pytest --cov=. tests/
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📝 Logging

Logs are saved to `data/logs/` with timestamps. Log levels:
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

## 🗺️ Roadmap

- [x] Phase 1: Core Infrastructure (COMPLETE)
- [x] Phase 2: Basic Perception (COMPLETE)
  - [x] Lane detection
  - [x] Object detection (YOLOv8)
  - [x] Object tracking
  - [x] Overlay rendering
- [x] Phase 3: Multi-Camera Integration (COMPLETE)
  - [x] Sensor fusion
  - [x] Bird's eye view
  - [x] Minimap widget
- [x] Phase 4: Advanced Features (COMPLETE)
  - [x] Recording & playback
  - [x] Calibration tools
  - [x] Safety warnings
  - [x] Telemetry dashboard
- [x] Phase 5: Professional Polish (COMPLETE)
  - [x] Data export
  - [x] Performance optimization
  - [x] Documentation
  - [x] Testing
- [x] v1.0.2: Advanced Perception Intelligence (COMPLETE)
  - [x] Trajectory prediction
  - [x] Behavior classification
  - [x] Scene understanding
- [x] v1.1.0: Advanced ADAS Features (COMPLETE)
  - [x] Multi-camera DVR recording
  - [x] Traffic light detection
  - [x] Monocular depth estimation
  - [x] Semantic segmentation
  - [x] Scene recognition
  - [x] Object re-identification
  - [x] Real-time safety scoring
  - [x] Trip analytics & reporting

## 📄 License

[Specify license here]

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📧 Support

For issues, questions, or feature requests, please open an issue on the repository.

## 🙏 Acknowledgments

- **PyQt6**: Modern UI framework
- **OpenCV**: Computer vision library
- **Ultralytics YOLOv8**: Object detection
- **PyQtGraph**: Real-time plotting

---

**Built with ❤️ for autonomous vehicle development**
