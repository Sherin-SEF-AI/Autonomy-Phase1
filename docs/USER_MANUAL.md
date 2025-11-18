# Autonomous Vehicle Perception System - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [System Overview](#system-overview)
5. [User Interface Guide](#user-interface-guide)
6. [Camera Setup and Configuration](#camera-setup-and-configuration)
7. [Camera Calibration](#camera-calibration)
8. [Running the Perception System](#running-the-perception-system)
9. [Recording Sessions](#recording-sessions)
10. [Safety Warning Systems](#safety-warning-systems)
11. [Telemetry and Monitoring](#telemetry-and-monitoring)
12. [Data Export](#data-export)
13. [Troubleshooting](#troubleshooting)
14. [Advanced Topics](#advanced-topics)
15. [FAQ](#faq)

---

## 1. Introduction

The Autonomous Vehicle Perception System is a professional-grade,  multi-camera perception platform for autonomous vehicle development. It provides:

- **Real-time multi-camera capture** (up to 4 cameras)
- **Lane detection** using classical computer vision
- **Object detection** with YOLOv8
- **Multi-object tracking** across cameras
- **Sensor fusion** for object deduplication
- **Safety warnings** (FCW, LDW, BSW)
- **Recording and playback** with metadata
- **Telemetry dashboard** for performance monitoring
- **Data export** in multiple formats

### System Requirements

**Hardware:**
- Multi-core CPU (4+ cores recommended)
- 8GB RAM minimum, 16GB recommended
- Up to 4 USB cameras (UVC compatible)
- 20GB+ free disk space for recordings

**Software:**
- Python 3.10+
- Linux (Ubuntu 20.04+), macOS, or Windows 10/11
- See `requirements.txt` for Python dependencies

---

## 2. Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Autonomy-Phase1
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import cv2, PyQt6; print('Installation OK')"
```

---

## 3. Quick Start

### Launch the Application

```bash
python main.py
```

### Basic Workflow

1. **Connect Cameras**: Plug in up to 4 USB cameras
2. **Discover Cameras**: Menu → Cameras → Discover Cameras
3. **Start System**: Click "Start Perception System" button
4. **View Results**: Watch real-time detection and tracking
5. **Stop System**: Click "Stop Perception System"

---

## 4. System Overview

### Architecture

The system consists of several key modules:

```
┌─────────────────────────────────────────────────┐
│              User Interface (PyQt6)              │
├─────────────────────────────────────────────────┤
│  Camera Grid  │  Control Panel  │  Status Bar   │
├───────────────┴────────────────┴────────────────┤
│         Perception Processor (QThread)           │
├──────────────┬──────────────┬───────────────────┤
│ Lane Detect  │ Object Detect│  Object Tracking  │
├──────────────┴──────────────┴───────────────────┤
│           Sensor Fusion & BEV                    │
├──────────────────────────────────────────────────┤
│         Camera Manager (4x QThread)              │
├──────────────────────────────────────────────────┤
│            USB Cameras (1-4)                     │
└──────────────────────────────────────────────────┘
```

### Data Flow

1. **Capture**: Cameras capture frames at 30 FPS
2. **Synchronization**: Frames are temporally aligned
3. **Perception**: Lane detection, object detection, tracking
4. **Fusion**: Objects deduplicated across cameras
5. **Rendering**: Overlays drawn on camera feeds
6. **Display**: Results shown in UI with telemetry

---

## 5. User Interface Guide

### Main Window Layout

```
┌────────────────────────────────────────────────────┐
│  Menu Bar: File | Cameras | Recording | Safety    │
├──────────────────────────────────┬─────────────────┤
│                                  │                 │
│   Camera Grid (2x2)              │  Control Panel  │
│   ┌────────┬────────┐            │                 │
│   │Camera 0│Camera 1│            │  [Start/Stop]   │
│   │        │        │            │                 │
│   ├────────┼────────┤            │  Status Info    │
│   │Camera 2│Camera 3│            │                 │
│   │        │        │            │  Camera Status  │
│   └────────┴────────┘            │                 │
│                                  │                 │
├──────────────────────────────────┴─────────────────┤
│  Status Bar: System Status | FPS | Warnings        │
└────────────────────────────────────────────────────┘
```

### Camera Widget Features

Each camera widget displays:
- Live video feed
- FPS counter
- Status indicator (green=active, gray=inactive)
- Overlays:
  - Bounding boxes (objects)
  - Lane lines (front camera)
  - Tracking IDs
  - Distance information

---

## 6. Camera Setup and Configuration

### Connecting Cameras

1. **Physical Connection**:
   - Connect USB cameras to computer
   - Ensure cameras are UVC-compatible
   - Use USB 3.0 ports for best performance

2. **Camera Discovery**:
   - Menu → Cameras → Discover Cameras
   - System will detect available cameras
   - Default assignment:
     - Camera 0: Dashboard/laptop camera
     - Camera 1: Front-facing
     - Camera 2: Left-side
     - Camera 3: Right-side

### Camera Configuration

**To configure a specific camera:**

1. Right-click on camera widget
2. Select "Camera Settings"
3. Adjust parameters:
   - **Resolution**: 640x480 to 1920x1080
   - **FPS**: 15, 30, or 60
   - **Exposure**: Auto or manual
   - **Brightness/Contrast**: Adjust for lighting conditions

**Recommended Settings:**

| Environment | Resolution | FPS | Exposure |
|-------------|------------|-----|----------|
| Indoor Lab  | 640x480    | 30  | Auto     |
| Outdoor     | 1280x720   | 30  | Manual   |
| Testing     | 1920x1080  | 30  | Auto     |

### Camera Positioning

For optimal perception results:

- **Front Camera**: Center of windshield, level with horizon
- **Left Camera**: Driver's side mirror position
- **Right Camera**: Passenger side mirror position
- **Rear Camera** (optional): Center of rear windshield

---

## 7. Camera Calibration

Camera calibration is **essential** for accurate distance estimation.

### When to Calibrate

- After initial camera installation
- When camera position changes
- For maximum distance accuracy
- Before recording production data

### Calibration Process

#### Materials Needed

- Printed chessboard pattern (9x6 or 10x7)
- Rigid backing board
- Good lighting
- 10-20 minutes

**Download chessboard patterns:**
- 9x6: [OpenCV Calibration Patterns](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)

#### Step-by-Step Guide

1. **Launch Calibration Wizard**:
   - Menu → Cameras → Calibrate Camera
   - Select camera to calibrate
   - Click "Next"

2. **Configure Chessboard**:
   - Enter number of internal corners (default: 9x6)
   - Enter square size in mm (default: 25mm)
   - Click "Next"

3. **Capture Images**:
   - Hold chessboard in view of camera
   - Move to different positions and angles
   - Click "Capture" when chessboard detected
   - Capture 15-20 images from various:
     - Distances (near to far)
     - Angles (tilted, rotated)
     - Positions (center, edges, corners)
   - Click "Next" when done

4. **Calibrate**:
   - Wizard processes images automatically
   - Wait for calibration completion (30-60 seconds)
   - Review calibration quality:
     - **Excellent**: < 0.3 pixels error
     - **Good**: 0.3-0.5 pixels
     - **Fair**: 0.5-1.0 pixels
     - **Poor**: > 1.0 pixels (recalibrate)

5. **Save Calibration**:
   - Click "Save Calibration Data"
   - Choose location (default: `config/calibration/`)
   - Calibration applied automatically

### Tips for Good Calibration

✅ **Do:**
- Use flat, rigid chessboard
- Capture from many angles
- Cover entire camera FOV
- Ensure good lighting
- Keep chessboard sharp (no blur)

❌ **Don't:**
- Use bent/warped chessboard
- Capture similar angles only
- Ignore corners of FOV
- Capture in dim lighting
- Move during capture (blur)

---

## 8. Running the Perception System

### Starting the System

1. Click **"Start Perception System"** button (or press F5)
2. System initializes:
   - Cameras start capturing
   - Perception pipeline starts
   - YOLOv8 model loads
3. Status changes to "System Running"

### What You'll See

**Real-time overlays on camera feeds:**

- **Lane Lines** (front camera):
  - Green/Yellow lines for lane boundaries
  - Filled polygon between lanes
  - Lateral offset indicator

- **Object Bounding Boxes**:
  - Colored boxes around detected objects
  - Class label and confidence
  - Tracking ID (persistent across frames)
  - Distance estimate

- **Info Panel** (top-left of each camera):
  - FPS counter
  - Number of detections
  - Processing latency

### Perception Features

#### Lane Detection (Front Camera Only)

- Detects left and right lane lines
- Calculates lateral offset
- Warns of lane departure
- Works best on marked roads

#### Object Detection (All Cameras)

Detects 8 object classes:
- Person
- Bicycle
- Car
- Motorcycle
- Bus
- Truck
- Traffic light
- Stop sign

**Detection confidence:**
- Green box: > 0.8 confidence
- Yellow box: 0.6-0.8 confidence
- Red box: < 0.6 confidence (filtered by default)

#### Object Tracking

- Assigns unique IDs to detected objects
- Maintains IDs across frames
- Tracks trajectories
- Estimates velocity
- Works across multiple cameras

#### Sensor Fusion

- Deduplicates objects seen by multiple cameras
- Transforms to vehicle coordinate frame
- Combines confidence scores
- Maintains which cameras see each object

---

## 9. Recording Sessions

Recording allows you to capture perception data for later analysis.

### Starting a Recording

1. **Menu → Recording → Start Recording** (or Ctrl+R)
2. Enter session name (optional, auto-generated if blank)
3. Recording starts - red indicator in status bar
4. All cameras and perception data recorded

### What Gets Recorded

- **Video**: Synchronized multi-camera video (MP4)
- **Metadata**: Frame-by-frame perception data (JSONL):
  - Object detections per camera
  - Tracked objects
  - Lane detection results
  - Sensor fusion output

**File structure:**
```
data/recordings/session_20250118_143022/
├── camera_0.mp4           # Camera 0 video
├── camera_1.mp4           # Camera 1 video
├── camera_2.mp4           # Camera 2 video
├── camera_3.mp4           # Camera 3 video
├── metadata.json          # Session info
├── detections.jsonl       # Detection data
├── tracking.jsonl         # Tracking data
├── lanes.jsonl            # Lane data
└── fusion.jsonl           # Fusion data
```

### Stopping a Recording

1. **Menu → Recording → Stop Recording** (or Ctrl+R again)
2. Recording finalizes and saves
3. Red indicator disappears

### Playback

1. **Menu → Recording → Open Recorded Session**
2. Select session directory
3. Playback controls:
   - Play/Pause (Space)
   - Step Forward (Right Arrow)
   - Step Backward (Left Arrow)
   - Seek (drag slider)
   - Speed (0.25x to 4.0x)

### Recording Tips

- 📀 **Disk Space**: ~200MB per minute (4 cameras @ 640x480)
- ⏱️ **Duration**: Record 1-5 minute sessions for manageability
- 🎬 **Scenarios**: Record different driving conditions:
  - Highway driving
  - Urban streets
  - Parking
  - Lane changes
  - Object interactions

---

## 10. Safety Warning Systems

The system includes three active safety systems:

### Forward Collision Warning (FCW)

**Purpose**: Warns of imminent collision with object ahead

**How it works:**
- Calculates Time-to-Collision (TTC) for objects ahead
- Issues warnings based on TTC thresholds

**Warning levels:**
- 🟢 **SAFE**: TTC > 5 seconds or no threat
- 🟡 **LOW**: TTC 3.5-5 seconds
- 🟠 **MEDIUM**: TTC 2-3.5 seconds
- 🔴 **HIGH**: TTC 1-2 seconds
- 🚨 **CRITICAL**: TTC < 1 second

**Requirements:**
- Minimum vehicle speed: 18 km/h (5 m/s)
- Minimum closing velocity: 1 m/s
- Object confidence: > 50%

### Lane Departure Warning (LDW)

**Purpose**: Warns when vehicle departs from lane

**How it works:**
- Monitors lateral offset from lane center
- Calculates Time-to-Lane-Crossing (TTLC)
- Issues warnings for unintended departures

**Warning levels:**
- 🟢 **SAFE**: Centered in lane (< 30cm offset)
- 🟡 **LOW**: 30-50cm from center
- 🟠 **MEDIUM**: 50-70cm from center
- 🔴 **HIGH**: 70-90cm from center
- 🚨 **CRITICAL**: > 90cm or crossing lane line

**Requirements:**
- Minimum vehicle speed: 36 km/h (10 m/s)
- Both lane lines detected
- Lane confidence: > 60%

### Blind Spot Warning (BSW)

**Purpose**: Alerts to vehicles in blind spots

**How it works:**
- Monitors areas beside and slightly behind vehicle
- Detects vehicles in blind spot zones
- Issues warnings when attempting lane change

**Warning levels:**
- 🟢 **SAFE**: No objects in blind spots
- 🟡 **LOW**: Object approaching blind spot
- 🟠 **MEDIUM**: Object entering blind spot
- 🔴 **HIGH**: Object in blind spot
- 🚨 **CRITICAL**: Object directly beside vehicle

**Blind spot zones:**
- Left side: 1-3m lateral, -2m to +1m longitudinal
- Right side: 1-3m lateral, -2m to +1m longitudinal

### Safety System Configuration

**Enable/Disable:**
- Menu → Safety → Enable FCW/LDW/BSW

**View Safety Dashboard:**
- Menu → Safety → Safety Dashboard
- Shows real-time status of all systems
- Warning history and statistics

---

## 11. Telemetry and Monitoring

The telemetry dashboard provides real-time system monitoring.

### Opening the Dashboard

**Menu → View → Telemetry Dashboard**

### Dashboard Tabs

#### 1. Camera Performance
- FPS graph per camera (real-time)
- Processing latency graph
- Frame drop detection

#### 2. System Resources
- CPU usage graph
- Memory usage graph
- Disk space monitoring

#### 3. Perception
- Detection count graph
- Tracking count graph
- Algorithm performance metrics

#### 4. Safety
- Safety status indicator
- Warning counts (FCW, LDW, BSW)
- Alert history

### Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Camera FPS | 30 FPS | 30 FPS |
| Pipeline Latency | < 100ms | 50-80ms |
| CPU Usage | < 80% | 40-60% |
| Memory | < 2GB | 1.5GB |

### Troubleshooting Performance

**Low FPS:**
- Reduce camera resolution
- Disable unused cameras
- Close other applications

**High Latency:**
- Increase detection interval (process every N frames)
- Use YOLOv8n (nano) instead of larger models
- Reduce resolution

**High CPU:**
- Limit number of active cameras
- Reduce FPS
- Use GPU acceleration (if available)

---

## 12. Data Export

Export perception data for external analysis.

### Export Formats

#### CSV Export
- Detections: Object detections per frame
- Tracking: Tracked object trajectories
- Lanes: Lane detection data

**Menu → File → Export Data → Export CSV**

#### JSON Export
- Complete session data
- Structured format for programmatic access
- Includes metadata and configuration

**Menu → File → Export Data → Export JSON**

#### Video Export
- Annotated video clips
- Multi-camera grid layouts
- Slow-motion playback

**Menu → File → Export Data → Export Video**

### Export Examples

**CSV Structure (detections.csv):**
```csv
timestamp,object_id,camera_id,class_name,confidence,bbox_x,bbox_y,bbox_w,bbox_h,distance
12345.67,1,0,car,0.95,320,240,150,100,15.5
12345.67,2,0,person,0.88,200,300,80,180,8.2
```

**JSON Structure:**
```json
{
  "session_name": "session_20250118_143022",
  "start_time": "2025-01-18T14:30:22",
  "detections": [
    {
      "timestamp": 12345.67,
      "camera_id": 0,
      "objects": [...]
    }
  ]
}
```

---

## 13. Troubleshooting

### Common Issues

#### Cameras Not Detected

**Symptoms**: "No cameras found" message

**Solutions**:
1. Check USB connections
2. Try different USB ports (use USB 3.0)
3. Check camera drivers (Linux: `lsusb`, `v4l2-ctl --list-devices`)
4. Restart application
5. Check permissions (Linux: add user to `video` group)

#### Low FPS / Laggy Display

**Symptoms**: FPS < 20, choppy video

**Solutions**:
1. Reduce camera resolution (try 640x480)
2. Close other applications
3. Disable unused cameras
4. Increase detection interval to 3-5 frames

#### YOLOv8 Model Not Loading

**Symptoms**: "Model not found" error

**Solutions**:
1. Ensure internet connection (model auto-downloads)
2. Check `models/` directory exists
3. Manually download: `wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt`
4. Place in `models/yolov8n.pt`

#### Inaccurate Distance Estimates

**Symptoms**: Distance values seem wrong

**Solutions**:
1. **Calibrate cameras** (most important!)
2. Check camera positioning (must be level)
3. Verify calibration quality (< 0.5 pixel error)
4. Recalibrate if camera moved

#### Lane Detection Not Working

**Symptoms**: No lane lines detected

**Solutions**:
1. Ensure front camera has clear road view
2. Check lighting conditions (avoid glare)
3. Verify road has visible lane markings
4. Adjust lane detection parameters
5. Clean camera lens

---

## 14. Advanced Topics

### Configuration Files

**System Configuration: `config/default_config.json`**
```json
{
  "camera": {
    "resolution": [640, 480],
    "fps": 30
  },
  "perception": {
    "lane_detection_enabled": true,
    "object_detection_enabled": true,
    "tracking_enabled": true,
    "fusion_enabled": true
  },
  "performance": {
    "detection_interval": 2,
    "use_gpu": false
  }
}
```

**Algorithm Parameters: `config/algorithm_params.json`**
```json
{
  "lane_detection": {
    "canny_low": 50,
    "canny_high": 150,
    "hough_threshold": 50
  },
  "object_detection": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }
}
```

### Extending the System

The system is designed for extensibility:

**Add new object classes:**
1. Update `YOLO_CLASSES` in `object_detection.py`
2. Retrain or use custom YOLOv8 model

**Add new perception algorithms:**
1. Create module in `perception/`
2. Integrate into `PerceptionProcessor`
3. Add UI controls if needed

**Add new safety systems:**
1. Create module in `safety/`
2. Add to `SafetyMonitor`
3. Configure thresholds

### API Usage

The system can be used programmatically:

```python
from camera import CameraManager, CameraConfig
from perception import PerceptionProcessor

# Initialize
camera_manager = CameraManager()
perception = PerceptionProcessor()

# Configure camera
config = CameraConfig(camera_id=0, device_index=0)
camera_manager.add_camera(config)

# Start
camera_manager.start_all_cameras()
perception.start()

# Process frames
for frame in camera_manager.get_frames():
    result = perception.process_frame(frame)
    # Use result...
```

---

## 15. FAQ

**Q: How many cameras can I use?**
A: Up to 4 cameras simultaneously. More cameras increase CPU usage.

**Q: What camera resolutions are supported?**
A: 320x240 to 1920x1080. Recommended: 640x480 or 1280x720 for balance of quality and performance.

**Q: Can I use a webcam?**
A: Yes! Any UVC-compatible USB webcam works. Built-in laptop cameras work too.

**Q: Do I need a GPU?**
A: No, system runs on CPU. GPU can improve performance but isn't required.

**Q: How accurate are distance estimates?**
A: With calibration: ±10-20% for objects 5-30m away. Without calibration: ±50% or more.

**Q: Can I record multiple sessions?**
A: Yes! Each session saved separately. ~200MB per minute with 4 cameras.

**Q: Does lane detection work at night?**
A: Yes, but requires visible lane markings. Performance depends on lighting and camera quality.

**Q: Can I export data to MATLAB/Python?**
A: Yes! CSV and JSON exports work with all analysis tools.

**Q: How do I get better object detection?**
A: Use YOLOv8s or YOLOv8m models (larger but more accurate). Edit `object_detection.py` to change model.

**Q: Can I use this for my research?**
A: Yes! System designed for research and development. See LICENSE for details.

---

## Support

**Documentation**: `docs/` directory
**Issues**: GitHub Issues
**Examples**: `examples/` directory
**Tests**: `tests/` directory

**Happy Perceiving! 🚗💨**
