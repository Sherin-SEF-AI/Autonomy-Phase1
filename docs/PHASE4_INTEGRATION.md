# Phase 4 Integration Guide

## Overview
This document describes how Phase 4 features are integrated into the main UI.

## New Features Added

### 1. Recording System
- **Location**: `recording/` module
- **Components**:
  - `RecordingManager`: Coordinates video and metadata recording
  - `MultiCameraVideoWriter`: Writes synchronized multi-camera video
  - `MetadataWriter`: Records perception data to JSON
  - `PlaybackManager`: Replays recorded sessions

- **UI Integration**:
  - Menu: `Recording` → `Start Recording`, `Stop Recording`, `Open Session`, `Export Session`
  - Status bar: Recording indicator
  - Control panel: Record button with status indicator

### 2. Safety Warning Systems
- **Location**: `safety/` module
- **Components**:
  - `ForwardCollisionWarning`: TTC-based collision detection
  - `LaneDepartureWarning`: Lane departure monitoring
  - `BlindSpotWarning`: Blind spot detection
  - `SafetyMonitor`: Unified safety coordination

- **UI Integration**:
  - Menu: `Safety` → `Enable FCW`, `Enable LDW`, `Enable BSW`, `Safety Dashboard`
  - Status bar: Safety status indicator (SAFE/CAUTION/WARNING/CRITICAL)
  - Control panel: Safety alert display

### 3. Telemetry Dashboard
- **Location**: `ui/telemetry_dashboard.py`
- **Features**:
  - Real-time FPS graphs (per camera)
  - Processing latency visualization
  - CPU/Memory usage monitoring
  - Detection and tracking statistics
  - Safety system status

- **UI Integration**:
  - Menu: `View` → `Telemetry Dashboard`
  - Dockable window or separate dialog

### 4. Data Export Tools
- **Location**: `utils/data_export.py`
- **Formats**:
  - CSV: Detections, tracking, lane data
  - JSON: Complete session data
  - Video: Annotated clips, multi-camera grid

- **UI Integration**:
  - Menu: `File` → `Export Data` → `Export CSV`, `Export JSON`, `Export Video`
  - Export dialog with format selection

## Menu Structure

```
File
├── New Session
├── Open Session...
├── Save Session...
├── ───────────
├── Export Data
│   ├── Export CSV...
│   ├── Export JSON...
│   └── Export Video...
├── ───────────
└── Exit

Cameras
├── Camera Settings...
├── Discover Cameras
├── ───────────
└── Test Cameras

Recording
├── Start Recording
├── Stop Recording
├── Pause Recording
├── ───────────
├── Open Recorded Session...
├── Playback Controls...
└── Export Session...

Safety
├── Enable FCW
├── Enable LDW
├── Enable BSW
├── ───────────
├── Safety Settings...
└── Safety Dashboard...

View
├── Full Screen
├── ───────────
├── Telemetry Dashboard
├── Safety Status Panel
├── Minimap (BEV)
└── ───────────
└── Reset Layout

Help
└── About
```

## Integration Steps

### Step 1: Import Phase 4 Modules

```python
# Recording
from recording import RecordingManager, RecordingConfig, RecordingStatus
from recording import PlaybackManager, PlaybackStatus

# Safety
from safety import SafetyMonitor, SafetyStatus
from safety import FCWConfig, LDWConfig, BSWConfig

# UI Components
from ui.telemetry_dashboard import TelemetryDashboard
from visualization.minimap import MinimapWidget

# Export
from utils.data_export import DataExporter
```

### Step 2: Initialize Components in MainWindow.__init__

```python
def __init__(self):
    super().__init__()

    # ... existing initialization ...

    # Phase 4: Recording
    self.recording_manager = RecordingManager()
    self.is_recording = False

    # Phase 4: Safety
    self.safety_monitor = SafetyMonitor(
        enable_fcw=True,
        enable_ldw=True,
        enable_bsw=True
    )

    # Phase 4: Telemetry
    self.telemetry_dashboard = None  # Created on demand

    # Phase 4: Export
    self.data_exporter = DataExporter(Path("data/exports"))
```

### Step 3: Add Menu Actions

```python
def _create_menus(self):
    # ... existing menus ...

    # Recording Menu
    recording_menu = menubar.addMenu("&Recording")

    start_recording_action = QAction("&Start Recording", self)
    start_recording_action.triggered.connect(self._on_start_recording)
    recording_menu.addAction(start_recording_action)

    # ... more actions ...

    # Safety Menu
    safety_menu = menubar.addMenu("&Safety")

    # ... safety actions ...

    # Update View Menu
    telemetry_action = QAction("&Telemetry Dashboard", self)
    telemetry_action.triggered.connect(self._on_show_telemetry)
    view_menu.addAction(telemetry_action)
```

### Step 4: Connect to Perception Pipeline

```python
def _on_perception_result(self, result: PerceptionResult):
    # ... existing overlay rendering ...

    # Phase 4: Update safety monitor
    if self.safety_monitor:
        safety_status = self.safety_monitor.update(
            tracked_objects=result.tracked_objects,
            lane_info=result.lane_info.get(0)  # Front camera
        )
        self._update_safety_status(safety_status)

    # Phase 4: Record data if recording
    if self.is_recording and self.recording_manager:
        self.recording_manager.record_perception_data(
            timestamp=result.timestamp,
            frame_number=result.frame_number,
            camera_detections=result.detections_by_camera,
            tracked_objects=result.tracked_objects,
            lane_info=result.lane_info,
            fused_detections=result.fused_detections
        )

    # Phase 4: Update telemetry
    if self.telemetry_dashboard:
        self._update_telemetry(result)
```

## Performance Considerations

1. **Recording Impact**: Video recording adds ~10-20ms per frame
2. **Safety Monitoring**: <5ms per frame
3. **Telemetry Updates**: Use 1-2 Hz update rate for graphs
4. **Memory Usage**: Recording session data kept in memory until finalized

## Configuration Files

Phase 4 adds the following configuration options:

```json
{
  "recording": {
    "enabled": true,
    "output_dir": "data/recordings",
    "video_codec": "mp4v",
    "video_fps": 30,
    "record_metadata": true
  },
  "safety": {
    "fcw_enabled": true,
    "ldw_enabled": true,
    "bsw_enabled": true,
    "ttc_warning_threshold": 2.0,
    "ldw_offset_threshold": 0.5
  },
  "telemetry": {
    "enabled": true,
    "update_rate_hz": 2,
    "history_length_seconds": 60
  }
}
```

## Testing

1. **Recording**: Start system, start recording, run for 30s, stop, verify video files
2. **Safety**: Drive with objects ahead, verify FCW triggers
3. **Telemetry**: Open dashboard, verify real-time graphs update
4. **Export**: Record session, export to CSV/JSON, verify data integrity

## Known Limitations

1. Recording requires significant disk space (~1GB per 5 minutes at 4 cameras)
2. Safety warnings currently use simplified distance estimation
3. Telemetry dashboard may impact performance if update rate too high
4. Playback doesn't yet support real-time perception re-processing
