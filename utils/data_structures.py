"""
Common data structures for the autonomous vehicle perception system.

This module defines the core data classes used throughout the application
for representing camera frames, detections, tracking information, and system state.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import time
import numpy as np


class CameraPosition(Enum):
    """Enumeration of camera positions on the vehicle."""
    DASHBOARD = 0  # Laptop built-in camera
    FRONT = 1      # Front-facing USB camera
    LEFT = 2       # Left side USB camera
    RIGHT = 3      # Right side USB camera


class CameraStatus(Enum):
    """Camera operational status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class ObjectClass(Enum):
    """Detected object classes."""
    UNKNOWN = 0
    PERSON = 1
    BICYCLE = 2
    CAR = 3
    MOTORCYCLE = 4
    BUS = 5
    TRUCK = 6
    TRAFFIC_LIGHT = 7
    STOP_SIGN = 8
    SPEED_LIMIT_SIGN = 9
    OTHER_SIGN = 10


class WarningLevel(Enum):
    """Warning severity levels."""
    NONE = 0
    ADVISORY = 1    # > 3 seconds
    CAUTION = 2     # 1-3 seconds
    CRITICAL = 3    # < 1 second


class WarningType(Enum):
    """Types of safety warnings."""
    FORWARD_COLLISION = "forward_collision"
    LANE_DEPARTURE_LEFT = "lane_departure_left"
    LANE_DEPARTURE_RIGHT = "lane_departure_right"
    BLIND_SPOT_LEFT = "blind_spot_left"
    BLIND_SPOT_RIGHT = "blind_spot_right"
    PEDESTRIAN_COLLISION = "pedestrian_collision"
    TRAFFIC_LIGHT_VIOLATION = "traffic_light_violation"


@dataclass
class CameraFrame:
    """Container for a camera frame with metadata."""
    camera_id: int
    timestamp: float
    frame_number: int
    image: np.ndarray
    width: int
    height: int
    camera_position: CameraPosition

    def __post_init__(self):
        """Validate frame data."""
        if self.image is None or self.image.size == 0:
            raise ValueError("Frame image cannot be None or empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Frame dimensions must be positive")


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    camera_id: int
    device_index: int
    position: CameraPosition
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    enabled: bool = True

    # Camera settings
    exposure: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None

    # Calibration parameters
    camera_matrix: Optional[np.ndarray] = None
    distortion_coeffs: Optional[np.ndarray] = None

    # Extrinsic parameters (position relative to vehicle center)
    position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z in meters
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw in degrees


@dataclass
class DetectedObject:
    """Represents a detected object in a single frame."""
    object_id: Optional[int]  # Tracking ID (None for new detections)
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    camera_id: int
    timestamp: float

    # Position and motion (in vehicle coordinates if available)
    position_3d: Optional[Tuple[float, float, float]] = None  # x, y, z in meters
    velocity: Optional[Tuple[float, float]] = None  # vx, vy in m/s
    distance: Optional[float] = None  # Distance from camera in meters

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedObject:
    """Represents a tracked object across multiple frames."""
    track_id: int
    class_id: int
    class_name: str
    first_seen: float
    last_seen: float

    # Current state
    current_bbox: Tuple[int, int, int, int]
    current_position: Optional[Tuple[float, float, float]]
    current_velocity: Optional[Tuple[float, float]]
    confidence: float

    # History
    trajectory: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)  # (timestamp, position)
    camera_history: List[int] = field(default_factory=list)  # Which cameras have seen this object

    # Predictions
    predicted_position: Optional[Tuple[float, float, float]] = None
    time_to_collision: Optional[float] = None

    # Metadata
    is_stationary: bool = False
    frames_tracked: int = 0
    frames_lost: int = 0


@dataclass
class LaneDetectionResult:
    """Result of lane detection algorithm."""
    timestamp: float
    camera_id: int

    # Lane boundaries (list of points)
    left_lane: Optional[List[Tuple[int, int]]] = None
    right_lane: Optional[List[Tuple[int, int]]] = None
    center_line: Optional[List[Tuple[int, int]]] = None

    # Lane parameters
    lane_width: Optional[float] = None  # meters
    curvature: Optional[float] = None  # 1/radius in meters
    lateral_offset: Optional[float] = None  # Distance from lane center in meters

    # Polynomial coefficients (for visualization)
    left_poly: Optional[np.ndarray] = None
    right_poly: Optional[np.ndarray] = None

    # Confidence
    detection_confidence: float = 0.0

    # Warnings
    departure_warning: bool = False
    departure_direction: Optional[str] = None  # "left" or "right"


@dataclass
class PerceptionResult:
    """Complete perception result for a frame."""
    timestamp: float
    frame_number: int

    # Detections per camera
    detections_by_camera: Dict[int, List[DetectedObject]] = field(default_factory=dict)

    # Tracked objects (fused across cameras)
    tracked_objects: List[TrackedObject] = field(default_factory=list)

    # Lane detection (typically from front camera only)
    lane_detection: Optional[LaneDetectionResult] = None

    # Safety warnings
    active_warnings: List['SafetyWarning'] = field(default_factory=list)

    # Performance metrics
    processing_time_ms: float = 0.0
    fps_per_camera: Dict[int, float] = field(default_factory=dict)


@dataclass
class SafetyWarning:
    """Safety warning generated by the system."""
    warning_type: WarningType
    level: WarningLevel
    timestamp: float
    message: str

    # Related objects/data
    related_object_id: Optional[int] = None
    time_to_event: Optional[float] = None  # seconds
    distance: Optional[float] = None  # meters

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float

    # FPS per camera
    fps_camera_0: float = 0.0
    fps_camera_1: float = 0.0
    fps_camera_2: float = 0.0
    fps_camera_3: float = 0.0

    # Processing latency
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0

    # Detection statistics
    total_detections: int = 0
    total_tracked_objects: int = 0

    # Frame statistics
    frames_dropped: int = 0
    total_frames_processed: int = 0


@dataclass
class SessionInfo:
    """Information about a recording/processing session."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None

    # Configuration
    camera_configs: Dict[int, CameraConfig] = field(default_factory=dict)

    # Statistics
    total_frames: int = 0
    total_detections: int = 0
    total_warnings: int = 0

    # File paths (for recordings)
    video_files: Dict[int, str] = field(default_factory=dict)
    metadata_file: Optional[str] = None

    def duration(self) -> float:
        """Calculate session duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
