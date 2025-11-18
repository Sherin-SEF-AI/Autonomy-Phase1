"""
Utility modules for the autonomous vehicle perception system.
"""

from .data_structures import (
    CameraPosition,
    CameraStatus,
    ObjectClass,
    WarningLevel,
    WarningType,
    CameraFrame,
    CameraConfig,
    DetectedObject,
    TrackedObject,
    LaneDetectionResult,
    PerceptionResult,
    SafetyWarning,
    SystemMetrics,
    SessionInfo
)

from .logger import PerceptionLogger, get_logger
from .coordinate_transforms import CoordinateTransformer
from .performance_monitor import (
    PerformanceMetrics,
    FPSCounter,
    LatencyTracker,
    SystemMonitor,
    PerformanceMonitor
)

__all__ = [
    # Data structures
    'CameraPosition',
    'CameraStatus',
    'ObjectClass',
    'WarningLevel',
    'WarningType',
    'CameraFrame',
    'CameraConfig',
    'DetectedObject',
    'TrackedObject',
    'LaneDetectionResult',
    'PerceptionResult',
    'SafetyWarning',
    'SystemMetrics',
    'SessionInfo',

    # Logger
    'PerceptionLogger',
    'get_logger',

    # Coordinate transforms
    'CoordinateTransformer',

    # Performance monitoring
    'PerformanceMetrics',
    'FPSCounter',
    'LatencyTracker',
    'SystemMonitor',
    'PerformanceMonitor',
]
