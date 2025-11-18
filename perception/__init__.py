"""
Perception modules for autonomous vehicle perception.

Provides:
- Lane detection (classical computer vision)
- Object detection (YOLOv8-based)
- Multi-object tracking
- Perception processing pipeline
"""

from .lane_detection import LaneDetector, LaneDetectionConfig
from .object_detection import ObjectDetector, ObjectDetectionConfig
from .object_tracking import CentroidTracker, MultiCameraTracker
from .perception_processor import PerceptionProcessor

__all__ = [
    'LaneDetector',
    'LaneDetectionConfig',
    'ObjectDetector',
    'ObjectDetectionConfig',
    'CentroidTracker',
    'MultiCameraTracker',
    'PerceptionProcessor',
]
