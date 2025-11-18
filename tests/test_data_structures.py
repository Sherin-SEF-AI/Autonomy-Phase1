"""
Unit tests for data structures module.

Tests core data classes used throughout the perception system.
"""

import pytest
import numpy as np
from datetime import datetime

from utils.data_structures import (
    CameraFrame, CameraPosition, CameraStatus, CameraConfig,
    DetectedObject, TrackedObject, LaneInfo
)


class TestCameraFrame:
    """Tests for CameraFrame data class."""

    def test_camera_frame_creation(self):
        """Test creating a CameraFrame."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = CameraFrame(
            camera_id=0,
            timestamp=12345.67,
            frame_number=100,
            image=image,
            width=640,
            height=480,
            camera_position=CameraPosition.FRONT
        )

        assert frame.camera_id == 0
        assert frame.timestamp == 12345.67
        assert frame.frame_number == 100
        assert frame.width == 640
        assert frame.height == 480
        assert frame.camera_position == CameraPosition.FRONT
        assert frame.image.shape == (480, 640, 3)

    def test_camera_frame_properties(self):
        """Test CameraFrame computed properties."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = CameraFrame(
            camera_id=1,
            timestamp=100.0,
            frame_number=50,
            image=image,
            width=640,
            height=480,
            camera_position=CameraPosition.LEFT
        )

        assert frame.resolution == (640, 480)
        assert frame.aspect_ratio == pytest.approx(640/480, rel=1e-5)


class TestCameraConfig:
    """Tests for CameraConfig data class."""

    def test_camera_config_creation(self):
        """Test creating a CameraConfig."""
        config = CameraConfig(
            camera_id=0,
            device_index=0,
            position=CameraPosition.FRONT,
            resolution=(1920, 1080),
            fps=30,
            enabled=True
        )

        assert config.camera_id == 0
        assert config.device_index == 0
        assert config.position == CameraPosition.FRONT
        assert config.resolution == (1920, 1080)
        assert config.fps == 30
        assert config.enabled is True

    def test_camera_config_defaults(self):
        """Test CameraConfig default values."""
        config = CameraConfig(
            camera_id=0,
            device_index=0,
            position=CameraPosition.FRONT
        )

        assert config.resolution == (640, 480)
        assert config.fps == 30
        assert config.enabled is True


class TestDetectedObject:
    """Tests for DetectedObject data class."""

    def test_detected_object_creation(self):
        """Test creating a DetectedObject."""
        obj = DetectedObject(
            object_id=1,
            camera_id=0,
            class_name="car",
            class_id=2,
            confidence=0.95,
            bbox=(100, 200, 150, 200),
            timestamp=12345.67
        )

        assert obj.object_id == 1
        assert obj.camera_id == 0
        assert obj.class_name == "car"
        assert obj.class_id == 2
        assert obj.confidence == 0.95
        assert obj.bbox == (100, 200, 150, 200)
        assert obj.timestamp == 12345.67

    def test_detected_object_with_3d_position(self):
        """Test DetectedObject with 3D position."""
        obj = DetectedObject(
            object_id=1,
            camera_id=0,
            class_name="person",
            class_id=0,
            confidence=0.88,
            bbox=(50, 60, 100, 200),
            position_3d=(5.0, -1.5, 0.0),
            distance=5.2,
            timestamp=100.0
        )

        assert obj.position_3d == (5.0, -1.5, 0.0)
        assert obj.distance == 5.2


class TestTrackedObject:
    """Tests for TrackedObject data class."""

    def test_tracked_object_creation(self):
        """Test creating a TrackedObject."""
        obj = TrackedObject(
            track_id=42,
            class_name="car",
            current_position=(10.0, 2.0, 0.0),
            confidence=0.9,
            age=10,
            times_seen=10,
            last_seen=12345.67
        )

        assert obj.track_id == 42
        assert obj.class_name == "car"
        assert obj.current_position == (10.0, 2.0, 0.0)
        assert obj.confidence == 0.9
        assert obj.age == 10
        assert obj.times_seen == 10
        assert obj.last_seen == 12345.67

    def test_tracked_object_trajectory(self):
        """Test TrackedObject trajectory tracking."""
        obj = TrackedObject(
            track_id=1,
            class_name="person",
            current_position=(5.0, 0.0, 0.0),
            confidence=0.85,
            age=5,
            times_seen=5,
            last_seen=100.0
        )

        # Add trajectory points
        obj.trajectory.append((100.0, (5.0, 0.0, 0.0)))
        obj.trajectory.append((101.0, (5.5, 0.1, 0.0)))
        obj.trajectory.append((102.0, (6.0, 0.2, 0.0)))

        assert len(obj.trajectory) == 3
        assert obj.trajectory[0] == (100.0, (5.0, 0.0, 0.0))
        assert obj.trajectory[-1] == (102.0, (6.0, 0.2, 0.0))

    def test_tracked_object_camera_ids(self):
        """Test TrackedObject camera ID tracking."""
        obj = TrackedObject(
            track_id=1,
            class_name="car",
            current_position=(10.0, 0.0, 0.0),
            confidence=0.9,
            age=1,
            times_seen=1,
            last_seen=100.0
        )

        obj.camera_ids.add(0)
        obj.camera_ids.add(1)
        obj.camera_ids.add(2)

        assert len(obj.camera_ids) == 3
        assert 0 in obj.camera_ids
        assert 1 in obj.camera_ids
        assert 2 in obj.camera_ids


class TestLaneInfo:
    """Tests for LaneInfo data class."""

    def test_lane_info_creation(self):
        """Test creating a LaneInfo."""
        lane_info = LaneInfo(
            left_lane_detected=True,
            right_lane_detected=True,
            left_coeffs=[0.001, -0.5, 200],
            right_coeffs=[0.001, 0.5, 400],
            lateral_offset=0.1,
            lane_width=3.5,
            departure_warning="",
            confidence=0.95
        )

        assert lane_info.left_lane_detected is True
        assert lane_info.right_lane_detected is True
        assert lane_info.left_coeffs == [0.001, -0.5, 200]
        assert lane_info.right_coeffs == [0.001, 0.5, 400]
        assert lane_info.lateral_offset == 0.1
        assert lane_info.lane_width == 3.5
        assert lane_info.confidence == 0.95

    def test_lane_info_no_detection(self):
        """Test LaneInfo with no lanes detected."""
        lane_info = LaneInfo(
            left_lane_detected=False,
            right_lane_detected=False,
            left_coeffs=None,
            right_coeffs=None,
            lateral_offset=0.0,
            lane_width=0.0,
            departure_warning="No lanes detected",
            confidence=0.0
        )

        assert lane_info.left_lane_detected is False
        assert lane_info.right_lane_detected is False
        assert lane_info.left_coeffs is None
        assert lane_info.right_coeffs is None
        assert lane_info.departure_warning == "No lanes detected"
        assert lane_info.confidence == 0.0


# Run tests with: pytest tests/test_data_structures.py -v
