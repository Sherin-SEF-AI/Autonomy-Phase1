"""
Performance benchmarks for the perception system.

Tests processing speeds and resource usage of key components.
"""

import pytest
import numpy as np
import time
from typing import List

from perception import LaneDetector, ObjectDetector, CentroidTracker
from utils.data_structures import DetectedObject, CameraFrame, CameraPosition


class TestPerformanceBenchmarks:
    """Performance benchmarks for perception algorithms."""

    @pytest.fixture
    def test_image_640x480(self):
        """Create a 640x480 test image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def test_image_1920x1080(self):
        """Create a 1920x1080 test image."""
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    def test_lane_detection_speed_640x480(self, test_image_640x480, benchmark):
        """Benchmark lane detection at 640x480."""
        lane_detector = LaneDetector()

        def run_lane_detection():
            return lane_detector.detect_lanes(test_image_640x480)

        result = benchmark(run_lane_detection)

        # Performance target: < 50ms per frame
        assert benchmark.stats.stats.mean < 0.050, \
            f"Lane detection too slow: {benchmark.stats.stats.mean*1000:.1f}ms (target: <50ms)"

    def test_lane_detection_speed_1920x1080(self, test_image_1920x1080, benchmark):
        """Benchmark lane detection at 1920x1080."""
        lane_detector = LaneDetector()

        def run_lane_detection():
            return lane_detector.detect_lanes(test_image_1920x1080)

        result = benchmark(run_lane_detection)

        # Larger images take longer
        assert benchmark.stats.stats.mean < 0.150, \
            f"Lane detection too slow at 1080p: {benchmark.stats.stats.mean*1000:.1f}ms"

    def test_object_tracking_speed(self, benchmark):
        """Benchmark object tracking performance."""
        tracker = CentroidTracker()

        # Create test detections
        detections = [
            DetectedObject(
                object_id=i,
                camera_id=0,
                class_name="car",
                class_id=2,
                confidence=0.9,
                bbox=(100 + i*50, 200, 100, 100),
                position_3d=(10.0 + i, 0.0, 0.0),
                timestamp=time.time()
            )
            for i in range(10)  # 10 objects
        ]

        def run_tracking():
            return tracker.update(detections)

        result = benchmark(run_tracking)

        # Performance target: < 10ms for 10 objects
        assert benchmark.stats.stats.mean < 0.010, \
            f"Tracking too slow: {benchmark.stats.stats.mean*1000:.1f}ms (target: <10ms)"

    def test_frame_synchronization_overhead(self, benchmark):
        """Benchmark frame synchronization overhead."""
        from camera.frame_synchronizer import FrameSynchronizer

        synchronizer = FrameSynchronizer(num_cameras=4, window_size_ms=50)

        # Create test frames
        def create_frame(camera_id, timestamp):
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            return CameraFrame(
                camera_id=camera_id,
                timestamp=timestamp,
                frame_number=0,
                image=image,
                width=640,
                height=480,
                camera_position=CameraPosition.FRONT
            )

        def run_synchronization():
            base_time = time.time()
            for i in range(4):
                frame = create_frame(i, base_time + i * 0.001)  # 1ms offset
                synchronizer.add_frame(frame)
            return synchronizer.get_synchronized_frames()

        result = benchmark(run_synchronization)

        # Performance target: < 5ms overhead
        assert benchmark.stats.stats.mean < 0.005, \
            f"Sync overhead too high: {benchmark.stats.stats.mean*1000:.1f}ms (target: <5ms)"

    def test_memory_usage_camera_frame(self):
        """Test memory usage of CameraFrame objects."""
        import sys

        # Create a frame
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = CameraFrame(
            camera_id=0,
            timestamp=time.time(),
            frame_number=0,
            image=image,
            width=640,
            height=480,
            camera_position=CameraPosition.FRONT
        )

        # Calculate size
        frame_size = sys.getsizeof(frame)
        image_size = image.nbytes

        total_size_mb = (frame_size + image_size) / (1024 * 1024)

        # 640x480x3 = 921,600 bytes ≈ 0.9 MB
        assert total_size_mb < 1.5, f"Frame too large: {total_size_mb:.2f}MB"

    def test_pipeline_throughput(self, test_image_640x480):
        """Test full pipeline throughput."""
        lane_detector = LaneDetector()
        tracker = CentroidTracker()

        num_frames = 100
        start_time = time.time()

        for i in range(num_frames):
            # Simulate full pipeline
            _ = lane_detector.detect_lanes(test_image_640x480)

        elapsed = time.time() - start_time
        fps = num_frames / elapsed

        # Performance target: >15 FPS for full pipeline
        assert fps > 15, f"Pipeline too slow: {fps:.1f} FPS (target: >15 FPS)"

    @pytest.mark.slow
    def test_sustained_performance(self, test_image_640x480):
        """Test performance over extended period (stress test)."""
        lane_detector = LaneDetector()

        num_frames = 1000  # Simulate ~33 seconds at 30 FPS
        frame_times = []

        for i in range(num_frames):
            start = time.time()
            _ = lane_detector.detect_lanes(test_image_640x480)
            frame_times.append(time.time() - start)

        mean_time = np.mean(frame_times)
        std_time = np.std(frame_times)
        max_time = np.max(frame_times)

        # Check consistency
        assert mean_time < 0.050, f"Mean time too high: {mean_time*1000:.1f}ms"
        assert std_time < 0.010, f"Too much variance: {std_time*1000:.1f}ms std"
        assert max_time < 0.100, f"Outlier detected: {max_time*1000:.1f}ms max"


# Run benchmarks with: pytest tests/test_performance.py --benchmark-only -v
# Run with slow tests: pytest tests/test_performance.py --benchmark-only --runslow -v
