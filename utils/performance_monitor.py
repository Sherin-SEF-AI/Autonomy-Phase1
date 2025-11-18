"""
Performance monitoring utilities for the autonomous vehicle perception system.

Tracks FPS, latency, CPU/memory usage, and other performance metrics.
"""

import time
import psutil
from collections import deque
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0


class FPSCounter:
    """
    Tracks frames per second for a video stream or processing pipeline.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.frame_count = 0

    def update(self):
        """Update FPS counter with a new frame."""
        self.timestamps.append(time.time())
        self.frame_count += 1

    def get_fps(self) -> float:
        """
        Calculate current FPS.

        Returns:
            Frames per second
        """
        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self.timestamps) - 1) / elapsed

    def reset(self):
        """Reset the FPS counter."""
        self.timestamps.clear()
        self.frame_count = 0


class LatencyTracker:
    """
    Tracks processing latency statistics.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize latency tracker.

        Args:
            window_size: Number of measurements to keep
        """
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)

    def add_measurement(self, latency_ms: float):
        """
        Add a latency measurement.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.latencies.append(latency_ms)

    def get_average(self) -> float:
        """Get average latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return float(np.mean(self.latencies))

    def get_max(self) -> float:
        """Get maximum latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return float(np.max(self.latencies))

    def get_percentile(self, percentile: float) -> float:
        """
        Get latency percentile.

        Args:
            percentile: Percentile to calculate (0-100)

        Returns:
            Latency at given percentile
        """
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, percentile))

    def reset(self):
        """Reset latency tracker."""
        self.latencies.clear()


class SystemMonitor:
    """
    Monitors system resource usage (CPU, memory, etc.).
    """

    def __init__(self):
        """Initialize system monitor."""
        self.process = psutil.Process()
        self.last_update = time.time()
        self.update_interval = 1.0  # Update every second

        # Initial measurement
        self._cpu_percent = 0.0
        self._memory_mb = 0.0
        self._memory_percent = 0.0

    def update(self):
        """Update system metrics."""
        current_time = time.time()

        # Only update at specified interval
        if current_time - self.last_update < self.update_interval:
            return

        try:
            # CPU usage
            self._cpu_percent = self.process.cpu_percent()

            # Memory usage
            memory_info = self.process.memory_info()
            self._memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            self._memory_percent = self.process.memory_percent()

        except Exception:
            # If process monitoring fails, use system-wide metrics
            self._cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self._memory_mb = memory.used / (1024 * 1024)
            self._memory_percent = memory.percent

        self.last_update = current_time

    def get_cpu_percent(self) -> float:
        """Get CPU usage percentage."""
        return self._cpu_percent

    def get_memory_mb(self) -> float:
        """Get memory usage in megabytes."""
        return self._memory_mb

    def get_memory_percent(self) -> float:
        """Get memory usage percentage."""
        return self._memory_percent

    def get_disk_usage(self, path: str = "/") -> Dict[str, float]:
        """
        Get disk usage statistics.

        Args:
            path: Path to check disk usage

        Returns:
            Dictionary with total, used, free, and percent
        """
        try:
            usage = psutil.disk_usage(path)
            return {
                "total_gb": usage.total / (1024 ** 3),
                "used_gb": usage.used / (1024 ** 3),
                "free_gb": usage.free / (1024 ** 3),
                "percent": usage.percent
            }
        except Exception:
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "free_gb": 0.0,
                "percent": 0.0
            }


class PerformanceMonitor:
    """
    Comprehensive performance monitor for the perception system.
    Tracks FPS, latency, and system resources.
    """

    def __init__(self):
        """Initialize performance monitor."""
        # Per-camera FPS counters
        self.fps_counters: Dict[int, FPSCounter] = {}

        # Latency tracker
        self.latency_tracker = LatencyTracker()

        # System monitor
        self.system_monitor = SystemMonitor()

        # Frame statistics
        self.frames_processed = 0
        self.frames_dropped = 0

        # Start time
        self.start_time = time.time()

    def register_camera(self, camera_id: int):
        """
        Register a camera for FPS tracking.

        Args:
            camera_id: Camera identifier
        """
        if camera_id not in self.fps_counters:
            self.fps_counters[camera_id] = FPSCounter()

    def update_camera_fps(self, camera_id: int):
        """
        Update FPS for a specific camera.

        Args:
            camera_id: Camera identifier
        """
        if camera_id not in self.fps_counters:
            self.register_camera(camera_id)
        self.fps_counters[camera_id].update()
        self.frames_processed += 1

    def add_latency(self, latency_ms: float):
        """
        Add a latency measurement.

        Args:
            latency_ms: Processing latency in milliseconds
        """
        self.latency_tracker.add_measurement(latency_ms)

    def record_frame_drop(self):
        """Record that a frame was dropped."""
        self.frames_dropped += 1

    def update_system_metrics(self):
        """Update system resource metrics."""
        self.system_monitor.update()

    def get_camera_fps(self, camera_id: int) -> float:
        """
        Get FPS for a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Current FPS
        """
        if camera_id in self.fps_counters:
            return self.fps_counters[camera_id].get_fps()
        return 0.0

    def get_all_fps(self) -> Dict[int, float]:
        """
        Get FPS for all cameras.

        Returns:
            Dictionary mapping camera ID to FPS
        """
        return {
            camera_id: counter.get_fps()
            for camera_id, counter in self.fps_counters.items()
        }

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics.

        Returns:
            PerformanceMetrics object
        """
        # Update system metrics
        self.update_system_metrics()

        # Calculate average FPS across all cameras
        all_fps = self.get_all_fps()
        avg_fps = np.mean(list(all_fps.values())) if all_fps else 0.0

        return PerformanceMetrics(
            fps=avg_fps,
            avg_latency_ms=self.latency_tracker.get_average(),
            max_latency_ms=self.latency_tracker.get_max(),
            cpu_percent=self.system_monitor.get_cpu_percent(),
            memory_mb=self.system_monitor.get_memory_mb(),
            memory_percent=self.system_monitor.get_memory_percent(),
            frames_processed=self.frames_processed,
            frames_dropped=self.frames_dropped
        )

    def get_uptime(self) -> float:
        """
        Get system uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time

    def reset(self):
        """Reset all performance counters."""
        for counter in self.fps_counters.values():
            counter.reset()
        self.latency_tracker.reset()
        self.frames_processed = 0
        self.frames_dropped = 0
        self.start_time = time.time()
