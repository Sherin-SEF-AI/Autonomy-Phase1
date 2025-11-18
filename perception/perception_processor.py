"""
Perception processor that coordinates all perception algorithms.

Integrates:
- Lane detection
- Object detection
- Object tracking
- Sensor fusion
- Overlay rendering
"""

import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, List, Optional

from utils.data_structures import (
    CameraFrame,
    CameraConfig,
    PerceptionResult,
    DetectedObject,
    TrackedObject,
    LaneDetectionResult
)
from utils.logger import get_logger
from utils.performance_monitor import PerformanceMonitor

from perception.lane_detection import LaneDetector, LaneDetectionConfig
from perception.object_detection import ObjectDetector, ObjectDetectionConfig
from perception.object_tracking import MultiCameraTracker
from perception.sensor_fusion import SensorFusion, FusionConfig
from visualization.overlay_renderer import OverlayRenderer


logger = get_logger()


class PerceptionProcessor(QThread):
    """
    Processes camera frames through perception pipeline.

    Runs lane detection, object detection, and tracking on camera frames,
    then renders overlays and emits results.
    """

    # Signals
    result_ready = pyqtSignal(int, np.ndarray, PerceptionResult)  # (camera_id, image_with_overlay, result)
    processing_error = pyqtSignal(int, str)  # (camera_id, error_message)
    statistics_updated = pyqtSignal(dict)  # Processing statistics

    def __init__(
        self,
        enable_lane_detection: bool = True,
        enable_object_detection: bool = True,
        enable_tracking: bool = True,
        enable_sensor_fusion: bool = True,
        camera_configs: Optional[Dict[int, CameraConfig]] = None,
        parent=None
    ):
        """
        Initialize perception processor.

        Args:
            enable_lane_detection: Enable lane detection
            enable_object_detection: Enable object detection
            enable_tracking: Enable object tracking
            enable_sensor_fusion: Enable sensor fusion
            camera_configs: Camera configurations for sensor fusion
            parent: Parent QObject
        """
        super().__init__(parent)

        self.enable_lane_detection = enable_lane_detection
        self.enable_object_detection = enable_object_detection
        self.enable_tracking = enable_tracking
        self.enable_sensor_fusion = enable_sensor_fusion

        # Perception modules
        self.lane_detector = LaneDetector() if enable_lane_detection else None
        self.object_detector = ObjectDetector() if enable_object_detection else None
        self.tracker = MultiCameraTracker(num_cameras=4) if enable_tracking else None
        self.sensor_fusion = None
        if enable_sensor_fusion and camera_configs:
            self.sensor_fusion = SensorFusion(camera_configs)
        self.overlay_renderer = OverlayRenderer()

        # Camera configs for sensor fusion
        self.camera_configs = camera_configs or {}

        # Frame queue
        self.frame_queue: Dict[int, CameraFrame] = {}
        self.queue_lock = False

        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()

        # Control flags
        self._running = False
        self._paused = False

        # Statistics update interval
        self.stats_update_interval = 1.0  # seconds
        self.last_stats_update = time.time()

        logger.info("Perception processor initialized with sensor fusion: {}".format(enable_sensor_fusion))

    def run(self):
        """Main processing loop."""
        logger.info("Perception processor thread started")
        self._running = True

        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            # Process available frames
            if self.frame_queue:
                self._process_frames()
            else:
                time.sleep(0.01)  # Small delay if no frames

            # Update statistics periodically
            if time.time() - self.last_stats_update >= self.stats_update_interval:
                self._emit_statistics()
                self.last_stats_update = time.time()

        logger.info("Perception processor thread stopped")

    def add_frame(self, frame: CameraFrame):
        """
        Add a frame to the processing queue.

        Args:
            frame: Camera frame to process
        """
        # Simple queue management - keep latest frame per camera
        if not self.queue_lock:
            self.frame_queue[frame.camera_id] = frame

    def _process_frames(self):
        """Process all frames in the queue."""
        # Lock queue and get frames
        self.queue_lock = True
        frames_to_process = dict(self.frame_queue)
        self.frame_queue.clear()
        self.queue_lock = False

        # Collect all detections from all cameras for fusion
        all_detections_by_camera: Dict[int, List[DetectedObject]] = {}
        frame_results: Dict[int, PerceptionResult] = {}

        # Process each frame
        for camera_id, frame in frames_to_process.items():
            try:
                start_time = time.time()

                # Run perception pipeline (detection + lane detection)
                result = self._process_single_frame(frame)
                frame_results[camera_id] = result

                # Collect detections for sensor fusion
                if camera_id in result.detections_by_camera:
                    all_detections_by_camera[camera_id] = result.detections_by_camera[camera_id]

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                result.processing_time_ms = processing_time

                # Update performance metrics
                self.perf_monitor.update_camera_fps(camera_id)
                self.perf_monitor.add_latency(processing_time)

            except Exception as e:
                logger.error(f"Error processing frame from camera {camera_id}: {e}")
                self.processing_error.emit(camera_id, str(e))

        # Apply sensor fusion if enabled and we have detections
        if self.enable_sensor_fusion and self.sensor_fusion and all_detections_by_camera:
            fused_detections = self.sensor_fusion.fuse_detections(all_detections_by_camera)

            # Update tracking with fused detections
            if self.enable_tracking and self.tracker:
                tracked = self.tracker.update(all_detections_by_camera)  # Still use per-camera for tracking

                # Update results with tracked objects (same for all cameras)
                for camera_id, result in frame_results.items():
                    result.tracked_objects = tracked

        # Render and emit results for each camera
        for camera_id, result in frame_results.items():
            try:
                frame = frames_to_process[camera_id]

                # Render overlays
                image_with_overlay = self._render_overlays(frame.image, result, camera_id)

                # Emit result
                self.result_ready.emit(camera_id, image_with_overlay, result)

            except Exception as e:
                logger.error(f"Error rendering camera {camera_id}: {e}")
                self.processing_error.emit(camera_id, str(e))

    def _process_single_frame(self, frame: CameraFrame) -> PerceptionResult:
        """
        Process a single frame through the perception pipeline.

        Args:
            frame: Camera frame

        Returns:
            PerceptionResult
        """
        result = PerceptionResult(
            timestamp=frame.timestamp,
            frame_number=frame.frame_number
        )

        # Lane detection (typically only for front camera)
        if self.enable_lane_detection and self.lane_detector and frame.camera_position.value == 1:
            lane_result = self.lane_detector.detect(
                frame.image,
                frame.camera_id,
                frame.timestamp
            )
            result.lane_detection = lane_result

        # Object detection
        detections = []
        if self.enable_object_detection and self.object_detector:
            detections = self.object_detector.detect(
                frame.image,
                frame.camera_id,
                frame.timestamp
            )
            result.detections_by_camera[frame.camera_id] = detections

        # Note: Object tracking is now done in _process_frames after sensor fusion

        return result

    def _render_overlays(
        self,
        image: np.ndarray,
        result: PerceptionResult,
        camera_id: int
    ) -> np.ndarray:
        """
        Render perception overlays on image.

        Args:
            image: Original camera image
            result: Perception result
            camera_id: Camera ID

        Returns:
            Image with overlays
        """
        # Get detections for this camera
        detections = result.detections_by_camera.get(camera_id, [])

        # Render overlays
        image_with_overlay = self.overlay_renderer.render(
            image=image,
            lane_result=result.lane_detection,
            detections=detections,
            tracked_objects=result.tracked_objects,
            warnings=result.active_warnings
        )

        return image_with_overlay

    def _emit_statistics(self):
        """Emit processing statistics."""
        stats = {
            "performance": self.perf_monitor.get_metrics().__dict__,
            "lane_detection": self.lane_detector.get_statistics() if self.lane_detector else {},
            "object_detection": self.object_detector.get_statistics() if self.object_detector else {},
            "tracking": self.tracker.get_statistics() if self.tracker else {},
            "sensor_fusion": self.sensor_fusion.get_statistics() if self.sensor_fusion else {}
        }

        self.statistics_updated.emit(stats)

    def stop(self):
        """Stop the processing thread."""
        logger.info("Stopping perception processor...")
        self._running = False

    def pause(self):
        """Pause processing."""
        self._paused = True
        logger.debug("Perception processor paused")

    def resume(self):
        """Resume processing."""
        self._paused = False
        logger.debug("Perception processor resumed")

    def set_lane_detection_enabled(self, enabled: bool):
        """Enable/disable lane detection."""
        self.enable_lane_detection = enabled
        if enabled and self.lane_detector is None:
            self.lane_detector = LaneDetector()
        logger.info(f"Lane detection {'enabled' if enabled else 'disabled'}")

    def set_object_detection_enabled(self, enabled: bool):
        """Enable/disable object detection."""
        self.enable_object_detection = enabled
        if enabled and self.object_detector is None:
            self.object_detector = ObjectDetector()
        logger.info(f"Object detection {'enabled' if enabled else 'disabled'}")

    def set_tracking_enabled(self, enabled: bool):
        """Enable/disable tracking."""
        self.enable_tracking = enabled
        if enabled and self.tracker is None:
            self.tracker = MultiCameraTracker(num_cameras=4)
        logger.info(f"Object tracking {'enabled' if enabled else 'disabled'}")

    def set_overlay_enabled(self, overlay_type: str, enabled: bool):
        """
        Enable/disable specific overlay types.

        Args:
            overlay_type: Type of overlay ('lanes', 'detections', 'tracking', 'warnings', 'info')
            enabled: Enable or disable
        """
        if overlay_type == 'lanes':
            self.overlay_renderer.toggle_lanes(enabled)
        elif overlay_type == 'detections':
            self.overlay_renderer.toggle_detections(enabled)
        elif overlay_type == 'tracking':
            self.overlay_renderer.toggle_tracking(enabled)
        elif overlay_type == 'warnings':
            self.overlay_renderer.toggle_warnings(enabled)
        elif overlay_type == 'info':
            self.overlay_renderer.toggle_info_panel(enabled)

        logger.debug(f"Overlay '{overlay_type}' {'enabled' if enabled else 'disabled'}")

    def reset_statistics(self):
        """Reset all statistics."""
        self.perf_monitor.reset()
        if self.lane_detector:
            self.lane_detector.reset()
        if self.object_detector:
            self.object_detector.reset()
        if self.tracker:
            self.tracker.reset()
        logger.info("Perception statistics reset")
