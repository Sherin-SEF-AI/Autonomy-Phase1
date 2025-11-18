"""
Camera capture module using QThread for non-blocking video capture.

Each camera runs in its own thread to maximize performance and prevent blocking.
"""

import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional
from utils.data_structures import CameraFrame, CameraConfig, CameraPosition, CameraStatus
from utils.logger import get_logger


logger = get_logger()


class CameraCaptureThread(QThread):
    """
    Thread for capturing frames from a single camera.

    Emits frames via Qt signals for thread-safe communication with the main UI thread.
    """

    # Signals
    frame_ready = pyqtSignal(CameraFrame)  # New frame available
    status_changed = pyqtSignal(int, CameraStatus)  # (camera_id, status)
    error_occurred = pyqtSignal(int, str)  # (camera_id, error_message)
    fps_updated = pyqtSignal(int, float)  # (camera_id, fps)

    def __init__(self, camera_config: CameraConfig):
        """
        Initialize camera capture thread.

        Args:
            camera_config: Configuration for this camera
        """
        super().__init__()

        self.config = camera_config
        self.camera_id = camera_config.camera_id
        self.device_index = camera_config.device_index

        # Control flags
        self._running = False
        self._paused = False

        # OpenCV capture object
        self.capture: Optional[cv2.VideoCapture] = None

        # Frame tracking
        self.frame_number = 0
        self.last_frame_time = 0.0

        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # Error handling
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10

    def run(self):
        """
        Main thread loop for capturing frames.
        """
        logger.info(f"Camera {self.camera_id} capture thread started (device: {self.device_index})")

        self._running = True
        self.status_changed.emit(self.camera_id, CameraStatus.CONNECTING)

        # Open camera
        if not self._open_camera():
            self.error_occurred.emit(self.camera_id, f"Failed to open camera device {self.device_index}")
            self.status_changed.emit(self.camera_id, CameraStatus.ERROR)
            return

        self.status_changed.emit(self.camera_id, CameraStatus.ACTIVE)

        # Main capture loop
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            try:
                frame = self._capture_frame()
                if frame is not None:
                    self.frame_ready.emit(frame)
                    self.consecutive_errors = 0
                    self._update_fps()
                else:
                    self._handle_capture_error()

            except Exception as e:
                logger.error(f"Camera {self.camera_id} capture error: {e}")
                self._handle_capture_error()

        # Cleanup
        self._close_camera()
        logger.info(f"Camera {self.camera_id} capture thread stopped")

    def _open_camera(self) -> bool:
        """
        Open the camera device.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.capture = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)

            if not self.capture.isOpened():
                logger.error(f"Camera {self.camera_id}: Failed to open device {self.device_index}")
                return False

            # Configure camera settings
            width, height = self.config.resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Apply camera settings if specified
            if self.config.exposure is not None:
                self.capture.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)

            if self.config.brightness is not None:
                self.capture.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)

            if self.config.contrast is not None:
                self.capture.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)

            if self.config.saturation is not None:
                self.capture.set(cv2.CAP_PROP_SATURATION, self.config.saturation)

            # Verify settings
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))

            logger.info(
                f"Camera {self.camera_id} opened: "
                f"{actual_width}x{actual_height} @ {actual_fps}fps "
                f"(requested: {width}x{height} @ {self.config.fps}fps)"
            )

            # Warm up camera (capture a few frames)
            for _ in range(5):
                self.capture.read()

            return True

        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Error opening device: {e}")
            return False

    def _close_camera(self):
        """Close the camera device."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.status_changed.emit(self.camera_id, CameraStatus.DISCONNECTED)

    def _capture_frame(self) -> Optional[CameraFrame]:
        """
        Capture a single frame from the camera.

        Returns:
            CameraFrame object if successful, None otherwise
        """
        if self.capture is None or not self.capture.isOpened():
            return None

        ret, frame = self.capture.read()

        if not ret or frame is None or frame.size == 0:
            return None

        # Create CameraFrame object
        timestamp = time.time()
        height, width = frame.shape[:2]

        camera_frame = CameraFrame(
            camera_id=self.camera_id,
            timestamp=timestamp,
            frame_number=self.frame_number,
            image=frame.copy(),
            width=width,
            height=height,
            camera_position=self.config.position
        )

        self.frame_number += 1
        self.last_frame_time = timestamp

        return camera_frame

    def _handle_capture_error(self):
        """Handle capture errors with reconnection attempts."""
        self.consecutive_errors += 1

        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.error(
                f"Camera {self.camera_id}: Too many consecutive errors ({self.consecutive_errors}), "
                "attempting to reconnect..."
            )
            self.status_changed.emit(self.camera_id, CameraStatus.ERROR)

            # Try to reconnect
            self._close_camera()
            time.sleep(1.0)

            if self._open_camera():
                logger.info(f"Camera {self.camera_id}: Reconnected successfully")
                self.status_changed.emit(self.camera_id, CameraStatus.ACTIVE)
                self.consecutive_errors = 0
            else:
                logger.error(f"Camera {self.camera_id}: Reconnection failed")
                self._running = False
        else:
            # Minor error, just wait a bit
            time.sleep(0.01)

    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1

        current_time = time.time()
        elapsed = current_time - self.fps_start_time

        if elapsed >= 1.0:  # Update FPS every second
            self.current_fps = self.fps_counter / elapsed
            self.fps_updated.emit(self.camera_id, self.current_fps)

            self.fps_counter = 0
            self.fps_start_time = current_time

    def stop(self):
        """Stop the capture thread gracefully."""
        logger.info(f"Stopping camera {self.camera_id} capture thread...")
        self._running = False

    def pause(self):
        """Pause frame capture."""
        self._paused = True
        logger.debug(f"Camera {self.camera_id} capture paused")

    def resume(self):
        """Resume frame capture."""
        self._paused = False
        logger.debug(f"Camera {self.camera_id} capture resumed")

    def update_config(self, config: CameraConfig):
        """
        Update camera configuration (requires restart).

        Args:
            config: New camera configuration
        """
        self.config = config
        # Note: Camera needs to be restarted for config changes to take effect
