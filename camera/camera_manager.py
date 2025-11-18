"""
Camera manager for multi-camera system.

Manages all camera capture threads and provides a unified interface
for camera control and frame access.
"""

import cv2
from typing import Dict, List, Optional, Callable
from PyQt6.QtCore import QObject, pyqtSignal
from utils.data_structures import CameraConfig, CameraFrame, CameraStatus, CameraPosition
from utils.logger import get_logger
from camera.camera_capture import CameraCaptureThread
from camera.frame_synchronizer import FrameSynchronizer


logger = get_logger()


class CameraManager(QObject):
    """
    Manages multiple camera capture threads.

    Provides centralized control over all cameras, frame synchronization,
    and camera discovery.
    """

    # Signals
    frame_received = pyqtSignal(CameraFrame)  # New frame from any camera
    synchronized_frames_ready = pyqtSignal(dict)  # Synchronized frame set
    camera_status_changed = pyqtSignal(int, CameraStatus)  # (camera_id, status)
    camera_error = pyqtSignal(int, str)  # (camera_id, error_message)
    camera_fps_updated = pyqtSignal(int, float)  # (camera_id, fps)

    def __init__(self, enable_synchronization: bool = True):
        """
        Initialize camera manager.

        Args:
            enable_synchronization: Enable frame synchronization across cameras
        """
        super().__init__()

        self.camera_threads: Dict[int, CameraCaptureThread] = {}
        self.camera_configs: Dict[int, CameraConfig] = {}
        self.camera_statuses: Dict[int, CameraStatus] = {}

        # Frame synchronization
        self.enable_synchronization = enable_synchronization
        self.synchronizer = FrameSynchronizer(num_cameras=4) if enable_synchronization else None

        # Frame callbacks
        self.frame_callbacks: List[Callable[[CameraFrame], None]] = []

        logger.info("Camera manager initialized")

    def discover_cameras(self, max_cameras: int = 10) -> List[int]:
        """
        Discover available camera devices.

        Args:
            max_cameras: Maximum number of devices to check

        Returns:
            List of available camera device indices
        """
        logger.info(f"Discovering cameras (checking 0-{max_cameras})...")
        available_cameras = []

        for device_index in range(max_cameras):
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                # Try to read a frame to verify camera works
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(device_index)
                    logger.info(f"Found camera at device index {device_index}")
                cap.release()

        logger.info(f"Camera discovery complete: found {len(available_cameras)} cameras")
        return available_cameras

    def add_camera(self, config: CameraConfig):
        """
        Add a camera to the manager.

        Args:
            config: Camera configuration
        """
        camera_id = config.camera_id

        if camera_id in self.camera_threads:
            logger.warning(f"Camera {camera_id} already exists, stopping old instance")
            self.remove_camera(camera_id)

        # Store configuration
        self.camera_configs[camera_id] = config
        self.camera_statuses[camera_id] = CameraStatus.DISCONNECTED

        # Create and configure capture thread
        thread = CameraCaptureThread(config)

        # Connect signals
        thread.frame_ready.connect(self._on_frame_ready)
        thread.status_changed.connect(self._on_status_changed)
        thread.error_occurred.connect(self._on_error)
        thread.fps_updated.connect(self._on_fps_updated)

        self.camera_threads[camera_id] = thread

        logger.info(f"Added camera {camera_id} (device {config.device_index}, position {config.position.name})")

    def remove_camera(self, camera_id: int):
        """
        Remove a camera from the manager.

        Args:
            camera_id: Camera to remove
        """
        if camera_id in self.camera_threads:
            self.stop_camera(camera_id)
            thread = self.camera_threads.pop(camera_id)
            thread.wait()  # Wait for thread to finish
            logger.info(f"Removed camera {camera_id}")

        if camera_id in self.camera_configs:
            del self.camera_configs[camera_id]

        if camera_id in self.camera_statuses:
            del self.camera_statuses[camera_id]

    def start_camera(self, camera_id: int):
        """
        Start capturing from a specific camera.

        Args:
            camera_id: Camera to start
        """
        if camera_id not in self.camera_threads:
            logger.error(f"Cannot start camera {camera_id}: not found")
            return

        thread = self.camera_threads[camera_id]
        if not thread.isRunning():
            thread.start()
            logger.info(f"Started camera {camera_id}")
        else:
            logger.warning(f"Camera {camera_id} already running")

    def stop_camera(self, camera_id: int):
        """
        Stop capturing from a specific camera.

        Args:
            camera_id: Camera to stop
        """
        if camera_id not in self.camera_threads:
            logger.error(f"Cannot stop camera {camera_id}: not found")
            return

        thread = self.camera_threads[camera_id]
        if thread.isRunning():
            thread.stop()
            thread.wait(5000)  # Wait up to 5 seconds
            logger.info(f"Stopped camera {camera_id}")

    def start_all_cameras(self):
        """Start all configured cameras."""
        logger.info("Starting all cameras...")
        for camera_id in self.camera_threads.keys():
            self.start_camera(camera_id)

    def stop_all_cameras(self):
        """Stop all cameras."""
        logger.info("Stopping all cameras...")
        for camera_id in self.camera_threads.keys():
            self.stop_camera(camera_id)

    def pause_camera(self, camera_id: int):
        """
        Pause a camera (keep thread running but don't capture).

        Args:
            camera_id: Camera to pause
        """
        if camera_id in self.camera_threads:
            self.camera_threads[camera_id].pause()

    def resume_camera(self, camera_id: int):
        """
        Resume a paused camera.

        Args:
            camera_id: Camera to resume
        """
        if camera_id in self.camera_threads:
            self.camera_threads[camera_id].resume()

    def get_camera_status(self, camera_id: int) -> Optional[CameraStatus]:
        """
        Get the current status of a camera.

        Args:
            camera_id: Camera to query

        Returns:
            CameraStatus or None if camera not found
        """
        return self.camera_statuses.get(camera_id)

    def get_all_statuses(self) -> Dict[int, CameraStatus]:
        """
        Get status of all cameras.

        Returns:
            Dictionary mapping camera_id to CameraStatus
        """
        return self.camera_statuses.copy()

    def is_camera_active(self, camera_id: int) -> bool:
        """
        Check if a camera is actively capturing.

        Args:
            camera_id: Camera to check

        Returns:
            True if camera is active
        """
        status = self.get_camera_status(camera_id)
        return status == CameraStatus.ACTIVE

    def get_camera_config(self, camera_id: int) -> Optional[CameraConfig]:
        """
        Get configuration for a camera.

        Args:
            camera_id: Camera to query

        Returns:
            CameraConfig or None if not found
        """
        return self.camera_configs.get(camera_id)

    def update_camera_config(self, camera_id: int, config: CameraConfig):
        """
        Update camera configuration (requires restart).

        Args:
            camera_id: Camera to update
            config: New configuration
        """
        if camera_id in self.camera_threads:
            was_running = self.camera_threads[camera_id].isRunning()

            if was_running:
                self.stop_camera(camera_id)

            self.camera_configs[camera_id] = config
            self.camera_threads[camera_id].update_config(config)

            if was_running:
                self.start_camera(camera_id)

            logger.info(f"Updated configuration for camera {camera_id}")

    def register_frame_callback(self, callback: Callable[[CameraFrame], None]):
        """
        Register a callback for frame events.

        Args:
            callback: Function to call when a frame is received
        """
        self.frame_callbacks.append(callback)

    def _on_frame_ready(self, frame: CameraFrame):
        """
        Handle frame ready signal from capture thread.

        Args:
            frame: Captured frame
        """
        # Emit frame received signal
        self.frame_received.emit(frame)

        # Call registered callbacks
        for callback in self.frame_callbacks:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"Error in frame callback: {e}")

        # Add to synchronizer if enabled
        if self.enable_synchronization and self.synchronizer is not None:
            self.synchronizer.add_frame(frame)

            # Try to get synchronized frames
            synced_frames = self.synchronizer.get_synchronized_frames()
            if synced_frames is not None:
                self.synchronized_frames_ready.emit(synced_frames)

    def _on_status_changed(self, camera_id: int, status: CameraStatus):
        """
        Handle camera status change.

        Args:
            camera_id: Camera that changed status
            status: New status
        """
        self.camera_statuses[camera_id] = status
        self.camera_status_changed.emit(camera_id, status)
        logger.info(f"Camera {camera_id} status changed to {status.value}")

    def _on_error(self, camera_id: int, error_message: str):
        """
        Handle camera error.

        Args:
            camera_id: Camera with error
            error_message: Error description
        """
        self.camera_error.emit(camera_id, error_message)
        logger.error(f"Camera {camera_id} error: {error_message}")

    def _on_fps_updated(self, camera_id: int, fps: float):
        """
        Handle FPS update from camera.

        Args:
            camera_id: Camera ID
            fps: Current FPS
        """
        self.camera_fps_updated.emit(camera_id, fps)

    def get_sync_statistics(self) -> Optional[Dict]:
        """
        Get frame synchronization statistics.

        Returns:
            Statistics dictionary or None if sync disabled
        """
        if self.synchronizer is not None:
            return self.synchronizer.get_statistics()
        return None

    def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up camera manager...")
        self.stop_all_cameras()

        for thread in self.camera_threads.values():
            thread.wait()

        self.camera_threads.clear()
        self.camera_configs.clear()
        self.camera_statuses.clear()

        logger.info("Camera manager cleanup complete")
