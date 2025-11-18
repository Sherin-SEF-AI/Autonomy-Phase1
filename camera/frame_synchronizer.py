"""
Frame synchronization for multi-camera system.

Ensures frames from different cameras are temporally aligned for sensor fusion.
"""

import time
from typing import Dict, List, Optional
from collections import deque
from utils.data_structures import CameraFrame
from utils.logger import get_logger


logger = get_logger()


class FrameSynchronizer:
    """
    Synchronizes frames from multiple cameras based on timestamps.

    Buffers frames from each camera and provides synchronized frame sets
    where all cameras have frames within a specified time window.
    """

    def __init__(self, num_cameras: int = 4, sync_threshold_ms: float = 50.0, buffer_size: int = 10):
        """
        Initialize frame synchronizer.

        Args:
            num_cameras: Number of cameras to synchronize
            sync_threshold_ms: Maximum time difference for frames to be considered synchronized (milliseconds)
            buffer_size: Maximum number of frames to buffer per camera
        """
        self.num_cameras = num_cameras
        self.sync_threshold = sync_threshold_ms / 1000.0  # Convert to seconds
        self.buffer_size = buffer_size

        # Frame buffers for each camera
        self.frame_buffers: Dict[int, deque] = {
            i: deque(maxlen=buffer_size) for i in range(num_cameras)
        }

        # Statistics
        self.total_synced_sets = 0
        self.total_frames_dropped = 0
        self.last_sync_time = 0.0

    def add_frame(self, frame: CameraFrame):
        """
        Add a frame to the synchronization buffer.

        Args:
            frame: CameraFrame to add
        """
        camera_id = frame.camera_id

        if camera_id not in self.frame_buffers:
            logger.warning(f"Frame from unknown camera {camera_id}, ignoring")
            return

        self.frame_buffers[camera_id].append(frame)

        # Clean old frames
        self._clean_old_frames()

    def get_synchronized_frames(self) -> Optional[Dict[int, CameraFrame]]:
        """
        Get a set of synchronized frames from all cameras.

        Returns:
            Dictionary mapping camera_id to CameraFrame, or None if synchronization not possible
        """
        # Check if all cameras have at least one frame
        if any(len(buffer) == 0 for buffer in self.frame_buffers.values()):
            return None

        # Get the oldest frame from each camera
        oldest_frames = {
            camera_id: buffer[0]
            for camera_id, buffer in self.frame_buffers.items()
            if len(buffer) > 0
        }

        if len(oldest_frames) != self.num_cameras:
            return None

        # Find the frame with the latest timestamp (slowest camera)
        latest_timestamp = max(frame.timestamp for frame in oldest_frames.values())

        # Check if all frames are within sync threshold
        synchronized = {}
        for camera_id, frame in oldest_frames.items():
            time_diff = abs(frame.timestamp - latest_timestamp)

            if time_diff <= self.sync_threshold:
                synchronized[camera_id] = frame
            else:
                # Frame is too old, remove it and try again later
                self.frame_buffers[camera_id].popleft()
                self.total_frames_dropped += 1
                return None

        # All frames are synchronized
        # Remove used frames from buffers
        for camera_id in synchronized.keys():
            self.frame_buffers[camera_id].popleft()

        self.total_synced_sets += 1
        self.last_sync_time = time.time()

        return synchronized

    def _clean_old_frames(self):
        """Remove frames that are too old to be synchronized."""
        current_time = time.time()
        max_age = 1.0  # Remove frames older than 1 second

        for camera_id, buffer in self.frame_buffers.items():
            while len(buffer) > 0:
                frame = buffer[0]
                age = current_time - frame.timestamp

                if age > max_age:
                    buffer.popleft()
                    self.total_frames_dropped += 1
                else:
                    break

    def get_buffer_sizes(self) -> Dict[int, int]:
        """
        Get current buffer sizes for all cameras.

        Returns:
            Dictionary mapping camera_id to buffer size
        """
        return {
            camera_id: len(buffer)
            for camera_id, buffer in self.frame_buffers.items()
        }

    def get_statistics(self) -> Dict[str, any]:
        """
        Get synchronization statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_synced_sets": self.total_synced_sets,
            "total_frames_dropped": self.total_frames_dropped,
            "buffer_sizes": self.get_buffer_sizes(),
            "last_sync_time": self.last_sync_time
        }

    def reset(self):
        """Reset the synchronizer and clear all buffers."""
        for buffer in self.frame_buffers.values():
            buffer.clear()
        self.total_synced_sets = 0
        self.total_frames_dropped = 0
        self.last_sync_time = 0.0

    def set_sync_threshold(self, threshold_ms: float):
        """
        Update synchronization threshold.

        Args:
            threshold_ms: New threshold in milliseconds
        """
        self.sync_threshold = threshold_ms / 1000.0
        logger.info(f"Frame sync threshold updated to {threshold_ms}ms")
