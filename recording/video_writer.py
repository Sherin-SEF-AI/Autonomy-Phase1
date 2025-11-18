"""
Multi-camera video writer for synchronized recording.

Handles writing video streams from multiple cameras to disk with
proper synchronization and codec configuration.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import threading

from utils.data_structures import CameraFrame
from utils.logger import get_logger


logger = get_logger()


@dataclass
class VideoWriterConfig:
    """Configuration for video writing."""
    codec: str = 'mp4v'  # 'mp4v', 'H264', 'XVID', 'MJPG'
    fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    quality: int = 90  # 0-100 for MJPG


class CameraVideoWriter:
    """
    Video writer for a single camera stream.

    Handles writing frames from one camera to a video file.
    """

    def __init__(
        self,
        camera_id: int,
        output_path: Path,
        config: VideoWriterConfig
    ):
        """
        Initialize camera video writer.

        Args:
            camera_id: Camera identifier
            output_path: Path to output video file
            config: Video writer configuration
        """
        self.camera_id = camera_id
        self.output_path = output_path
        self.config = config

        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*config.codec)
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            config.fps,
            (config.frame_width, config.frame_height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for camera {camera_id}")

        # Statistics
        self.frames_written = 0
        self.lock = threading.Lock()

        logger.info(f"Video writer created for camera {camera_id}: {output_path}")

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the video file.

        Args:
            frame: Frame to write (BGR format)

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Resize if needed
                if frame.shape[1] != self.config.frame_width or frame.shape[0] != self.config.frame_height:
                    frame = cv2.resize(
                        frame,
                        (self.config.frame_width, self.config.frame_height)
                    )

                self.writer.write(frame)
                self.frames_written += 1
                return True

            except Exception as e:
                logger.error(f"Error writing frame for camera {self.camera_id}: {e}")
                return False

    def release(self):
        """Release the video writer."""
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                logger.info(
                    f"Video writer released for camera {self.camera_id} "
                    f"({self.frames_written} frames written)"
                )


class MultiCameraVideoWriter:
    """
    Manages video recording for multiple cameras simultaneously.

    Ensures synchronized recording across all camera streams.
    """

    def __init__(
        self,
        output_dir: Path,
        session_name: str,
        config: Optional[VideoWriterConfig] = None
    ):
        """
        Initialize multi-camera video writer.

        Args:
            output_dir: Directory to save video files
            session_name: Name for this recording session
            config: Video writer configuration
        """
        self.output_dir = Path(output_dir)
        self.session_name = session_name
        self.config = config or VideoWriterConfig()

        # Create output directory
        self.session_dir = self.output_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Camera writers
        self.writers: Dict[int, CameraVideoWriter] = {}

        # Statistics
        self.start_time: Optional[float] = None
        self.total_frames = 0
        self.lock = threading.Lock()

        logger.info(f"Multi-camera video writer initialized: {self.session_dir}")

    def add_camera(self, camera_id: int) -> bool:
        """
        Add a camera to the recording session.

        Args:
            camera_id: Camera identifier

        Returns:
            True if successful
        """
        with self.lock:
            if camera_id in self.writers:
                logger.warning(f"Camera {camera_id} already added to recording")
                return False

            try:
                output_path = self.session_dir / f"camera_{camera_id}.mp4"
                writer = CameraVideoWriter(camera_id, output_path, self.config)
                self.writers[camera_id] = writer
                logger.info(f"Added camera {camera_id} to recording")
                return True

            except Exception as e:
                logger.error(f"Failed to add camera {camera_id}: {e}")
                return False

    def remove_camera(self, camera_id: int):
        """
        Remove a camera from the recording session.

        Args:
            camera_id: Camera identifier
        """
        with self.lock:
            if camera_id in self.writers:
                self.writers[camera_id].release()
                del self.writers[camera_id]
                logger.info(f"Removed camera {camera_id} from recording")

    def write_frame(self, camera_id: int, frame: np.ndarray) -> bool:
        """
        Write a frame for a specific camera.

        Args:
            camera_id: Camera identifier
            frame: Frame to write

        Returns:
            True if successful
        """
        if camera_id not in self.writers:
            return False

        success = self.writers[camera_id].write_frame(frame)
        if success:
            self.total_frames += 1

        return success

    def write_camera_frame(self, camera_frame: CameraFrame) -> bool:
        """
        Write a CameraFrame object.

        Args:
            camera_frame: CameraFrame to write

        Returns:
            True if successful
        """
        return self.write_frame(camera_frame.camera_id, camera_frame.image)

    def get_statistics(self) -> Dict:
        """
        Get recording statistics.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            stats = {
                'session_name': self.session_name,
                'session_dir': str(self.session_dir),
                'num_cameras': len(self.writers),
                'total_frames': self.total_frames,
                'cameras': {}
            }

            for camera_id, writer in self.writers.items():
                stats['cameras'][camera_id] = {
                    'frames_written': writer.frames_written,
                    'output_path': str(writer.output_path)
                }

            return stats

    def release_all(self):
        """Release all video writers."""
        with self.lock:
            for camera_id in list(self.writers.keys()):
                self.writers[camera_id].release()

            self.writers.clear()
            logger.info(f"All video writers released ({self.total_frames} total frames)")

    def __del__(self):
        """Destructor to ensure writers are released."""
        self.release_all()
