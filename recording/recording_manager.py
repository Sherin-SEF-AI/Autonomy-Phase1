"""
Recording manager for coordinating multi-camera recording sessions.

Manages synchronized video recording and metadata logging across
all cameras and perception modules.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading
from datetime import datetime
import shutil

from utils.data_structures import CameraFrame, DetectedObject, TrackedObject, LaneInfo, CameraConfig
from utils.logger import get_logger
from .video_writer import MultiCameraVideoWriter, VideoWriterConfig
from .metadata_writer import MetadataWriter


logger = get_logger()


class RecordingStatus(Enum):
    """Recording status states."""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class RecordingConfig:
    """Configuration for recording sessions."""
    # Output settings
    output_dir: Path = Path("data/recordings")
    auto_naming: bool = True  # Auto-generate session names

    # Video settings
    video_codec: str = 'mp4v'
    video_fps: int = 30
    video_width: int = 640
    video_height: int = 480

    # Recording options
    record_video: bool = True
    record_metadata: bool = True
    record_raw_frames: bool = False  # Save individual frames as images

    # Storage management
    max_session_size_gb: float = 10.0  # Max size per session
    auto_split: bool = True  # Auto-split when reaching max size


class RecordingManager:
    """
    Manages recording sessions for the perception system.

    Coordinates video recording, metadata logging, and storage management
    across all cameras and perception modules.
    """

    def __init__(self, config: Optional[RecordingConfig] = None):
        """
        Initialize recording manager.

        Args:
            config: Recording configuration
        """
        self.config = config or RecordingConfig()

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Recording state
        self.status = RecordingStatus.IDLE
        self.current_session: Optional[str] = None
        self.session_start_time: Optional[float] = None

        # Video and metadata writers
        self.video_writer: Optional[MultiCameraVideoWriter] = None
        self.metadata_writer: Optional[MetadataWriter] = None

        # Camera configurations
        self.camera_configs: Dict[int, CameraConfig] = {}

        # Statistics
        self.total_frames_recorded = 0
        self.recording_duration = 0.0

        # Thread safety
        self.lock = threading.Lock()

        logger.info("Recording manager initialized")

    def start_recording(self, session_name: Optional[str] = None) -> bool:
        """
        Start a new recording session.

        Args:
            session_name: Optional custom session name

        Returns:
            True if recording started successfully
        """
        with self.lock:
            if self.status == RecordingStatus.RECORDING:
                logger.warning("Recording already in progress")
                return False

            try:
                # Generate session name if not provided
                if session_name is None and self.config.auto_naming:
                    session_name = self._generate_session_name()
                elif session_name is None:
                    logger.error("Session name required when auto_naming is disabled")
                    return False

                self.current_session = session_name
                self.session_start_time = datetime.now().timestamp()

                # Initialize video writer
                if self.config.record_video:
                    video_config = VideoWriterConfig(
                        codec=self.config.video_codec,
                        fps=self.config.video_fps,
                        frame_width=self.config.video_width,
                        frame_height=self.config.video_height
                    )
                    self.video_writer = MultiCameraVideoWriter(
                        self.config.output_dir,
                        session_name,
                        video_config
                    )

                # Initialize metadata writer
                if self.config.record_metadata:
                    self.metadata_writer = MetadataWriter(
                        self.config.output_dir,
                        session_name
                    )

                    # Add camera configurations to metadata
                    for camera_id, config in self.camera_configs.items():
                        self.metadata_writer.add_camera_info(
                            camera_id,
                            {
                                'camera_id': camera_id,
                                'position': config.position.value,
                                'resolution': (config.resolution_width, config.resolution_height),
                                'fps': config.fps
                            }
                        )

                self.status = RecordingStatus.RECORDING
                self.total_frames_recorded = 0

                logger.info(f"Recording started: {session_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                self.status = RecordingStatus.ERROR
                return False

    def stop_recording(self) -> bool:
        """
        Stop the current recording session.

        Returns:
            True if recording stopped successfully
        """
        with self.lock:
            if self.status != RecordingStatus.RECORDING:
                logger.warning("No recording in progress")
                return False

            try:
                self.status = RecordingStatus.STOPPING

                # Finalize video recording
                if self.video_writer:
                    stats = self.video_writer.get_statistics()
                    logger.info(f"Video recording stats: {stats}")
                    self.video_writer.release_all()
                    self.video_writer = None

                # Finalize metadata
                if self.metadata_writer:
                    self.metadata_writer.finalize()
                    self.metadata_writer = None

                # Calculate duration
                if self.session_start_time:
                    self.recording_duration = datetime.now().timestamp() - self.session_start_time

                logger.info(
                    f"Recording stopped: {self.current_session} "
                    f"({self.total_frames_recorded} frames, "
                    f"{self.recording_duration:.1f}s)"
                )

                self.status = RecordingStatus.IDLE
                self.current_session = None
                return True

            except Exception as e:
                logger.error(f"Error stopping recording: {e}")
                self.status = RecordingStatus.ERROR
                return False

    def record_frame(
        self,
        camera_frame: CameraFrame,
        detections: Optional[List[DetectedObject]] = None,
        lane_info: Optional[LaneInfo] = None
    ) -> bool:
        """
        Record a single camera frame with optional perception data.

        Args:
            camera_frame: Camera frame to record
            detections: Optional detected objects
            lane_info: Optional lane detection info

        Returns:
            True if recorded successfully
        """
        if self.status != RecordingStatus.RECORDING:
            return False

        try:
            # Record video frame
            if self.video_writer:
                # Ensure camera writer exists
                if camera_frame.camera_id not in self.video_writer.writers:
                    self.video_writer.add_camera(camera_frame.camera_id)

                self.video_writer.write_camera_frame(camera_frame)

            # Record metadata
            if self.metadata_writer and detections:
                self.metadata_writer.write_detections(
                    camera_frame.timestamp,
                    camera_frame.camera_id,
                    camera_frame.frame_number,
                    detections
                )

            if self.metadata_writer and lane_info:
                self.metadata_writer.write_lane_info(
                    camera_frame.timestamp,
                    camera_frame.camera_id,
                    camera_frame.frame_number,
                    lane_info
                )

            self.total_frames_recorded += 1
            return True

        except Exception as e:
            logger.error(f"Error recording frame: {e}")
            return False

    def record_perception_data(
        self,
        timestamp: float,
        frame_number: int,
        camera_detections: Dict[int, List[DetectedObject]],
        tracked_objects: Optional[List[TrackedObject]] = None,
        lane_info: Optional[Dict[int, LaneInfo]] = None,
        fused_detections: Optional[List[DetectedObject]] = None
    ) -> bool:
        """
        Record complete perception data for a frame set.

        Args:
            timestamp: Frame timestamp
            frame_number: Frame number
            camera_detections: Detections per camera
            tracked_objects: Tracked objects
            lane_info: Lane info per camera
            fused_detections: Fused detections

        Returns:
            True if recorded successfully
        """
        if self.status != RecordingStatus.RECORDING or not self.metadata_writer:
            return False

        try:
            self.metadata_writer.write_frame_data(
                timestamp,
                frame_number,
                camera_detections,
                tracked_objects,
                lane_info,
                fused_detections
            )
            return True

        except Exception as e:
            logger.error(f"Error recording perception data: {e}")
            return False

    def set_camera_configs(self, camera_configs: Dict[int, CameraConfig]):
        """
        Set camera configurations.

        Args:
            camera_configs: Dictionary mapping camera_id to CameraConfig
        """
        with self.lock:
            self.camera_configs = camera_configs

    def get_status(self) -> RecordingStatus:
        """Get current recording status."""
        return self.status

    def get_statistics(self) -> Dict:
        """
        Get recording statistics.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            stats = {
                'status': self.status.value,
                'current_session': self.current_session,
                'total_frames_recorded': self.total_frames_recorded,
                'recording_duration': self.recording_duration,
                'video_enabled': self.config.record_video,
                'metadata_enabled': self.config.record_metadata
            }

            if self.video_writer:
                stats['video_stats'] = self.video_writer.get_statistics()

            return stats

    def list_sessions(self) -> List[Dict]:
        """
        List all recorded sessions.

        Returns:
            List of session information dictionaries
        """
        sessions = []

        if not self.config.output_dir.exists():
            return sessions

        for session_dir in self.config.output_dir.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        sessions.append({
                            'name': session_dir.name,
                            'path': str(session_dir),
                            'metadata': metadata
                        })

        return sessions

    def delete_session(self, session_name: str) -> bool:
        """
        Delete a recorded session.

        Args:
            session_name: Name of session to delete

        Returns:
            True if deleted successfully
        """
        with self.lock:
            if self.current_session == session_name and self.status == RecordingStatus.RECORDING:
                logger.error("Cannot delete session currently being recorded")
                return False

            session_dir = self.config.output_dir / session_name
            if not session_dir.exists():
                logger.error(f"Session not found: {session_name}")
                return False

            try:
                shutil.rmtree(session_dir)
                logger.info(f"Session deleted: {session_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete session: {e}")
                return False

    def _generate_session_name(self) -> str:
        """Generate a unique session name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"

    def __del__(self):
        """Destructor to ensure recording is stopped."""
        if self.status == RecordingStatus.RECORDING:
            self.stop_recording()
