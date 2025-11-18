"""
Playback manager for replaying recorded sessions.

Provides synchronized playback of multi-camera video recordings with
perception metadata visualization.
"""

import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

from utils.data_structures import DetectedObject, TrackedObject, LaneInfo, CameraFrame, CameraPosition
from utils.logger import get_logger


logger = get_logger()


class PlaybackStatus(Enum):
    """Playback status states."""
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PlaybackState:
    """Current playback state information."""
    current_frame: int = 0
    total_frames: int = 0
    current_time: float = 0.0
    total_time: float = 0.0
    playback_speed: float = 1.0
    fps: int = 30


class CameraVideoReader:
    """
    Video reader for a single camera stream.

    Handles reading frames from a recorded video file.
    """

    def __init__(self, camera_id: int, video_path: Path):
        """
        Initialize camera video reader.

        Args:
            camera_id: Camera identifier
            video_path: Path to video file
        """
        self.camera_id = camera_id
        self.video_path = video_path

        # Open video capture
        self.capture = cv2.VideoCapture(str(video_path))

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Video properties
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame = 0

        logger.info(
            f"Video reader opened for camera {camera_id}: "
            f"{self.frame_count} frames @ {self.fps} FPS"
        )

    def read_frame(self) -> Optional[Tuple[bool, CameraFrame]]:
        """
        Read the next frame.

        Returns:
            Tuple of (success, CameraFrame) or None if end of video
        """
        ret, frame = self.capture.read()

        if not ret:
            return None

        camera_frame = CameraFrame(
            camera_id=self.camera_id,
            timestamp=self.current_frame / self.fps,
            frame_number=self.current_frame,
            image=frame,
            width=self.width,
            height=self.height,
            camera_position=CameraPosition.UNKNOWN
        )

        self.current_frame += 1
        return (ret, camera_frame)

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to

        Returns:
            True if successful
        """
        if 0 <= frame_number < self.frame_count:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        return False

    def reset(self):
        """Reset to the beginning."""
        self.seek(0)

    def release(self):
        """Release the video capture."""
        if self.capture is not None:
            self.capture.release()
            logger.info(f"Video reader released for camera {self.camera_id}")


class MetadataReader:
    """
    Reads perception metadata from recorded sessions.

    Loads detection, tracking, lane, and fusion data for synchronized
    playback with video.
    """

    def __init__(self, session_dir: Path):
        """
        Initialize metadata reader.

        Args:
            session_dir: Path to session directory
        """
        self.session_dir = Path(session_dir)

        # Load session metadata
        metadata_file = self.session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.session_metadata = json.load(f)
        else:
            self.session_metadata = {}

        # Load JSONL files into memory (indexed by frame number)
        self.detections_by_frame: Dict[int, Dict] = {}
        self.tracking_by_frame: Dict[int, Dict] = {}
        self.lanes_by_frame: Dict[int, Dict] = {}
        self.fusion_by_frame: Dict[int, Dict] = {}

        self._load_metadata_files()

        logger.info(f"Metadata reader initialized for session: {session_dir.name}")

    def _load_metadata_files(self):
        """Load all metadata files into memory."""
        # Load detections
        detections_file = self.session_dir / "detections.jsonl"
        if detections_file.exists():
            with open(detections_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    frame_num = record['frame_number']
                    camera_id = record['camera_id']

                    if frame_num not in self.detections_by_frame:
                        self.detections_by_frame[frame_num] = {}

                    self.detections_by_frame[frame_num][camera_id] = record['detections']

        # Load tracking
        tracking_file = self.session_dir / "tracking.jsonl"
        if tracking_file.exists():
            with open(tracking_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    frame_num = record['frame_number']
                    self.tracking_by_frame[frame_num] = record['tracked_objects']

        # Load lanes
        lanes_file = self.session_dir / "lanes.jsonl"
        if lanes_file.exists():
            with open(lanes_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    frame_num = record['frame_number']
                    camera_id = record['camera_id']

                    if frame_num not in self.lanes_by_frame:
                        self.lanes_by_frame[frame_num] = {}

                    self.lanes_by_frame[frame_num][camera_id] = record['lane_info']

        # Load fusion
        fusion_file = self.session_dir / "fusion.jsonl"
        if fusion_file.exists():
            with open(fusion_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    frame_num = record['frame_number']
                    self.fusion_by_frame[frame_num] = record['fused_detections']

        logger.debug(
            f"Loaded metadata: {len(self.detections_by_frame)} detection frames, "
            f"{len(self.tracking_by_frame)} tracking frames"
        )

    def get_detections(self, frame_number: int, camera_id: int) -> List[Dict]:
        """Get detections for a specific frame and camera."""
        if frame_number in self.detections_by_frame:
            return self.detections_by_frame[frame_number].get(camera_id, [])
        return []

    def get_tracking(self, frame_number: int) -> List[Dict]:
        """Get tracked objects for a specific frame."""
        return self.tracking_by_frame.get(frame_number, [])

    def get_lane_info(self, frame_number: int, camera_id: int) -> Optional[Dict]:
        """Get lane info for a specific frame and camera."""
        if frame_number in self.lanes_by_frame:
            return self.lanes_by_frame[frame_number].get(camera_id)
        return None

    def get_fusion_data(self, frame_number: int) -> List[Dict]:
        """Get fusion data for a specific frame."""
        return self.fusion_by_frame.get(frame_number, [])


class PlaybackManager:
    """
    Manages playback of recorded sessions.

    Provides synchronized multi-camera video playback with perception
    metadata overlay and playback controls.
    """

    def __init__(self):
        """Initialize playback manager."""
        # Playback state
        self.status = PlaybackStatus.IDLE
        self.state = PlaybackState()

        # Session data
        self.session_dir: Optional[Path] = None
        self.session_name: Optional[str] = None

        # Video readers
        self.video_readers: Dict[int, CameraVideoReader] = {}

        # Metadata reader
        self.metadata_reader: Optional[MetadataReader] = None

        # Thread safety
        self.lock = threading.Lock()

        logger.info("Playback manager initialized")

    def load_session(self, session_path: Path) -> bool:
        """
        Load a recorded session for playback.

        Args:
            session_path: Path to session directory

        Returns:
            True if loaded successfully
        """
        with self.lock:
            try:
                self.session_dir = Path(session_path)
                self.session_name = self.session_dir.name

                # Find all video files
                video_files = list(self.session_dir.glob("camera_*.mp4"))

                if not video_files:
                    logger.error(f"No video files found in {session_path}")
                    return False

                # Create video readers
                self.video_readers.clear()
                for video_file in video_files:
                    # Extract camera ID from filename (camera_0.mp4 -> 0)
                    camera_id = int(video_file.stem.split('_')[1])
                    reader = CameraVideoReader(camera_id, video_file)
                    self.video_readers[camera_id] = reader

                # Calculate total frames (use first camera as reference)
                if self.video_readers:
                    first_reader = next(iter(self.video_readers.values()))
                    self.state.total_frames = first_reader.frame_count
                    self.state.fps = first_reader.fps
                    self.state.total_time = self.state.total_frames / self.state.fps

                # Load metadata
                self.metadata_reader = MetadataReader(self.session_dir)

                self.state.current_frame = 0
                self.state.current_time = 0.0
                self.status = PlaybackStatus.STOPPED

                logger.info(
                    f"Session loaded: {self.session_name} "
                    f"({len(self.video_readers)} cameras, {self.state.total_frames} frames)"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to load session: {e}")
                self.status = PlaybackStatus.ERROR
                return False

    def play(self):
        """Start or resume playback."""
        with self.lock:
            if self.status in [PlaybackStatus.STOPPED, PlaybackStatus.PAUSED]:
                self.status = PlaybackStatus.PLAYING
                logger.info("Playback started")

    def pause(self):
        """Pause playback."""
        with self.lock:
            if self.status == PlaybackStatus.PLAYING:
                self.status = PlaybackStatus.PAUSED
                logger.info("Playback paused")

    def stop(self):
        """Stop playback and reset to beginning."""
        with self.lock:
            self.status = PlaybackStatus.STOPPED
            self.seek(0)
            logger.info("Playback stopped")

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to

        Returns:
            True if successful
        """
        with self.lock:
            if 0 <= frame_number < self.state.total_frames:
                for reader in self.video_readers.values():
                    reader.seek(frame_number)

                self.state.current_frame = frame_number
                self.state.current_time = frame_number / self.state.fps
                return True
            return False

    def seek_time(self, time_seconds: float) -> bool:
        """
        Seek to a specific time.

        Args:
            time_seconds: Time in seconds

        Returns:
            True if successful
        """
        frame_number = int(time_seconds * self.state.fps)
        return self.seek(frame_number)

    def step_forward(self, frames: int = 1) -> bool:
        """Step forward by N frames."""
        return self.seek(self.state.current_frame + frames)

    def step_backward(self, frames: int = 1) -> bool:
        """Step backward by N frames."""
        return self.seek(self.state.current_frame - frames)

    def set_playback_speed(self, speed: float):
        """Set playback speed multiplier (0.25x to 4.0x)."""
        with self.lock:
            self.state.playback_speed = max(0.25, min(4.0, speed))
            logger.debug(f"Playback speed set to {self.state.playback_speed}x")

    def read_next_frame(self) -> Optional[Dict[int, CameraFrame]]:
        """
        Read the next frame from all cameras.

        Returns:
            Dictionary mapping camera_id to CameraFrame, or None if end of video
        """
        if self.status != PlaybackStatus.PLAYING:
            return None

        with self.lock:
            if self.state.current_frame >= self.state.total_frames:
                self.status = PlaybackStatus.STOPPED
                return None

            frames = {}
            for camera_id, reader in self.video_readers.items():
                result = reader.read_frame()
                if result:
                    success, camera_frame = result
                    if success:
                        frames[camera_id] = camera_frame

            if frames:
                self.state.current_frame += 1
                self.state.current_time = self.state.current_frame / self.state.fps
                return frames

            return None

    def get_current_metadata(self) -> Dict:
        """
        Get metadata for the current frame.

        Returns:
            Dictionary with detection, tracking, and lane data
        """
        if not self.metadata_reader:
            return {}

        frame_num = self.state.current_frame

        metadata = {
            'frame_number': frame_num,
            'timestamp': self.state.current_time,
            'detections': {},
            'tracking': [],
            'lanes': {},
            'fusion': []
        }

        # Get detections per camera
        for camera_id in self.video_readers.keys():
            detections = self.metadata_reader.get_detections(frame_num, camera_id)
            if detections:
                metadata['detections'][camera_id] = detections

        # Get tracking data
        metadata['tracking'] = self.metadata_reader.get_tracking(frame_num)

        # Get lane info per camera
        for camera_id in self.video_readers.keys():
            lane_info = self.metadata_reader.get_lane_info(frame_num, camera_id)
            if lane_info:
                metadata['lanes'][camera_id] = lane_info

        # Get fusion data
        metadata['fusion'] = self.metadata_reader.get_fusion_data(frame_num)

        return metadata

    def get_status(self) -> PlaybackStatus:
        """Get current playback status."""
        return self.status

    def get_state(self) -> PlaybackState:
        """Get current playback state."""
        return self.state

    def unload_session(self):
        """Unload the current session."""
        with self.lock:
            # Release video readers
            for reader in self.video_readers.values():
                reader.release()

            self.video_readers.clear()
            self.metadata_reader = None
            self.session_dir = None
            self.session_name = None
            self.status = PlaybackStatus.IDLE

            logger.info("Session unloaded")

    def __del__(self):
        """Destructor to ensure resources are released."""
        if self.video_readers:
            self.unload_session()
