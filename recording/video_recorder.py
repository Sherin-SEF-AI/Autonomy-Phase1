"""
Multi-camera video recording system with event triggering.

Features:
- Synchronized multi-camera recording
- Configurable quality and compression (H.264/H.265)
- Circular buffer (continuous recording, keep last N minutes)
- Event-triggered clip saving
- Metadata recording
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
from threading import Thread, Lock
import queue

from utils.data_structures import CameraFrame, PerceptionResult
from utils.logger import get_logger


logger = get_logger()


@dataclass
class RecordingConfig:
    """Configuration for video recording."""
    # Output settings
    output_dir: Path = Path("data/recordings")

    # Video encoding
    codec: str = "mp4v"  # 'mp4v' (H.264), 'avc1', 'XVID', 'MJPG'
    fps: int = 30
    quality: int = 90  # 0-100

    # Circular buffer settings
    buffer_duration_sec: float = 300.0  # 5 minutes
    buffer_enabled: bool = True

    # Event recording
    event_pre_buffer_sec: float = 10.0  # Save 10s before event
    event_post_buffer_sec: float = 10.0  # Save 10s after event

    # Storage
    max_storage_gb: float = 50.0  # Max disk usage
    auto_cleanup: bool = True  # Delete old recordings

    # Metadata
    save_metadata: bool = True
    metadata_format: str = "json"  # 'json' or 'csv'


@dataclass
class RecordingEvent:
    """Represents a recording event (collision, hard brake, etc)."""
    timestamp: float
    event_type: str  # 'collision_warning', 'hard_brake', 'manual', etc.
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    metadata: Dict = None


class CameraRecorder:
    """
    Records video from a single camera with circular buffering.
    """

    def __init__(
        self,
        camera_id: int,
        config: RecordingConfig
    ):
        """
        Initialize camera recorder.

        Args:
            camera_id: Camera ID
            config: Recording configuration
        """
        self.camera_id = camera_id
        self.config = config

        # Video writer
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_file: Optional[Path] = None

        # Circular buffer (stores frames in memory)
        max_frames = int(config.buffer_duration_sec * config.fps)
        self.frame_buffer: deque = deque(maxlen=max_frames)

        # Recording state
        self.is_recording = False
        self.recording_lock = Lock()

        # Frame dimensions
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None

        logger.debug(f"Camera {camera_id} recorder initialized (buffer: {max_frames} frames)")

    def add_frame(self, frame: np.ndarray, timestamp: float):
        """
        Add frame to buffer and recording.

        Args:
            frame: BGR image
            timestamp: Frame timestamp
        """
        # Update dimensions if needed
        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]

        # Add to circular buffer if enabled
        if self.config.buffer_enabled:
            # Store copy of frame with timestamp
            self.frame_buffer.append((frame.copy(), timestamp))

        # Write to file if recording
        with self.recording_lock:
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)

    def start_recording(self, filename: Path) -> bool:
        """
        Start recording to file.

        Args:
            filename: Output video file path

        Returns:
            Success status
        """
        with self.recording_lock:
            if self.is_recording:
                logger.warning(f"Camera {self.camera_id}: Already recording")
                return False

            if self.frame_width is None or self.frame_height is None:
                logger.error(f"Camera {self.camera_id}: No frames received yet")
                return False

            # Create output directory
            filename.parent.mkdir(parents=True, exist_ok=True)

            # Get codec
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)

            # Create video writer
            self.video_writer = cv2.VideoWriter(
                str(filename),
                fourcc,
                self.config.fps,
                (self.frame_width, self.frame_height)
            )

            if not self.video_writer.isOpened():
                logger.error(f"Camera {self.camera_id}: Failed to open video writer")
                self.video_writer = None
                return False

            self.current_file = filename
            self.is_recording = True

            logger.info(f"Camera {self.camera_id}: Recording to {filename}")
            return True

    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording.

        Returns:
            Path to recorded file, or None if not recording
        """
        with self.recording_lock:
            if not self.is_recording:
                return None

            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            self.is_recording = False
            recorded_file = self.current_file
            self.current_file = None

            logger.info(f"Camera {self.camera_id}: Recording stopped")
            return recorded_file

    def save_event_clip(
        self,
        output_file: Path,
        event_timestamp: float
    ) -> bool:
        """
        Save a clip around an event from the circular buffer.

        Args:
            output_file: Output video file
            event_timestamp: Event timestamp

        Returns:
            Success status
        """
        if not self.config.buffer_enabled or len(self.frame_buffer) == 0:
            logger.warning(f"Camera {self.camera_id}: Buffer empty, cannot save event clip")
            return False

        # Calculate time range
        start_time = event_timestamp - self.config.event_pre_buffer_sec
        end_time = event_timestamp + self.config.event_post_buffer_sec

        # Extract frames in time range
        frames_to_save = []
        for frame, ts in self.frame_buffer:
            if start_time <= ts <= end_time:
                frames_to_save.append(frame)

        if not frames_to_save:
            logger.warning(f"Camera {self.camera_id}: No frames in event time range")
            return False

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            self.config.fps,
            (self.frame_width, self.frame_height)
        )

        if not writer.isOpened():
            logger.error(f"Camera {self.camera_id}: Failed to create event clip writer")
            return False

        # Write frames
        for frame in frames_to_save:
            writer.write(frame)

        writer.release()

        logger.info(
            f"Camera {self.camera_id}: Saved event clip with {len(frames_to_save)} frames to {output_file}"
        )
        return True

    def get_buffer_info(self) -> Dict:
        """Get buffer statistics."""
        return {
            "camera_id": self.camera_id,
            "buffer_size": len(self.frame_buffer),
            "buffer_max": self.frame_buffer.maxlen,
            "buffer_duration_sec": len(self.frame_buffer) / self.config.fps if self.config.fps > 0 else 0,
            "is_recording": self.is_recording,
            "current_file": str(self.current_file) if self.current_file else None
        }


class MultiCameraRecorder:
    """
    Manages recording for multiple cameras with event triggering.
    """

    def __init__(self, config: Optional[RecordingConfig] = None):
        """
        Initialize multi-camera recorder.

        Args:
            config: Recording configuration
        """
        self.config = config or RecordingConfig()

        # Camera recorders
        self.recorders: Dict[int, CameraRecorder] = {}

        # Event queue for processing
        self.event_queue: queue.Queue = queue.Queue()

        # Recording sessions
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[float] = None

        # Metadata storage
        self.metadata_log: List[Dict] = []

        # Event processing thread
        self.event_processor_thread: Optional[Thread] = None
        self.event_processor_running = False

        # Statistics
        self.total_events_saved = 0
        self.total_recordings = 0

        logger.info("Multi-camera recorder initialized")

    def add_camera(self, camera_id: int):
        """
        Add a camera to the recorder.

        Args:
            camera_id: Camera ID
        """
        if camera_id not in self.recorders:
            self.recorders[camera_id] = CameraRecorder(camera_id, self.config)
            logger.info(f"Added camera {camera_id} to recorder")

    def add_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        timestamp: float,
        perception_result: Optional[PerceptionResult] = None
    ):
        """
        Add frame from camera.

        Args:
            camera_id: Camera ID
            frame: BGR image
            timestamp: Frame timestamp
            perception_result: Optional perception result for metadata
        """
        # Auto-add camera if not exists
        if camera_id not in self.recorders:
            self.add_camera(camera_id)

        # Add frame to recorder
        self.recorders[camera_id].add_frame(frame, timestamp)

        # Store metadata if enabled
        if self.config.save_metadata and perception_result:
            self._log_perception_metadata(camera_id, timestamp, perception_result)

    def start_session_recording(self, session_name: Optional[str] = None) -> str:
        """
        Start a new recording session for all cameras.

        Args:
            session_name: Optional session name

        Returns:
            Session ID
        """
        # Generate session ID
        if session_name:
            session_id = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        self.current_session_id = session_id
        self.session_start_time = time.time()

        # Create session directory
        session_dir = self.config.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Start recording for each camera
        for camera_id, recorder in self.recorders.items():
            filename = session_dir / f"camera_{camera_id}.mp4"
            recorder.start_recording(filename)

        # Start event processor
        if not self.event_processor_running:
            self._start_event_processor()

        self.total_recordings += 1

        logger.info(f"Started recording session: {session_id}")
        return session_id

    def stop_session_recording(self) -> Optional[Dict]:
        """
        Stop current recording session.

        Returns:
            Session information dictionary
        """
        if not self.current_session_id:
            logger.warning("No active recording session")
            return None

        session_id = self.current_session_id
        session_dir = self.config.output_dir / session_id

        # Stop recording for each camera
        recorded_files = {}
        for camera_id, recorder in self.recorders.items():
            file_path = recorder.stop_recording()
            if file_path:
                recorded_files[camera_id] = str(file_path)

        # Save metadata
        if self.config.save_metadata and self.metadata_log:
            metadata_file = session_dir / "metadata.json"
            self._save_metadata(metadata_file)

        # Calculate duration
        duration = time.time() - self.session_start_time if self.session_start_time else 0

        session_info = {
            "session_id": session_id,
            "start_time": self.session_start_time,
            "duration_sec": duration,
            "recorded_files": recorded_files,
            "metadata_entries": len(self.metadata_log)
        }

        # Reset session
        self.current_session_id = None
        self.session_start_time = None
        self.metadata_log.clear()

        logger.info(f"Stopped recording session: {session_id} (duration: {duration:.1f}s)")
        return session_info

    def trigger_event(self, event: RecordingEvent):
        """
        Trigger an event and save clips.

        Args:
            event: Recording event
        """
        logger.info(
            f"Event triggered: {event.event_type} (severity: {event.severity}) - {event.description}"
        )

        # Add to processing queue
        self.event_queue.put(event)

    def _process_events(self):
        """Process events from queue (runs in separate thread)."""
        while self.event_processor_running:
            try:
                # Get event with timeout
                event = self.event_queue.get(timeout=0.5)

                # Create event directory
                event_dir = self.config.output_dir / "events" / datetime.fromtimestamp(
                    event.timestamp
                ).strftime("%Y%m%d_%H%M%S") / event.event_type
                event_dir.mkdir(parents=True, exist_ok=True)

                # Save clip from each camera
                for camera_id, recorder in self.recorders.items():
                    clip_file = event_dir / f"camera_{camera_id}.mp4"
                    recorder.save_event_clip(clip_file, event.timestamp)

                # Save event metadata
                event_metadata_file = event_dir / "event_info.json"
                with open(event_metadata_file, 'w') as f:
                    json.dump({
                        "timestamp": event.timestamp,
                        "datetime": datetime.fromtimestamp(event.timestamp).isoformat(),
                        "event_type": event.event_type,
                        "severity": event.severity,
                        "description": event.description,
                        "metadata": event.metadata or {}
                    }, f, indent=2)

                self.total_events_saved += 1
                logger.info(f"Saved event clips to {event_dir}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    def _start_event_processor(self):
        """Start event processing thread."""
        if self.event_processor_running:
            return

        self.event_processor_running = True
        self.event_processor_thread = Thread(
            target=self._process_events,
            daemon=True,
            name="EventProcessor"
        )
        self.event_processor_thread.start()
        logger.debug("Event processor started")

    def _stop_event_processor(self):
        """Stop event processing thread."""
        if not self.event_processor_running:
            return

        self.event_processor_running = False
        if self.event_processor_thread:
            self.event_processor_thread.join(timeout=2.0)
        logger.debug("Event processor stopped")

    def _log_perception_metadata(
        self,
        camera_id: int,
        timestamp: float,
        result: PerceptionResult
    ):
        """Log perception metadata."""
        metadata_entry = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "frame_number": result.frame_number,
            "detections_count": sum(len(dets) for dets in result.detections_by_camera.values()),
            "tracked_objects_count": len(result.tracked_objects),
            "warnings_count": len(result.active_warnings),
            "processing_time_ms": result.processing_time_ms
        }

        self.metadata_log.append(metadata_entry)

    def _save_metadata(self, output_file: Path):
        """Save metadata to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump({
                    "session_id": self.current_session_id,
                    "start_time": self.session_start_time,
                    "metadata": self.metadata_log
                }, f, indent=2)
            logger.info(f"Saved metadata to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_statistics(self) -> Dict:
        """Get recording statistics."""
        buffer_info = [rec.get_buffer_info() for rec in self.recorders.values()]

        return {
            "total_cameras": len(self.recorders),
            "current_session": self.current_session_id,
            "is_recording": self.current_session_id is not None,
            "total_events_saved": self.total_events_saved,
            "total_recordings": self.total_recordings,
            "event_queue_size": self.event_queue.qsize(),
            "camera_buffers": buffer_info
        }

    def cleanup(self):
        """Cleanup and shutdown recorder."""
        # Stop any active recording
        if self.current_session_id:
            self.stop_session_recording()

        # Stop event processor
        self._stop_event_processor()

        logger.info("Multi-camera recorder cleaned up")
