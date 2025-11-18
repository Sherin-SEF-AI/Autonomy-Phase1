"""
Recording and playback modules for the perception system.

Provides:
- Multi-camera synchronized video recording
- Metadata recording (detections, tracking, sensor fusion)
- Session playback with controls
- Data export utilities
"""

from .recording_manager import RecordingManager, RecordingConfig, RecordingStatus
from .video_writer import MultiCameraVideoWriter, VideoWriterConfig, CameraVideoWriter
from .metadata_writer import MetadataWriter
from .playback_manager import PlaybackManager, PlaybackStatus, PlaybackState

__all__ = [
    'RecordingManager',
    'RecordingConfig',
    'RecordingStatus',
    'MultiCameraVideoWriter',
    'VideoWriterConfig',
    'CameraVideoWriter',
    'MetadataWriter',
    'PlaybackManager',
    'PlaybackStatus',
    'PlaybackState',
]
