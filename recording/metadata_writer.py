"""
Metadata writer for recording perception data.

Records detection, tracking, lane detection, and sensor fusion data
as JSON sidecar files alongside video recordings.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import threading
from datetime import datetime

from utils.data_structures import DetectedObject, TrackedObject, LaneInfo
from utils.logger import get_logger


logger = get_logger()


class MetadataWriter:
    """
    Writes perception metadata to JSON files.

    Records frame-by-frame data about detections, tracking, and other
    perception outputs for later playback and analysis.
    """

    def __init__(self, output_dir: Path, session_name: str):
        """
        Initialize metadata writer.

        Args:
            output_dir: Directory to save metadata files
            session_name: Name for this recording session
        """
        self.output_dir = Path(output_dir)
        self.session_name = session_name

        # Create session directory
        self.session_dir = self.output_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Metadata files
        self.metadata_file = self.session_dir / "metadata.json"
        self.detections_file = self.session_dir / "detections.jsonl"
        self.tracking_file = self.session_dir / "tracking.jsonl"
        self.lanes_file = self.session_dir / "lanes.jsonl"
        self.fusion_file = self.session_dir / "fusion.jsonl"

        # Session metadata
        self.session_metadata = {
            'session_name': session_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_frames': 0,
            'cameras': {},
            'configuration': {}
        }

        # Open file handles for streaming writes (JSONL format)
        self.detections_handle = open(self.detections_file, 'w')
        self.tracking_handle = open(self.tracking_file, 'w')
        self.lanes_handle = open(self.lanes_file, 'w')
        self.fusion_handle = open(self.fusion_file, 'w')

        # Statistics
        self.frame_count = 0
        self.lock = threading.Lock()

        logger.info(f"Metadata writer initialized: {self.session_dir}")

    def write_detections(
        self,
        timestamp: float,
        camera_id: int,
        frame_number: int,
        detections: List[DetectedObject]
    ):
        """
        Write detection data for a frame.

        Args:
            timestamp: Frame timestamp
            camera_id: Camera identifier
            frame_number: Frame number
            detections: List of detected objects
        """
        with self.lock:
            record = {
                'timestamp': timestamp,
                'camera_id': camera_id,
                'frame_number': frame_number,
                'detections': [self._serialize_detection(det) for det in detections]
            }

            self.detections_handle.write(json.dumps(record) + '\n')
            self.detections_handle.flush()

    def write_tracking(
        self,
        timestamp: float,
        frame_number: int,
        tracked_objects: List[TrackedObject]
    ):
        """
        Write tracking data for a frame.

        Args:
            timestamp: Frame timestamp
            frame_number: Frame number
            tracked_objects: List of tracked objects
        """
        with self.lock:
            record = {
                'timestamp': timestamp,
                'frame_number': frame_number,
                'tracked_objects': [self._serialize_tracked_object(obj) for obj in tracked_objects]
            }

            self.tracking_handle.write(json.dumps(record) + '\n')
            self.tracking_handle.flush()

    def write_lane_info(
        self,
        timestamp: float,
        camera_id: int,
        frame_number: int,
        lane_info: Optional[LaneInfo]
    ):
        """
        Write lane detection data for a frame.

        Args:
            timestamp: Frame timestamp
            camera_id: Camera identifier
            frame_number: Frame number
            lane_info: Lane detection information
        """
        if lane_info is None:
            return

        with self.lock:
            record = {
                'timestamp': timestamp,
                'camera_id': camera_id,
                'frame_number': frame_number,
                'lane_info': self._serialize_lane_info(lane_info)
            }

            self.lanes_handle.write(json.dumps(record) + '\n')
            self.lanes_handle.flush()

    def write_fusion_data(
        self,
        timestamp: float,
        frame_number: int,
        fused_detections: List[DetectedObject]
    ):
        """
        Write sensor fusion data.

        Args:
            timestamp: Frame timestamp
            frame_number: Frame number
            fused_detections: List of fused detections
        """
        with self.lock:
            record = {
                'timestamp': timestamp,
                'frame_number': frame_number,
                'fused_detections': [self._serialize_detection(det) for det in fused_detections]
            }

            self.fusion_handle.write(json.dumps(record) + '\n')
            self.fusion_handle.flush()

    def write_frame_data(
        self,
        timestamp: float,
        frame_number: int,
        camera_detections: Dict[int, List[DetectedObject]],
        tracked_objects: Optional[List[TrackedObject]] = None,
        lane_info: Optional[Dict[int, LaneInfo]] = None,
        fused_detections: Optional[List[DetectedObject]] = None
    ):
        """
        Write all perception data for a frame at once.

        Args:
            timestamp: Frame timestamp
            frame_number: Frame number
            camera_detections: Dictionary mapping camera_id to detections
            tracked_objects: List of tracked objects
            lane_info: Dictionary mapping camera_id to lane info
            fused_detections: List of fused detections
        """
        # Write detections per camera
        for camera_id, detections in camera_detections.items():
            self.write_detections(timestamp, camera_id, frame_number, detections)

        # Write tracking data
        if tracked_objects:
            self.write_tracking(timestamp, frame_number, tracked_objects)

        # Write lane info
        if lane_info:
            for camera_id, info in lane_info.items():
                self.write_lane_info(timestamp, camera_id, frame_number, info)

        # Write fusion data
        if fused_detections:
            self.write_fusion_data(timestamp, frame_number, fused_detections)

        with self.lock:
            self.frame_count += 1

    def update_session_metadata(self, key: str, value: Any):
        """
        Update session metadata.

        Args:
            key: Metadata key
            value: Metadata value
        """
        with self.lock:
            self.session_metadata[key] = value

    def add_camera_info(self, camera_id: int, camera_info: Dict[str, Any]):
        """
        Add camera information to session metadata.

        Args:
            camera_id: Camera identifier
            camera_info: Camera configuration and info
        """
        with self.lock:
            self.session_metadata['cameras'][camera_id] = camera_info

    def finalize(self):
        """Finalize recording and write session metadata."""
        with self.lock:
            # Update session metadata
            self.session_metadata['end_time'] = datetime.now().isoformat()
            self.session_metadata['total_frames'] = self.frame_count

            # Write session metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.session_metadata, f, indent=2)

            # Close file handles
            self.detections_handle.close()
            self.tracking_handle.close()
            self.lanes_handle.close()
            self.fusion_handle.close()

            logger.info(
                f"Metadata finalized: {self.frame_count} frames recorded"
            )

    def _serialize_detection(self, det: DetectedObject) -> Dict:
        """Serialize DetectedObject to dictionary."""
        return {
            'object_id': det.object_id,
            'camera_id': det.camera_id,
            'class_name': det.class_name,
            'class_id': det.class_id,
            'confidence': det.confidence,
            'bbox': det.bbox,
            'bbox_3d': det.bbox_3d,
            'position_3d': det.position_3d,
            'distance': det.distance,
            'timestamp': det.timestamp
        }

    def _serialize_tracked_object(self, obj: TrackedObject) -> Dict:
        """Serialize TrackedObject to dictionary."""
        return {
            'track_id': obj.track_id,
            'class_name': obj.class_name,
            'current_position': obj.current_position,
            'current_velocity': obj.current_velocity,
            'confidence': obj.confidence,
            'age': obj.age,
            'times_seen': obj.times_seen,
            'last_seen': obj.last_seen,
            'camera_ids': list(obj.camera_ids),
            'trajectory': [(ts, pos) for ts, pos in obj.trajectory[-20:]]  # Last 20 points
        }

    def _serialize_lane_info(self, lane_info: LaneInfo) -> Dict:
        """Serialize LaneInfo to dictionary."""
        return {
            'left_lane_detected': lane_info.left_lane_detected,
            'right_lane_detected': lane_info.right_lane_detected,
            'left_coeffs': lane_info.left_coeffs,
            'right_coeffs': lane_info.right_coeffs,
            'lateral_offset': lane_info.lateral_offset,
            'lane_width': lane_info.lane_width,
            'departure_warning': lane_info.departure_warning,
            'confidence': lane_info.confidence
        }

    def __del__(self):
        """Destructor to ensure files are closed."""
        try:
            if hasattr(self, 'detections_handle') and not self.detections_handle.closed:
                self.finalize()
        except Exception as e:
            logger.error(f"Error in MetadataWriter destructor: {e}")
