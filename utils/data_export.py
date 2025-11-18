"""
Data export utilities for perception system.

Provides tools to export perception data to various formats:
- CSV: Detection and tracking data
- JSON: Complete session data
- Video: Annotated video clips
"""

import csv
import json
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np

from .data_structures import DetectedObject, TrackedObject, LaneInfo, CameraFrame
from .logger import get_logger


logger = get_logger()


class CSVExporter:
    """
    Export perception data to CSV format.

    Creates CSV files for detections, tracking, and lane information.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize CSV exporter.

        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CSV exporter initialized: {output_dir}")

    def export_detections(
        self,
        detections: List[DetectedObject],
        filename: str = "detections.csv"
    ) -> Path:
        """
        Export detections to CSV.

        Args:
            detections: List of detected objects
            filename: Output filename

        Returns:
            Path to created CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'timestamp', 'object_id', 'camera_id', 'class_name', 'class_id',
                'confidence', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                'position_x', 'position_y', 'position_z', 'distance'
            ])

            # Data rows
            for det in detections:
                bbox_x, bbox_y, bbox_w, bbox_h = det.bbox if det.bbox else (0, 0, 0, 0)
                pos_x, pos_y, pos_z = det.position_3d if det.position_3d else (0, 0, 0)

                writer.writerow([
                    det.timestamp,
                    det.object_id,
                    det.camera_id,
                    det.class_name,
                    det.class_id,
                    det.confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    pos_x, pos_y, pos_z,
                    det.distance
                ])

        logger.info(f"Exported {len(detections)} detections to {output_path}")
        return output_path

    def export_tracking(
        self,
        tracked_objects: List[TrackedObject],
        filename: str = "tracking.csv"
    ) -> Path:
        """
        Export tracked objects to CSV.

        Args:
            tracked_objects: List of tracked objects
            filename: Output filename

        Returns:
            Path to created CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'track_id', 'class_name', 'position_x', 'position_y', 'position_z',
                'velocity_x', 'velocity_y', 'confidence', 'age', 'times_seen',
                'last_seen', 'camera_ids'
            ])

            # Data rows
            for obj in tracked_objects:
                pos_x, pos_y, pos_z = obj.current_position if obj.current_position else (0, 0, 0)
                vel_x, vel_y = obj.current_velocity if obj.current_velocity else (0, 0)

                writer.writerow([
                    obj.track_id,
                    obj.class_name,
                    pos_x, pos_y, pos_z,
                    vel_x, vel_y,
                    obj.confidence,
                    obj.age,
                    obj.times_seen,
                    obj.last_seen,
                    ','.join(str(cid) for cid in obj.camera_ids)
                ])

        logger.info(f"Exported {len(tracked_objects)} tracked objects to {output_path}")
        return output_path

    def export_lane_info(
        self,
        lane_data: List[Dict[str, Any]],
        filename: str = "lanes.csv"
    ) -> Path:
        """
        Export lane information to CSV.

        Args:
            lane_data: List of lane info dictionaries
            filename: Output filename

        Returns:
            Path to created CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'timestamp', 'camera_id', 'left_detected', 'right_detected',
                'lateral_offset', 'lane_width', 'departure_warning', 'confidence'
            ])

            # Data rows
            for data in lane_data:
                writer.writerow([
                    data.get('timestamp', 0),
                    data.get('camera_id', 0),
                    data.get('left_lane_detected', False),
                    data.get('right_lane_detected', False),
                    data.get('lateral_offset', 0),
                    data.get('lane_width', 0),
                    data.get('departure_warning', ''),
                    data.get('confidence', 0)
                ])

        logger.info(f"Exported {len(lane_data)} lane records to {output_path}")
        return output_path


class JSONExporter:
    """
    Export perception data to JSON format.

    Creates comprehensive JSON files with all perception data.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize JSON exporter.

        Args:
            output_dir: Directory to save JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"JSON exporter initialized: {output_dir}")

    def export_session(
        self,
        session_data: Dict[str, Any],
        filename: str = "session.json"
    ) -> Path:
        """
        Export complete session data to JSON.

        Args:
            session_data: Session data dictionary
            filename: Output filename

        Returns:
            Path to created JSON file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Exported session data to {output_path}")
        return output_path

    def export_detections(
        self,
        detections: List[DetectedObject],
        filename: str = "detections.json"
    ) -> Path:
        """
        Export detections to JSON.

        Args:
            detections: List of detected objects
            filename: Output filename

        Returns:
            Path to created JSON file
        """
        output_path = self.output_dir / filename

        data = {
            'timestamp': datetime.now().isoformat(),
            'count': len(detections),
            'detections': [
                {
                    'timestamp': det.timestamp,
                    'object_id': det.object_id,
                    'camera_id': det.camera_id,
                    'class_name': det.class_name,
                    'class_id': det.class_id,
                    'confidence': det.confidence,
                    'bbox': det.bbox,
                    'bbox_3d': det.bbox_3d,
                    'position_3d': det.position_3d,
                    'distance': det.distance
                }
                for det in detections
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(detections)} detections to {output_path}")
        return output_path


class VideoExporter:
    """
    Export annotated video clips.

    Creates video files with perception overlays.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize video exporter.

        Args:
            output_dir: Directory to save video files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Video exporter initialized: {output_dir}")

    def export_clip(
        self,
        frames: List[np.ndarray],
        filename: str = "clip.mp4",
        fps: int = 30,
        codec: str = 'mp4v'
    ) -> Path:
        """
        Export video clip from frames.

        Args:
            frames: List of frames (BGR format)
            filename: Output filename
            fps: Frames per second
            codec: Video codec

        Returns:
            Path to created video file
        """
        if not frames:
            logger.warning("No frames to export")
            return None

        output_path = self.output_dir / filename

        # Get frame dimensions from first frame
        height, width = frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            logger.error(f"Failed to create video writer: {output_path}")
            return None

        # Write frames
        for frame in frames:
            # Resize if needed
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)

        writer.release()

        logger.info(f"Exported {len(frames)} frames to {output_path}")
        return output_path

    def export_multi_camera_clip(
        self,
        camera_frames: Dict[int, List[np.ndarray]],
        filename: str = "multi_camera.mp4",
        fps: int = 30,
        layout: str = "2x2"
    ) -> Path:
        """
        Export multi-camera video with grid layout.

        Args:
            camera_frames: Dictionary mapping camera_id to frame list
            filename: Output filename
            fps: Frames per second
            layout: Layout pattern ("2x2", "1x4", etc.)

        Returns:
            Path to created video file
        """
        if not camera_frames:
            logger.warning("No camera frames to export")
            return None

        # Determine number of frames
        num_frames = min(len(frames) for frames in camera_frames.values())

        if num_frames == 0:
            logger.warning("No frames to export")
            return None

        # Parse layout
        if layout == "2x2":
            rows, cols = 2, 2
        elif layout == "1x4":
            rows, cols = 1, 4
        elif layout == "4x1":
            rows, cols = 4, 1
        else:
            rows, cols = 2, 2

        # Get frame size (use first camera)
        first_camera_id = list(camera_frames.keys())[0]
        frame_h, frame_w = camera_frames[first_camera_id][0].shape[:2]

        # Create output grid
        grid_w = frame_w * cols
        grid_h = frame_h * rows

        output_path = self.output_dir / filename

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (grid_w, grid_h)
        )

        if not writer.isOpened():
            logger.error(f"Failed to create video writer: {output_path}")
            return None

        # Write frames
        camera_ids = sorted(camera_frames.keys())

        for frame_idx in range(num_frames):
            # Create grid frame
            grid_frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for i, camera_id in enumerate(camera_ids):
                if i >= rows * cols:
                    break

                row = i // cols
                col = i % cols

                frame = camera_frames[camera_id][frame_idx]

                # Resize if needed
                if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
                    frame = cv2.resize(frame, (frame_w, frame_h))

                # Place in grid
                y1 = row * frame_h
                y2 = y1 + frame_h
                x1 = col * frame_w
                x2 = x1 + frame_w

                grid_frame[y1:y2, x1:x2] = frame

            writer.write(grid_frame)

        writer.release()

        logger.info(f"Exported {num_frames} frames from {len(camera_ids)} cameras to {output_path}")
        return output_path


class DataExporter:
    """
    Unified data exporter with all formats.

    Provides a single interface for exporting perception data
    to CSV, JSON, and video formats.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize data exporter.

        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create sub-exporters
        self.csv_exporter = CSVExporter(self.output_dir / "csv")
        self.json_exporter = JSONExporter(self.output_dir / "json")
        self.video_exporter = VideoExporter(self.output_dir / "video")

        logger.info(f"Data exporter initialized: {output_dir}")

    def export_all(
        self,
        detections: Optional[List[DetectedObject]] = None,
        tracked_objects: Optional[List[TrackedObject]] = None,
        lane_data: Optional[List[Dict]] = None,
        session_data: Optional[Dict] = None,
        video_frames: Optional[Dict[int, List[np.ndarray]]] = None
    ) -> Dict[str, List[Path]]:
        """
        Export all available data to all formats.

        Args:
            detections: Detected objects
            tracked_objects: Tracked objects
            lane_data: Lane information
            session_data: Session metadata
            video_frames: Video frames per camera

        Returns:
            Dictionary mapping format to list of created files
        """
        created_files = {
            'csv': [],
            'json': [],
            'video': []
        }

        # Export CSV
        if detections:
            path = self.csv_exporter.export_detections(detections)
            created_files['csv'].append(path)

        if tracked_objects:
            path = self.csv_exporter.export_tracking(tracked_objects)
            created_files['csv'].append(path)

        if lane_data:
            path = self.csv_exporter.export_lane_info(lane_data)
            created_files['csv'].append(path)

        # Export JSON
        if detections:
            path = self.json_exporter.export_detections(detections)
            created_files['json'].append(path)

        if session_data:
            path = self.json_exporter.export_session(session_data)
            created_files['json'].append(path)

        # Export video
        if video_frames:
            path = self.video_exporter.export_multi_camera_clip(video_frames)
            if path:
                created_files['video'].append(path)

        logger.info(f"Exported data: {sum(len(files) for files in created_files.values())} files created")
        return created_files
