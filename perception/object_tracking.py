"""
Object tracking module for multi-object tracking.

Implements centroid-based tracking to maintain consistent IDs across frames.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from scipy.spatial import distance as dist
import time

from utils.data_structures import DetectedObject, TrackedObject
from utils.logger import get_logger


logger = get_logger()


class CentroidTracker:
    """
    Simple centroid-based object tracker.

    Assigns unique IDs to detected objects and tracks them across frames
    based on centroid distance.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100.0
    ):
        """
        Initialize centroid tracker.

        Args:
            max_disappeared: Maximum frames an object can be missing before removal
            max_distance: Maximum centroid distance for matching (pixels)
        """
        self.next_object_id = 0
        self.objects: Dict[int, TrackedObject] = OrderedDict()
        self.disappeared: Dict[int, int] = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Statistics
        self.total_tracks = 0

    def update(
        self,
        detections: List[DetectedObject],
        camera_id: int
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detected objects
            camera_id: Camera that produced these detections

        Returns:
            List of tracked objects with assigned IDs
        """
        # If no detections, increment disappeared counter for all objects
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Remove objects that have disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            return list(self.objects.values())

        # Extract centroids from detections
        input_centroids = np.zeros((len(detections), 2), dtype=np.float32)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids[i] = (cx, cy)

        # If we have no existing objects, register all new detections
        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self._register(det, input_centroids[i], camera_id)

        # Otherwise, match detections to existing objects
        else:
            # Extract existing object centroids
            object_ids = list(self.objects.keys())
            object_centroids = np.zeros((len(object_ids), 2), dtype=np.float32)

            for i, object_id in enumerate(object_ids):
                tracked_obj = self.objects[object_id]
                x1, y1, x2, y2 = tracked_obj.current_bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                object_centroids[i] = (cx, cy)

            # Compute distance matrix
            D = dist.cdist(object_centroids, input_centroids)

            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Match objects to detections
            for (row, col) in zip(rows, cols):
                # Skip if already matched
                if row in used_rows or col in used_cols:
                    continue

                # Skip if distance too large
                if D[row, col] > self.max_distance:
                    continue

                # Update object
                object_id = object_ids[row]
                self._update_object(
                    object_id,
                    detections[col],
                    input_centroids[col],
                    camera_id
                )

                # Reset disappeared counter
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle objects that weren't matched
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            # Register new detections
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self._register(detections[col], input_centroids[col], camera_id)

        return list(self.objects.values())

    def _register(
        self,
        detection: DetectedObject,
        centroid: np.ndarray,
        camera_id: int
    ):
        """Register a new object."""
        track_id = self.next_object_id

        # Calculate 3D position if available
        position_3d = detection.position_3d

        # Create tracked object
        tracked_obj = TrackedObject(
            track_id=track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            first_seen=detection.timestamp,
            last_seen=detection.timestamp,
            current_bbox=detection.bbox,
            current_position=position_3d,
            current_velocity=None,
            confidence=detection.confidence,
            trajectory=[(detection.timestamp, position_3d)] if position_3d else [],
            camera_history=[camera_id],
            frames_tracked=1
        )

        self.objects[track_id] = tracked_obj
        self.disappeared[track_id] = 0
        self.next_object_id += 1
        self.total_tracks += 1

    def _update_object(
        self,
        object_id: int,
        detection: DetectedObject,
        centroid: np.ndarray,
        camera_id: int
    ):
        """Update an existing tracked object."""
        tracked_obj = self.objects[object_id]

        # Update position
        old_position = tracked_obj.current_position
        new_position = detection.position_3d

        # Calculate velocity if we have positions
        velocity = None
        if old_position and new_position:
            dt = detection.timestamp - tracked_obj.last_seen
            if dt > 0:
                dx = new_position[0] - old_position[0]
                dy = new_position[1] - old_position[1]
                velocity = (dx / dt, dy / dt)

        # Update tracked object
        tracked_obj.last_seen = detection.timestamp
        tracked_obj.current_bbox = detection.bbox
        tracked_obj.current_position = new_position
        tracked_obj.current_velocity = velocity
        tracked_obj.confidence = detection.confidence
        tracked_obj.frames_tracked += 1

        # Add to trajectory
        if new_position:
            tracked_obj.trajectory.append((detection.timestamp, new_position))

            # Limit trajectory length
            if len(tracked_obj.trajectory) > 100:
                tracked_obj.trajectory.pop(0)

        # Add camera to history
        if camera_id not in tracked_obj.camera_history:
            tracked_obj.camera_history.append(camera_id)

        # Check if stationary
        if velocity:
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            tracked_obj.is_stationary = speed < 0.5  # < 0.5 m/s

    def _deregister(self, object_id: int):
        """Remove an object from tracking."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def get_object(self, object_id: int) -> Optional[TrackedObject]:
        """
        Get a tracked object by ID.

        Args:
            object_id: Object track ID

        Returns:
            TrackedObject or None if not found
        """
        return self.objects.get(object_id)

    def get_all_objects(self) -> List[TrackedObject]:
        """Get all currently tracked objects."""
        return list(self.objects.values())

    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        return {
            "active_tracks": len(self.objects),
            "total_tracks_created": self.total_tracks,
            "next_object_id": self.next_object_id
        }

    def reset(self):
        """Reset tracker state."""
        self.next_object_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.total_tracks = 0
        logger.info("Object tracker reset")


class MultiCameraTracker:
    """
    Tracker that handles objects from multiple cameras.

    Maintains separate trackers per camera and can associate objects
    seen by different cameras.
    """

    def __init__(
        self,
        num_cameras: int = 4,
        max_disappeared: int = 30,
        max_distance: float = 100.0
    ):
        """
        Initialize multi-camera tracker.

        Args:
            num_cameras: Number of cameras
            max_disappeared: Maximum frames for an object to disappear
            max_distance: Maximum centroid distance for matching
        """
        self.num_cameras = num_cameras

        # Create tracker for each camera
        self.trackers = {
            i: CentroidTracker(max_disappeared, max_distance)
            for i in range(num_cameras)
        }

        # Global track ID management
        self.next_global_id = 0
        self.camera_to_global_id: Dict[Tuple[int, int], int] = {}  # (camera_id, local_id) -> global_id

    def update(
        self,
        detections_by_camera: Dict[int, List[DetectedObject]]
    ) -> List[TrackedObject]:
        """
        Update trackers with detections from multiple cameras.

        Args:
            detections_by_camera: Dictionary mapping camera_id to list of detections

        Returns:
            List of all tracked objects (with global IDs)
        """
        all_tracked = []

        # Update each camera's tracker
        for camera_id, detections in detections_by_camera.items():
            if camera_id in self.trackers:
                tracked = self.trackers[camera_id].update(detections, camera_id)

                # Assign global IDs
                for obj in tracked:
                    key = (camera_id, obj.track_id)

                    if key not in self.camera_to_global_id:
                        # Assign new global ID
                        self.camera_to_global_id[key] = self.next_global_id
                        self.next_global_id += 1

                    # Update track_id to global ID
                    obj.track_id = self.camera_to_global_id[key]

                all_tracked.extend(tracked)

        return all_tracked

    def get_statistics(self) -> Dict:
        """Get statistics from all trackers."""
        stats = {
            "global_tracks": self.next_global_id,
            "per_camera": {}
        }

        for camera_id, tracker in self.trackers.items():
            stats["per_camera"][camera_id] = tracker.get_statistics()

        return stats

    def reset(self):
        """Reset all trackers."""
        for tracker in self.trackers.values():
            tracker.reset()

        self.next_global_id = 0
        self.camera_to_global_id.clear()
        logger.info("Multi-camera tracker reset")
