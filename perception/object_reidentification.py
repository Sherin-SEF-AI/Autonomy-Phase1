"""
Object re-identification for tracking across camera views and occlusions.

Uses appearance-based features to re-identify objects that disappear and reappear,
either in the same camera or across different camera views.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

from utils.data_structures import TrackedObject, DetectedObject
from utils.logger import get_logger


logger = get_logger()


@dataclass
class ObjectAppearance:
    """Stores appearance features for an object."""
    track_id: int
    features: np.ndarray  # Appearance feature vector
    timestamp: float
    camera_id: int
    bbox: Tuple[int, int, int, int]
    image_patch: Optional[np.ndarray] = None  # Cropped image for visualization


class ObjectReIdentifier:
    """
    Re-identifies objects across cameras and occlusions.

    Uses color histograms and (optionally) deep features for matching.
    """

    def __init__(
        self,
        feature_type: str = "histogram",  # "histogram", "deep", or "hybrid"
        similarity_threshold: float = 0.7,
        max_appearance_history: int = 50,
        occlusion_timeout: float = 5.0
    ):
        """
        Initialize object re-identifier.

        Args:
            feature_type: Type of features to use
            similarity_threshold: Minimum similarity for re-identification (0-1)
            max_appearance_history: Maximum appearances to store per object
            occlusion_timeout: Max time to keep lost object in memory (seconds)
        """
        self.feature_type = feature_type
        self.similarity_threshold = similarity_threshold
        self.max_appearance_history = max_appearance_history
        self.occlusion_timeout = occlusion_timeout

        # Appearance database: track_id -> deque of ObjectAppearance
        self.appearance_db: Dict[int, deque] = {}

        # Lost objects (for re-identification after occlusion)
        self.lost_objects: Dict[int, float] = {}  # track_id -> last_seen_time

        # Deep learning model (optional)
        self.reid_model = None
        if feature_type in ["deep", "hybrid"]:
            self._initialize_deep_model()

        # Statistics
        self.total_reidentifications = 0
        self.total_extractions = 0

        logger.info(f"Object re-identifier initialized (type: {feature_type})")

    def _initialize_deep_model(self):
        """Initialize deep learning re-ID model (optional)."""
        try:
            import torch
            # Could use torchreid or similar
            # For now, we'll use histogram features
            logger.info("Deep re-ID models not yet implemented, using histogram features")
            self.feature_type = "histogram"
        except ImportError:
            logger.warning("PyTorch not available, using histogram features only")
            self.feature_type = "histogram"

    def extract_features(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract appearance features from object bbox.

        Args:
            image: Full image (BGR)
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Feature vector
        """
        self.total_extractions += 1

        x1, y1, x2, y2 = bbox

        # Ensure valid bbox
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # Extract patch
        patch = image[y1:y2, x1:x2]

        if patch.size == 0:
            return np.zeros(256, dtype=np.float32)

        if self.feature_type == "histogram":
            return self._extract_histogram_features(patch)
        elif self.feature_type == "deep":
            return self._extract_deep_features(patch)
        else:  # hybrid
            hist = self._extract_histogram_features(patch)
            deep = self._extract_deep_features(patch)
            return np.concatenate([hist, deep])

    def _extract_histogram_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        # Calculate histograms
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

        # Normalize
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()

        # Concatenate
        features = np.concatenate([h_hist, s_hist, v_hist])

        return features.astype(np.float32)

    def _extract_deep_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract deep learning features (placeholder)."""
        # Would use pre-trained ReID network here
        # For now, return empty array
        return np.array([], dtype=np.float32)

    def update_appearance(
        self,
        tracked_object: TrackedObject,
        image: np.ndarray,
        camera_id: int
    ):
        """
        Update appearance database for a tracked object.

        Args:
            tracked_object: Tracked object
            image: Full image
            camera_id: Camera ID
        """
        track_id = tracked_object.track_id

        # Extract features
        features = self.extract_features(image, tracked_object.current_bbox)

        # Create appearance record
        appearance = ObjectAppearance(
            track_id=track_id,
            features=features,
            timestamp=time.time(),
            camera_id=camera_id,
            bbox=tracked_object.current_bbox
        )

        # Add to database
        if track_id not in self.appearance_db:
            self.appearance_db[track_id] = deque(maxlen=self.max_appearance_history)

        self.appearance_db[track_id].append(appearance)

        # Remove from lost objects if it was there
        if track_id in self.lost_objects:
            del self.lost_objects[track_id]

    def mark_lost(self, track_id: int):
        """
        Mark object as lost (for potential re-identification).

        Args:
            track_id: Track ID
        """
        if track_id in self.appearance_db:
            self.lost_objects[track_id] = time.time()
            logger.debug(f"Object {track_id} marked as lost")

    def try_reidentify(
        self,
        detection: DetectedObject,
        image: np.ndarray,
        camera_id: int
    ) -> Optional[int]:
        """
        Try to re-identify a detection as a lost object.

        Args:
            detection: New detection
            image: Full image
            camera_id: Camera ID

        Returns:
            Track ID if re-identified, None otherwise
        """
        if not self.lost_objects:
            return None

        # Extract features for new detection
        new_features = self.extract_features(image, detection.bbox)

        # Cleanup old lost objects
        current_time = time.time()
        expired = [
            tid for tid, last_seen in self.lost_objects.items()
            if current_time - last_seen > self.occlusion_timeout
        ]
        for tid in expired:
            del self.lost_objects[tid]
            if tid in self.appearance_db:
                del self.appearance_db[tid]

        # Try to match with lost objects
        best_match_id = None
        best_similarity = self.similarity_threshold

        for track_id in self.lost_objects.keys():
            if track_id not in self.appearance_db:
                continue

            # Get recent appearances
            appearances = list(self.appearance_db[track_id])
            if not appearances:
                continue

            # Calculate average similarity to recent appearances
            similarities = []
            for appearance in appearances[-5:]:  # Last 5 appearances
                sim = self._calculate_similarity(new_features, appearance.features)
                similarities.append(sim)

            avg_similarity = np.mean(similarities)

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_id = track_id

        if best_match_id is not None:
            self.total_reidentifications += 1
            logger.info(
                f"Re-identified object {best_match_id} "
                f"(similarity: {best_similarity:.2f})"
            )

            # Remove from lost objects
            del self.lost_objects[best_match_id]

        return best_match_id

    def _calculate_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if features1.size == 0 or features2.size == 0:
            return 0.0

        # Use cosine similarity
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(features1, features2) / (norm1 * norm2)

        # Normalize to 0-1
        similarity = (similarity + 1.0) / 2.0

        return float(similarity)

    def get_cross_camera_matches(
        self,
        track_id: int,
        target_camera_id: int
    ) -> List[ObjectAppearance]:
        """
        Get appearances of an object in a specific camera.

        Args:
            track_id: Track ID
            target_camera_id: Camera to search

        Returns:
            List of appearances in that camera
        """
        if track_id not in self.appearance_db:
            return []

        matches = [
            app for app in self.appearance_db[track_id]
            if app.camera_id == target_camera_id
        ]

        return matches

    def get_appearance_history(
        self,
        track_id: int,
        max_age_seconds: Optional[float] = None
    ) -> List[ObjectAppearance]:
        """
        Get appearance history for an object.

        Args:
            track_id: Track ID
            max_age_seconds: Maximum age of appearances to return

        Returns:
            List of appearances
        """
        if track_id not in self.appearance_db:
            return []

        appearances = list(self.appearance_db[track_id])

        if max_age_seconds is not None:
            cutoff_time = time.time() - max_age_seconds
            appearances = [
                app for app in appearances
                if app.timestamp >= cutoff_time
            ]

        return appearances

    def cleanup_old_appearances(self, active_track_ids: List[int]):
        """
        Remove appearances for tracks that no longer exist.

        Args:
            active_track_ids: List of currently active track IDs
        """
        active_set = set(active_track_ids)

        # Don't remove lost objects (they might come back)
        keep_set = active_set | set(self.lost_objects.keys())

        to_remove = [
            tid for tid in self.appearance_db.keys()
            if tid not in keep_set
        ]

        for track_id in to_remove:
            del self.appearance_db[track_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old appearance records")

    def get_statistics(self) -> Dict:
        """Get re-identification statistics."""
        return {
            "feature_type": self.feature_type,
            "total_reidentifications": self.total_reidentifications,
            "total_feature_extractions": self.total_extractions,
            "objects_in_database": len(self.appearance_db),
            "lost_objects": len(self.lost_objects),
            "similarity_threshold": self.similarity_threshold
        }


class CrossCameraTracker:
    """
    Tracks objects across multiple cameras using re-identification.
    """

    def __init__(self, reidentifier: ObjectReIdentifier):
        """
        Initialize cross-camera tracker.

        Args:
            reidentifier: Object re-identifier instance
        """
        self.reidentifier = reidentifier

        # Global track ID mapping: camera_track_id -> global_track_id
        self.global_track_mapping: Dict[Tuple[int, int], int] = {}

        # Next global ID
        self.next_global_id = 1

        # Track cameras: global_track_id -> set of camera_ids
        self.track_cameras: Dict[int, set] = {}

        logger.info("Cross-camera tracker initialized")

    def update(
        self,
        tracked_objects_by_camera: Dict[int, List[TrackedObject]],
        images_by_camera: Dict[int, np.ndarray]
    ) -> Dict[int, int]:
        """
        Update cross-camera tracking.

        Args:
            tracked_objects_by_camera: Dict of camera_id -> tracked objects
            images_by_camera: Dict of camera_id -> images

        Returns:
            Mapping of (camera_id, local_track_id) -> global_track_id
        """
        # Update appearances for all tracked objects
        for camera_id, tracked_objects in tracked_objects_by_camera.items():
            if camera_id not in images_by_camera:
                continue

            image = images_by_camera[camera_id]

            for obj in tracked_objects:
                # Update appearance database
                self.reidentifier.update_appearance(obj, image, camera_id)

                # Create or update global track mapping
                key = (camera_id, obj.track_id)

                if key not in self.global_track_mapping:
                    # Try to re-identify across cameras
                    global_id = self._try_cross_camera_match(obj, camera_id, image)

                    if global_id is None:
                        # New global track
                        global_id = self.next_global_id
                        self.next_global_id += 1

                    self.global_track_mapping[key] = global_id

                    # Update track cameras
                    if global_id not in self.track_cameras:
                        self.track_cameras[global_id] = set()
                    self.track_cameras[global_id].add(camera_id)

        return self.global_track_mapping

    def _try_cross_camera_match(
        self,
        obj: TrackedObject,
        camera_id: int,
        image: np.ndarray
    ) -> Optional[int]:
        """Try to match object to existing global track from another camera."""
        # Extract features
        features = self.reidentifier.extract_features(image, obj.current_bbox)

        best_match_id = None
        best_similarity = self.reidentifier.similarity_threshold

        # Check all global tracks from OTHER cameras
        for (cam_id, local_id), global_id in self.global_track_mapping.items():
            if cam_id == camera_id:
                continue  # Skip same camera

            # Get appearance history
            appearances = self.reidentifier.get_appearance_history(
                local_id,
                max_age_seconds=10.0
            )

            if not appearances:
                continue

            # Calculate similarity
            similarities = [
                self.reidentifier._calculate_similarity(features, app.features)
                for app in appearances[-3:]  # Last 3 appearances
            ]

            avg_similarity = np.mean(similarities)

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_id = global_id

        if best_match_id is not None:
            logger.info(
                f"Cross-camera match: Camera {camera_id} track {obj.track_id} "
                f"matched to global track {best_match_id} (sim: {best_similarity:.2f})"
            )

        return best_match_id

    def get_global_track_id(self, camera_id: int, local_track_id: int) -> Optional[int]:
        """Get global track ID for a local track."""
        return self.global_track_mapping.get((camera_id, local_track_id))

    def get_track_cameras(self, global_track_id: int) -> set:
        """Get set of cameras that have seen this global track."""
        return self.track_cameras.get(global_track_id, set())

    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        return {
            "global_tracks": self.next_global_id - 1,
            "local_tracks": len(self.global_track_mapping),
            "multi_camera_tracks": sum(
                1 for cameras in self.track_cameras.values()
                if len(cameras) > 1
            )
        }
