"""
Sensor fusion module for combining detections from multiple cameras.

Performs:
- Coordinate transformation to vehicle frame
- Object association across cameras
- Detection deduplication
- Confidence aggregation
- Unified object list generation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.spatial import distance as dist

from utils.data_structures import (
    DetectedObject,
    TrackedObject,
    CameraPosition,
    CameraConfig
)
from utils.coordinate_transforms import CoordinateTransformer
from utils.logger import get_logger


logger = get_logger()


@dataclass
class FusionConfig:
    """Configuration for sensor fusion."""
    # Spatial thresholds
    spatial_threshold_meters: float = 2.0  # Max distance for object association

    # Object association
    class_match_required: bool = True  # Must match object class
    confidence_weight: float = 0.3  # Weight for confidence in matching
    distance_weight: float = 0.7  # Weight for spatial distance in matching

    # Confidence aggregation
    confidence_aggregation: str = "max"  # "max", "mean", "weighted_mean"

    # Deduplication
    min_confidence_for_fusion: float = 0.3
    max_objects_per_fusion: int = 100


class SensorFusion:
    """
    Fuses detections from multiple cameras into a unified representation.

    Handles coordinate transformations, object association, and deduplication.
    """

    def __init__(
        self,
        camera_configs: Dict[int, CameraConfig],
        config: Optional[FusionConfig] = None
    ):
        """
        Initialize sensor fusion module.

        Args:
            camera_configs: Dictionary mapping camera_id to CameraConfig
            config: Fusion configuration
        """
        self.camera_configs = camera_configs
        self.config = config or FusionConfig()

        # Coordinate transformer
        self.coord_transformer = CoordinateTransformer()

        # Statistics
        self.total_fusions = 0
        self.total_objects_before_fusion = 0
        self.total_objects_after_fusion = 0

        logger.info("Sensor fusion module initialized")

    def fuse_detections(
        self,
        detections_by_camera: Dict[int, List[DetectedObject]]
    ) -> List[DetectedObject]:
        """
        Fuse detections from multiple cameras.

        Args:
            detections_by_camera: Dictionary mapping camera_id to list of detections

        Returns:
            List of fused detections
        """
        if not detections_by_camera:
            return []

        # Transform all detections to vehicle coordinates
        vehicle_detections = self._transform_to_vehicle_coords(detections_by_camera)

        # Count before fusion
        total_before = sum(len(dets) for dets in detections_by_camera.values())
        self.total_objects_before_fusion += total_before

        # Associate and deduplicate
        fused_detections = self._associate_and_deduplicate(vehicle_detections)

        # Count after fusion
        self.total_objects_after_fusion += len(fused_detections)
        self.total_fusions += 1

        return fused_detections

    def _transform_to_vehicle_coords(
        self,
        detections_by_camera: Dict[int, List[DetectedObject]]
    ) -> List[DetectedObject]:
        """
        Transform all detections to vehicle coordinate system.

        Args:
            detections_by_camera: Detections organized by camera

        Returns:
            List of detections with vehicle coordinates
        """
        vehicle_detections = []

        for camera_id, detections in detections_by_camera.items():
            if camera_id not in self.camera_configs:
                logger.warning(f"No config for camera {camera_id}, skipping")
                continue

            camera_config = self.camera_configs[camera_id]

            for detection in detections:
                # Skip low confidence detections
                if detection.confidence < self.config.min_confidence_for_fusion:
                    continue

                # Estimate 3D position if not already available
                if detection.position_3d is None and detection.distance is not None:
                    # Use bounding box to estimate position
                    x1, y1, x2, y2 = detection.bbox
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Simplified: assume object is at center of bbox in image
                    # and use distance estimate
                    # This would be improved with actual camera calibration
                    position_3d = self._estimate_3d_position(
                        cx, cy, detection.distance, camera_config
                    )

                    # Transform to vehicle coordinates
                    if position_3d:
                        vehicle_pos = self.coord_transformer.camera_to_vehicle_coords(
                            position_3d[0],
                            position_3d[1],
                            position_3d[2],
                            camera_config.position_offset,
                            camera_config.orientation
                        )
                        detection.position_3d = vehicle_pos

                vehicle_detections.append(detection)

        return vehicle_detections

    def _estimate_3d_position(
        self,
        pixel_x: float,
        pixel_y: float,
        distance: float,
        camera_config: CameraConfig
    ) -> Optional[Tuple[float, float, float]]:
        """
        Estimate 3D position in camera coordinates from pixel location and distance.

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            distance: Distance from camera in meters
            camera_config: Camera configuration

        Returns:
            (x, y, z) in camera coordinates or None
        """
        # Simplified estimation
        # In a full implementation, this would use camera calibration

        # Assume camera is pointing forward
        # X is horizontal (left-right)
        # Y is vertical (up-down)
        # Z is depth (forward)

        width, height = camera_config.resolution

        # Normalize pixel coordinates to [-1, 1]
        norm_x = (pixel_x - width / 2) / (width / 2)
        norm_y = (pixel_y - height / 2) / (height / 2)

        # Estimate angles (simplified, assumes ~60 degree FOV)
        fov_h = np.radians(60)
        fov_v = np.radians(45)

        angle_x = norm_x * (fov_h / 2)
        angle_y = norm_y * (fov_v / 2)

        # Convert to 3D coordinates
        x = distance * np.tan(angle_x)
        y = distance * np.tan(angle_y)
        z = distance

        return (x, y, z)

    def _associate_and_deduplicate(
        self,
        detections: List[DetectedObject]
    ) -> List[DetectedObject]:
        """
        Associate and deduplicate detections from multiple cameras.

        Args:
            detections: List of detections in vehicle coordinates

        Returns:
            List of fused, deduplicated detections
        """
        if not detections:
            return []

        # Filter detections with valid 3D positions
        valid_detections = [d for d in detections if d.position_3d is not None]

        if not valid_detections:
            return detections  # Return original if no valid positions

        # Build groups of associated detections
        groups = self._build_association_groups(valid_detections)

        # Fuse each group into a single detection
        fused = []
        for group in groups:
            fused_detection = self._fuse_group(group)
            fused.append(fused_detection)

        return fused

    def _build_association_groups(
        self,
        detections: List[DetectedObject]
    ) -> List[List[DetectedObject]]:
        """
        Build groups of detections that likely represent the same object.

        Args:
            detections: List of detections

        Returns:
            List of detection groups
        """
        # Use a union-find approach to group nearby detections
        n = len(detections)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build distance matrix
        positions = np.array([d.position_3d[:2] for d in detections])  # Use x, y only
        D = dist.cdist(positions, positions)

        # Associate detections within threshold
        for i in range(n):
            for j in range(i + 1, n):
                det_i = detections[i]
                det_j = detections[j]

                # Check spatial proximity
                if D[i, j] > self.config.spatial_threshold_meters:
                    continue

                # Check class match if required
                if self.config.class_match_required:
                    if det_i.class_id != det_j.class_id:
                        continue

                # Check if from different cameras (avoid associating duplicates from same camera)
                if det_i.camera_id == det_j.camera_id:
                    continue

                # Associate
                union(i, j)

        # Build groups
        groups_dict: Dict[int, List[DetectedObject]] = {}
        for i in range(n):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(detections[i])

        return list(groups_dict.values())

    def _fuse_group(
        self,
        group: List[DetectedObject]
    ) -> DetectedObject:
        """
        Fuse a group of detections into a single detection.

        Args:
            group: List of detections representing the same object

        Returns:
            Fused detection
        """
        if len(group) == 1:
            return group[0]

        # Use detection with highest confidence as base
        base_detection = max(group, key=lambda d: d.confidence)

        # Aggregate confidence
        if self.config.confidence_aggregation == "max":
            fused_confidence = max(d.confidence for d in group)
        elif self.config.confidence_aggregation == "mean":
            fused_confidence = np.mean([d.confidence for d in group])
        else:  # weighted_mean
            weights = np.array([d.confidence for d in group])
            weights = weights / weights.sum()
            fused_confidence = np.sum(weights * np.array([d.confidence for d in group]))

        # Average 3D position
        positions = [d.position_3d for d in group if d.position_3d]
        if positions:
            avg_position = tuple(np.mean(positions, axis=0))
        else:
            avg_position = base_detection.position_3d

        # Average distance
        distances = [d.distance for d in group if d.distance]
        avg_distance = np.mean(distances) if distances else base_detection.distance

        # Create fused detection
        fused = DetectedObject(
            object_id=base_detection.object_id,
            class_id=base_detection.class_id,
            class_name=base_detection.class_name,
            confidence=float(fused_confidence),
            bbox=base_detection.bbox,  # Keep bbox from highest confidence detection
            camera_id=base_detection.camera_id,  # Primary camera
            timestamp=base_detection.timestamp,
            position_3d=avg_position,
            velocity=base_detection.velocity,
            distance=float(avg_distance) if avg_distance else None,
            metadata={
                "fused_from_cameras": [d.camera_id for d in group],
                "num_cameras": len(group),
                "original_confidences": [d.confidence for d in group]
            }
        )

        return fused

    def get_statistics(self) -> Dict:
        """
        Get fusion statistics.

        Returns:
            Dictionary with statistics
        """
        avg_reduction = 0.0
        if self.total_fusions > 0 and self.total_objects_before_fusion > 0:
            avg_objects_before = self.total_objects_before_fusion / self.total_fusions
            avg_objects_after = self.total_objects_after_fusion / self.total_fusions
            avg_reduction = (avg_objects_before - avg_objects_after) / avg_objects_before * 100

        return {
            "total_fusions": self.total_fusions,
            "total_objects_before_fusion": self.total_objects_before_fusion,
            "total_objects_after_fusion": self.total_objects_after_fusion,
            "average_reduction_percent": avg_reduction,
            "spatial_threshold_m": self.config.spatial_threshold_meters
        }

    def reset(self):
        """Reset statistics."""
        self.total_fusions = 0
        self.total_objects_before_fusion = 0
        self.total_objects_after_fusion = 0
        logger.info("Sensor fusion statistics reset")
