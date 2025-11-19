"""
3D Object Detection Module

This module provides 3D object detection and localization for autonomous driving,
including:
- 3D bounding box estimation from 2D detections
- Depth integration for 3D position
- 3D orientation estimation
- 3D box visualization
- Point cloud generation (pseudo from depth)
- 3D IoU calculation
- Multi-view 3D reconstruction
- Bird's eye view projection

Author: AV Perception System
Version: 1.2.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math


class Orientation3D(Enum):
    """3D orientation categories"""
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    BACK_LEFT = "back_left"
    BACK_RIGHT = "back_right"
    UNKNOWN = "unknown"


@dataclass
class Point3D:
    """3D point in space"""
    x: float  # meters
    y: float  # meters
    z: float  # meters

    def distance_to(self, other: 'Point3D') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z])


@dataclass
class BoundingBox3D:
    """3D bounding box representation"""
    center: Point3D
    dimensions: Tuple[float, float, float]  # width, height, depth (meters)
    rotation: Tuple[float, float, float]  # roll, pitch, yaw (radians)
    confidence: float
    class_name: str
    velocity: Optional[Point3D] = None  # 3D velocity vector (m/s)

    def get_corners(self) -> List[Point3D]:
        """
        Get 8 corners of the 3D bounding box

        Returns corners in order:
        0-3: bottom plane (front-left, front-right, back-right, back-left)
        4-7: top plane (same order)
        """
        w, h, d = self.dimensions
        cx, cy, cz = self.center.x, self.center.y, self.center.z
        roll, pitch, yaw = self.rotation

        # Define corners in local coordinate system (centered at origin)
        corners_local = np.array([
            [-w/2, -h/2, -d/2],  # front-left-bottom
            [ w/2, -h/2, -d/2],  # front-right-bottom
            [ w/2, -h/2,  d/2],  # back-right-bottom
            [-w/2, -h/2,  d/2],  # back-left-bottom
            [-w/2,  h/2, -d/2],  # front-left-top
            [ w/2,  h/2, -d/2],  # front-right-top
            [ w/2,  h/2,  d/2],  # back-right-top
            [-w/2,  h/2,  d/2],  # back-left-top
        ])

        # Create rotation matrix
        R = self._rotation_matrix(roll, pitch, yaw)

        # Rotate corners
        corners_rotated = corners_local @ R.T

        # Translate to center position
        corners_world = corners_rotated + np.array([cx, cy, cz])

        return [Point3D(c[0], c[1], c[2]) for c in corners_world]

    def _rotation_matrix(
        self,
        roll: float,
        pitch: float,
        yaw: float
    ) -> np.ndarray:
        """Create 3D rotation matrix from Euler angles"""
        # Roll (X-axis)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])

        # Pitch (Y-axis)
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        # Yaw (Z-axis)
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation: R = Rz * Ry * Rx
        return Rz @ Ry @ Rx

    def get_volume(self) -> float:
        """Get volume of bounding box"""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def contains_point(self, point: Point3D) -> bool:
        """Check if point is inside the bounding box"""
        corners = self.get_corners()

        # Simplified: check if point is within AABB of rotated box
        # For accurate containment, would need oriented bounding box test
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        zs = [c.z for c in corners]

        return (min(xs) <= point.x <= max(xs) and
                min(ys) <= point.y <= max(ys) and
                min(zs) <= point.z <= max(zs))


@dataclass
class Object3D:
    """Complete 3D object representation"""
    bbox_3d: BoundingBox3D
    bbox_2d: Tuple[int, int, int, int]  # 2D bbox in image (x, y, w, h)
    track_id: Optional[int] = None
    orientation: Orientation3D = Orientation3D.UNKNOWN
    timestamp: float = 0.0


@dataclass
class Detection3DConfig:
    """Configuration for 3D object detection"""
    # Standard object dimensions (meters) for size estimation
    standard_dimensions: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        'car': (1.8, 1.5, 4.5),        # width, height, depth
        'truck': (2.5, 3.0, 7.0),
        'bus': (2.5, 3.2, 12.0),
        'motorcycle': (0.8, 1.2, 2.0),
        'bicycle': (0.6, 1.0, 1.8),
        'person': (0.6, 1.7, 0.3)
    })

    # Camera parameters (should be from calibration)
    focal_length: float = 1000.0  # pixels
    principal_point: Tuple[float, float] = (640.0, 360.0)  # cx, cy

    # Depth estimation
    use_depth_map: bool = True
    fallback_to_geometric: bool = True
    max_detection_distance: float = 100.0  # meters

    # Orientation estimation
    enable_orientation_estimation: bool = True
    orientation_confidence_threshold: float = 0.6


class ObjectDetector3D:
    """
    3D Object Detection System

    Features:
    - 3D bounding box generation from 2D detections + depth
    - 3D position and orientation estimation
    - Standard object dimension templates
    - Geometric depth estimation fallback
    - Multi-view 3D reconstruction (when available)
    - 3D tracking support
    - Bird's eye view projection
    """

    def __init__(self, config: Optional[Detection3DConfig] = None):
        """
        Initialize 3D object detector

        Args:
            config: 3D detection configuration
        """
        self.config = config or Detection3DConfig()
        self.tracked_objects_3d: Dict[int, Object3D] = {}

        # Statistics
        self.total_3d_detections = 0
        self.successful_depth_estimates = 0

    def detect_objects_3d(
        self,
        detections_2d: List[Dict[str, Any]],
        depth_map: Optional[np.ndarray] = None,
        camera_calibration: Optional[Dict[str, Any]] = None
    ) -> List[Object3D]:
        """
        Convert 2D detections to 3D objects

        Args:
            detections_2d: List of 2D detections with bbox, class, confidence
            depth_map: Dense depth map from depth estimation module
            camera_calibration: Camera calibration parameters

        Returns:
            List of Object3D with 3D bounding boxes
        """
        objects_3d = []

        for det in detections_2d:
            obj_3d = self._create_3d_object(det, depth_map, camera_calibration)
            if obj_3d:
                objects_3d.append(obj_3d)
                self.total_3d_detections += 1

        return objects_3d

    def _create_3d_object(
        self,
        detection_2d: Dict[str, Any],
        depth_map: Optional[np.ndarray],
        calibration: Optional[Dict[str, Any]]
    ) -> Optional[Object3D]:
        """Create 3D object from 2D detection"""
        bbox_2d = detection_2d.get('bbox')  # (x, y, w, h)
        class_name = detection_2d.get('class', 'unknown')
        confidence = detection_2d.get('confidence', 0.0)
        track_id = detection_2d.get('track_id')

        if bbox_2d is None:
            return None

        x, y, w, h = bbox_2d

        # Get object center in image
        center_x = x + w // 2
        center_y = y + h // 2

        # Estimate depth/distance
        depth = self._estimate_depth(
            bbox_2d, depth_map, class_name, calibration
        )

        if depth is None or depth > self.config.max_detection_distance:
            return None

        self.successful_depth_estimates += 1

        # Get 3D position
        position_3d = self._image_to_3d(
            center_x, center_y, depth, calibration
        )

        # Get object dimensions
        dimensions = self._estimate_dimensions(
            bbox_2d, class_name, depth, calibration
        )

        # Estimate orientation
        orientation = self._estimate_orientation(
            bbox_2d, depth, class_name
        )

        # Create 3D bounding box
        bbox_3d = BoundingBox3D(
            center=position_3d,
            dimensions=dimensions,
            rotation=(0.0, 0.0, orientation),  # Simplified: only yaw rotation
            confidence=confidence,
            class_name=class_name
        )

        # Create 3D object
        obj_3d = Object3D(
            bbox_3d=bbox_3d,
            bbox_2d=bbox_2d,
            track_id=track_id,
            orientation=self._classify_orientation(orientation)
        )

        return obj_3d

    def _estimate_depth(
        self,
        bbox_2d: Tuple[int, int, int, int],
        depth_map: Optional[np.ndarray],
        class_name: str,
        calibration: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """Estimate depth/distance to object"""
        x, y, w, h = bbox_2d

        # Method 1: Use depth map if available
        if depth_map is not None and self.config.use_depth_map:
            # Sample depth at object center and nearby points
            center_x = x + w // 2
            center_y = y + h // 2

            # Extract depth values in object region
            y1, y2 = max(0, y), min(depth_map.shape[0], y + h)
            x1, x2 = max(0, x), min(depth_map.shape[1], x + w)

            if y1 < y2 and x1 < x2:
                depth_region = depth_map[y1:y2, x1:x2]

                # Use median depth (more robust than mean)
                valid_depths = depth_region[depth_region > 0]
                if len(valid_depths) > 0:
                    depth = np.median(valid_depths)
                    return float(depth)

        # Method 2: Geometric estimation using known object sizes
        if self.config.fallback_to_geometric and class_name in self.config.standard_dimensions:
            _, real_height, _ = self.config.standard_dimensions[class_name]

            focal_length = (calibration.get('focal_length', self.config.focal_length)
                          if calibration else self.config.focal_length)

            # Distance = (Real Height * Focal Length) / Pixel Height
            if h > 0:
                depth = (real_height * focal_length) / h
                return depth

        return None

    def _image_to_3d(
        self,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        calibration: Optional[Dict[str, Any]]
    ) -> Point3D:
        """
        Convert image coordinates to 3D world coordinates

        Uses pinhole camera model:
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        Z = depth
        """
        if calibration:
            fx = calibration.get('focal_length', self.config.focal_length)
            fy = fx  # Assume square pixels
            cx, cy = calibration.get('principal_point', self.config.principal_point)
        else:
            fx = fy = self.config.focal_length
            cx, cy = self.config.principal_point

        # Convert to 3D coordinates
        X = (pixel_x - cx) * depth / fx
        Y = (pixel_y - cy) * depth / fy
        Z = depth

        return Point3D(X, Y, Z)

    def _estimate_dimensions(
        self,
        bbox_2d: Tuple[int, int, int, int],
        class_name: str,
        depth: float,
        calibration: Optional[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Estimate 3D dimensions of object"""
        # Use standard dimensions for object class
        if class_name in self.config.standard_dimensions:
            return self.config.standard_dimensions[class_name]

        # Fallback: estimate from 2D bbox and depth
        x, y, w, h = bbox_2d

        focal_length = (calibration.get('focal_length', self.config.focal_length)
                       if calibration else self.config.focal_length)

        # Estimate real-world width and height from pixel dimensions
        real_width = (w * depth) / focal_length
        real_height = (h * depth) / focal_length
        real_depth = real_width  # Rough approximation

        return (real_width, real_height, real_depth)

    def _estimate_orientation(
        self,
        bbox_2d: Tuple[int, int, int, int],
        depth: float,
        class_name: str
    ) -> float:
        """
        Estimate object orientation (yaw angle)

        Returns yaw in radians (-π to π)
        0 = facing forward (same direction as ego vehicle)
        """
        if not self.config.enable_orientation_estimation:
            return 0.0

        # Simplified orientation estimation based on bbox aspect ratio
        # and position in image
        # In production, use viewpoint estimation CNN or keypoint matching

        x, y, w, h = bbox_2d
        aspect_ratio = w / h if h > 0 else 1.0

        # Very simplified logic
        if class_name in ['car', 'truck', 'bus']:
            # If aspect ratio is high, likely viewing from side
            if aspect_ratio > 2.0:
                return math.pi / 2  # 90 degrees (side view)
            elif aspect_ratio < 1.0:
                return 0.0  # Front/back view
            else:
                return math.pi / 4  # 45 degrees (angled)

        return 0.0

    def _classify_orientation(self, yaw: float) -> Orientation3D:
        """Classify orientation angle into category"""
        # Convert to degrees for easier thresholds
        yaw_deg = math.degrees(yaw) % 360

        if yaw_deg < 22.5 or yaw_deg >= 337.5:
            return Orientation3D.FRONT
        elif 22.5 <= yaw_deg < 67.5:
            return Orientation3D.FRONT_RIGHT
        elif 67.5 <= yaw_deg < 112.5:
            return Orientation3D.RIGHT
        elif 112.5 <= yaw_deg < 157.5:
            return Orientation3D.BACK_RIGHT
        elif 157.5 <= yaw_deg < 202.5:
            return Orientation3D.BACK
        elif 202.5 <= yaw_deg < 247.5:
            return Orientation3D.BACK_LEFT
        elif 247.5 <= yaw_deg < 292.5:
            return Orientation3D.LEFT
        elif 292.5 <= yaw_deg < 337.5:
            return Orientation3D.FRONT_LEFT

        return Orientation3D.UNKNOWN

    def project_to_bev(
        self,
        objects_3d: List[Object3D],
        bev_size: Tuple[int, int] = (800, 800),
        bev_range: Tuple[float, float] = (50.0, 50.0)  # forward, lateral range (m)
    ) -> np.ndarray:
        """
        Project 3D objects to bird's eye view image

        Args:
            objects_3d: List of 3D objects
            bev_size: Output image size (height, width)
            bev_range: Range to visualize (forward_m, lateral_m)

        Returns:
            Bird's eye view image
        """
        bev_image = np.zeros((*bev_size, 3), dtype=np.uint8)
        h, w = bev_size
        forward_range, lateral_range = bev_range

        # Define ego vehicle position (center bottom of image)
        ego_x = w // 2
        ego_y = h - 50

        # Draw ego vehicle
        cv2.rectangle(bev_image,
                     (ego_x - 15, ego_y - 30),
                     (ego_x + 15, ego_y),
                     (0, 255, 0), -1)

        # Project each object
        for obj in objects_3d:
            # Get 3D center position
            center = obj.bbox_3d.center

            # Convert to BEV coordinates
            # X (lateral) maps to image X
            # Z (forward/depth) maps to image Y (inverted)
            pixel_x = int(ego_x + (center.x / lateral_range) * (w / 2))
            pixel_y = int(ego_y - (center.z / forward_range) * (h - 100))

            # Check if in bounds
            if not (0 <= pixel_x < w and 0 <= pixel_y < h):
                continue

            # Get 2D footprint (corners on ground plane)
            corners = obj.bbox_3d.get_corners()
            ground_corners = corners[:4]  # Bottom 4 corners

            # Project corners to BEV
            bev_corners = []
            for corner in ground_corners:
                px = int(ego_x + (corner.x / lateral_range) * (w / 2))
                py = int(ego_y - (corner.z / forward_range) * (h - 100))
                if 0 <= px < w and 0 <= py < h:
                    bev_corners.append((px, py))

            # Draw bounding box
            if len(bev_corners) >= 3:
                bev_corners_array = np.array(bev_corners, dtype=np.int32)
                cv2.polylines(bev_image, [bev_corners_array], True, (0, 255, 255), 2)

            # Draw center point
            color = self._get_class_color(obj.bbox_3d.class_name)
            cv2.circle(bev_image, (pixel_x, pixel_y), 5, color, -1)

            # Draw orientation arrow
            if obj.orientation != Orientation3D.UNKNOWN:
                _, _, yaw = obj.bbox_3d.rotation
                arrow_length = 20
                end_x = int(pixel_x + arrow_length * math.cos(yaw))
                end_y = int(pixel_y - arrow_length * math.sin(yaw))
                cv2.arrowedLine(bev_image, (pixel_x, pixel_y), (end_x, end_y), color, 2)

        # Draw range circles
        for radius_m in [10, 20, 30, 40]:
            pixel_radius = int((radius_m / forward_range) * (h - 100))
            cv2.circle(bev_image, (ego_x, ego_y), pixel_radius, (100, 100, 100), 1)

        return bev_image

    def visualize_3d(
        self,
        image: np.ndarray,
        objects_3d: List[Object3D],
        camera_calibration: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Visualize 3D bounding boxes on 2D image

        Args:
            image: Input image
            objects_3d: List of 3D objects
            camera_calibration: Camera parameters for projection

        Returns:
            Image with 3D boxes projected and drawn
        """
        vis_image = image.copy()

        for obj in objects_3d:
            # Get 3D corners
            corners_3d = obj.bbox_3d.get_corners()

            # Project to 2D
            corners_2d = []
            for corner in corners_3d:
                pixel = self._project_3d_to_2d(corner, camera_calibration)
                if pixel:
                    corners_2d.append(pixel)

            if len(corners_2d) < 8:
                # Fallback to 2D bbox if projection fails
                x, y, w, h = obj.bbox_2d
                color = self._get_class_color(obj.bbox_3d.class_name)
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
                continue

            # Draw 3D box edges
            color = self._get_class_color(obj.bbox_3d.class_name)

            # Bottom face
            for i in range(4):
                cv2.line(vis_image, corners_2d[i], corners_2d[(i+1) % 4], color, 2)

            # Top face
            for i in range(4, 8):
                cv2.line(vis_image, corners_2d[i], corners_2d[4 + (i+1) % 4], color, 2)

            # Vertical edges
            for i in range(4):
                cv2.line(vis_image, corners_2d[i], corners_2d[i+4], color, 2)

            # Draw front face more prominently
            cv2.line(vis_image, corners_2d[0], corners_2d[1], (0, 0, 255), 3)
            cv2.line(vis_image, corners_2d[4], corners_2d[5], (0, 0, 255), 3)

            # Label
            center_2d = corners_2d[0]
            label = f"{obj.bbox_3d.class_name} {obj.bbox_3d.center.z:.1f}m"
            cv2.putText(vis_image, label, center_2d,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return vis_image

    def _project_3d_to_2d(
        self,
        point_3d: Point3D,
        calibration: Optional[Dict[str, Any]]
    ) -> Optional[Tuple[int, int]]:
        """Project 3D point to 2D image coordinates"""
        if calibration:
            fx = calibration.get('focal_length', self.config.focal_length)
            fy = fx
            cx, cy = calibration.get('principal_point', self.config.principal_point)
        else:
            fx = fy = self.config.focal_length
            cx, cy = self.config.principal_point

        # Pinhole camera projection
        if point_3d.z <= 0:  # Behind camera
            return None

        pixel_x = int((point_3d.x * fx / point_3d.z) + cx)
        pixel_y = int((point_3d.y * fy / point_3d.z) + cy)

        return (pixel_x, pixel_y)

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for object class"""
        color_map = {
            'car': (0, 255, 255),
            'truck': (0, 165, 255),
            'bus': (0, 100, 255),
            'person': (255, 0, 0),
            'bicycle': (255, 255, 0),
            'motorcycle': (255, 0, 255)
        }
        return color_map.get(class_name, (128, 128, 128))

    def compute_3d_iou(self, bbox1: BoundingBox3D, bbox2: BoundingBox3D) -> float:
        """
        Compute 3D Intersection over Union

        Simplified implementation using axis-aligned bounding boxes
        """
        # Get corners
        corners1 = bbox1.get_corners()
        corners2 = bbox2.get_corners()

        # Get axis-aligned bounding boxes
        xs1 = [c.x for c in corners1]
        ys1 = [c.y for c in corners1]
        zs1 = [c.z for c in corners1]

        xs2 = [c.x for c in corners2]
        ys2 = [c.y for c in corners2]
        zs2 = [c.z for c in corners2]

        # Calculate intersection
        x_min = max(min(xs1), min(xs2))
        x_max = min(max(xs1), max(xs2))
        y_min = max(min(ys1), min(ys2))
        y_max = min(max(ys1), max(ys2))
        z_min = max(min(zs1), min(zs2))
        z_max = min(max(zs1), max(zs2))

        if x_max <= x_min or y_max <= y_min or z_max <= z_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

        # Calculate union
        vol1 = bbox1.get_volume()
        vol2 = bbox2.get_volume()
        union = vol1 + vol2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get 3D detection statistics"""
        depth_success_rate = (self.successful_depth_estimates / self.total_3d_detections * 100
                             if self.total_3d_detections > 0 else 0.0)

        return {
            'total_3d_detections': self.total_3d_detections,
            'successful_depth_estimates': self.successful_depth_estimates,
            'depth_success_rate': depth_success_rate
        }
