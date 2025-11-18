"""
Coordinate transformation utilities for multi-camera perception.

Handles transformations between:
- Image coordinates (pixels)
- Camera coordinates (3D)
- Vehicle coordinates (3D, vehicle-centric)
- World coordinates (future: GPS-based)
"""

import numpy as np
from typing import Tuple, Optional
import cv2


class CoordinateTransformer:
    """
    Handles coordinate transformations for the perception system.
    """

    def __init__(self):
        """Initialize the coordinate transformer."""
        self.vehicle_length = 4.5  # meters (typical car)
        self.vehicle_width = 1.8   # meters (typical car)

    def pixel_to_camera_coords(
        self,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        camera_matrix: np.ndarray,
        distortion_coeffs: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to 3D camera coordinates.

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            depth: Depth (distance from camera) in meters
            camera_matrix: Camera intrinsic matrix (3x3)
            distortion_coeffs: Distortion coefficients (optional)

        Returns:
            (x, y, z) in camera coordinate system (meters)
        """
        # Undistort pixel if distortion coefficients provided
        if distortion_coeffs is not None:
            point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(point, camera_matrix, distortion_coeffs, P=camera_matrix)
            pixel_x, pixel_y = undistorted[0, 0]

        # Extract camera parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Convert to camera coordinates
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth

        return (x, y, z)

    def camera_to_vehicle_coords(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        camera_position: Tuple[float, float, float],
        camera_orientation: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Transform from camera coordinates to vehicle coordinates.

        Args:
            camera_x, camera_y, camera_z: Point in camera coordinates
            camera_position: Camera position (x, y, z) relative to vehicle center
            camera_orientation: Camera orientation (roll, pitch, yaw) in degrees

        Returns:
            (x, y, z) in vehicle coordinate system (meters)
            Vehicle coordinates: x forward, y left, z up, origin at vehicle center
        """
        # Create rotation matrix from camera orientation
        roll, pitch, yaw = np.radians(camera_orientation)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Point in camera frame
        point_camera = np.array([camera_x, camera_y, camera_z])

        # Rotate to vehicle frame
        point_rotated = R @ point_camera

        # Translate by camera position
        vehicle_x = point_rotated[0] + camera_position[0]
        vehicle_y = point_rotated[1] + camera_position[1]
        vehicle_z = point_rotated[2] + camera_position[2]

        return (vehicle_x, vehicle_y, vehicle_z)

    def vehicle_to_bev_coords(
        self,
        vehicle_x: float,
        vehicle_y: float,
        bev_width: int,
        bev_height: int,
        meters_per_pixel: float = 0.05
    ) -> Tuple[int, int]:
        """
        Convert vehicle coordinates to bird's eye view image coordinates.

        Args:
            vehicle_x: X in vehicle coordinates (forward, meters)
            vehicle_y: Y in vehicle coordinates (left, meters)
            bev_width: BEV image width in pixels
            bev_height: BEV image height in pixels
            meters_per_pixel: Scale factor

        Returns:
            (pixel_x, pixel_y) in BEV image
        """
        # BEV origin is at bottom center
        # X axis goes up in image (forward in vehicle)
        # Y axis goes right in image (left in vehicle becomes right in image)

        pixel_x = int(bev_width / 2 - vehicle_y / meters_per_pixel)
        pixel_y = int(bev_height - vehicle_x / meters_per_pixel)

        return (pixel_x, pixel_y)

    def estimate_distance_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        object_real_height: float,
        camera_matrix: np.ndarray,
        image_height: int
    ) -> float:
        """
        Estimate distance to object from bounding box size.

        This is a simplified approach assuming known object height.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            object_real_height: Actual object height in meters
            camera_matrix: Camera intrinsic matrix
            image_height: Image height in pixels

        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1

        if bbox_height <= 0:
            return 100.0  # Default far distance

        # Focal length in pixels
        fy = camera_matrix[1, 1]

        # Distance estimation
        distance = (object_real_height * fy) / bbox_height

        return max(0.1, distance)  # Minimum 0.1 meters

    def get_homography_for_bev(
        self,
        camera_matrix: np.ndarray,
        camera_height: float,
        camera_tilt: float,
        image_width: int,
        image_height: int
    ) -> np.ndarray:
        """
        Calculate homography matrix for bird's eye view transformation.

        Args:
            camera_matrix: Camera intrinsic matrix
            camera_height: Height of camera above ground (meters)
            camera_tilt: Camera tilt angle in degrees (pitch)
            image_width: Source image width
            image_height: Source image height

        Returns:
            3x3 homography matrix
        """
        # Define source points (trapezoid in image)
        # These represent the road plane in the image
        src_points = np.float32([
            [image_width * 0.1, image_height * 0.95],   # Bottom left
            [image_width * 0.9, image_height * 0.95],   # Bottom right
            [image_width * 0.4, image_height * 0.6],    # Top left
            [image_width * 0.6, image_height * 0.6]     # Top right
        ])

        # Define destination points (rectangle in BEV)
        # This represents the road in top-down view
        dst_width = 400
        dst_height = 600

        dst_points = np.float32([
            [dst_width * 0.2, dst_height],               # Bottom left
            [dst_width * 0.8, dst_height],               # Bottom right
            [dst_width * 0.2, 0],                        # Top left
            [dst_width * 0.8, 0]                         # Top right
        ])

        # Calculate homography
        H = cv2.getPerspectiveTransform(src_points, dst_points)

        return H

    def apply_homography(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply homography transformation to image.

        Args:
            image: Input image
            homography: 3x3 homography matrix
            output_size: (width, height) of output image

        Returns:
            Transformed image
        """
        warped = cv2.warpPerspective(
            image,
            homography,
            output_size,
            flags=cv2.INTER_LINEAR
        )

        return warped
