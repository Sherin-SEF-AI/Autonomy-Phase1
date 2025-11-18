"""
Bird's Eye View (BEV) generator for creating top-down visualizations.

Creates a top-down view of the vehicle's surroundings showing:
- Vehicle position (center)
- Detected objects
- Camera field of view cones
- Distance markers
- Occupancy grid
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from utils.data_structures import DetectedObject, TrackedObject, CameraConfig, CameraPosition
from utils.logger import get_logger


logger = get_logger()


@dataclass
class BEVConfig:
    """Configuration for bird's eye view generation."""
    # BEV dimensions
    width_pixels: int = 400
    height_pixels: int = 600
    meters_per_pixel: float = 0.05  # 5cm per pixel

    # View range
    forward_range_m: float = 30.0  # Meters forward
    rear_range_m: float = 10.0  # Meters behind
    lateral_range_m: float = 10.0  # Meters to each side

    # Vehicle dimensions (for drawing)
    vehicle_length_m: float = 4.5
    vehicle_width_m: float = 1.8

    # Visualization options
    show_grid: bool = True
    show_distance_circles: bool = True
    show_camera_fov: bool = True
    show_trajectories: bool = True

    # Colors (BGR)
    background_color: Tuple[int, int, int] = (30, 30, 30)
    grid_color: Tuple[int, int, int] = (60, 60, 60)
    vehicle_color: Tuple[int, int, int] = (0, 255, 0)
    object_color: Tuple[int, int, int] = (255, 128, 0)
    trajectory_color: Tuple[int, int, int] = (255, 128, 255)
    fov_color: Tuple[int, int, int] = (100, 100, 100)


class BEVGenerator:
    """
    Generates bird's eye view visualizations.

    Creates a top-down view showing the vehicle, detected objects,
    and other relevant information.
    """

    def __init__(self, config: Optional[BEVConfig] = None):
        """
        Initialize BEV generator.

        Args:
            config: BEV configuration
        """
        self.config = config or BEVConfig()

        logger.info(f"BEV generator initialized ({self.config.width_pixels}x{self.config.height_pixels})")

    def generate(
        self,
        detections: Optional[List[DetectedObject]] = None,
        tracked_objects: Optional[List[TrackedObject]] = None,
        camera_configs: Optional[Dict[int, CameraConfig]] = None
    ) -> np.ndarray:
        """
        Generate bird's eye view image.

        Args:
            detections: List of detected objects
            tracked_objects: List of tracked objects
            camera_configs: Camera configurations for FOV visualization

        Returns:
            BEV image (BGR format)
        """
        # Create blank BEV image
        bev = np.full(
            (self.config.height_pixels, self.config.width_pixels, 3),
            self.config.background_color,
            dtype=np.uint8
        )

        # Draw grid
        if self.config.show_grid:
            bev = self._draw_grid(bev)

        # Draw distance circles
        if self.config.show_distance_circles:
            bev = self._draw_distance_circles(bev)

        # Draw camera field of view
        if self.config.show_camera_fov and camera_configs:
            bev = self._draw_camera_fov(bev, camera_configs)

        # Draw tracked objects with trajectories
        if tracked_objects:
            bev = self._draw_tracked_objects(bev, tracked_objects)

        # Draw detections (if no tracked objects)
        elif detections:
            bev = self._draw_detections(bev, detections)

        # Draw ego vehicle (always on top)
        bev = self._draw_ego_vehicle(bev)

        # Add labels and legend
        bev = self._add_labels(bev)

        return bev

    def _vehicle_to_bev_coords(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Convert vehicle coordinates (meters) to BEV pixel coordinates.

        Vehicle coordinates:
        - Origin at vehicle center
        - X: forward (positive forward, negative backward)
        - Y: left (positive left, negative right)

        BEV coordinates:
        - Origin at top-left
        - X: right (increases to the right)
        - Y: down (increases downward)

        Args:
            x_m: X coordinate in meters (forward)
            y_m: Y coordinate in meters (left)

        Returns:
            (pixel_x, pixel_y) in BEV image
        """
        # Vehicle is at bottom-center of BEV
        vehicle_pixel_x = self.config.width_pixels // 2
        vehicle_pixel_y = int(self.config.height_pixels - self.config.rear_range_m / self.config.meters_per_pixel)

        # Convert meters to pixels and translate
        pixel_x = int(vehicle_pixel_x - y_m / self.config.meters_per_pixel)
        pixel_y = int(vehicle_pixel_y - x_m / self.config.meters_per_pixel)

        return (pixel_x, pixel_y)

    def _draw_grid(self, bev: np.ndarray) -> np.ndarray:
        """Draw grid lines on BEV."""
        # Draw horizontal lines (every 5 meters)
        for dist_m in range(0, int(self.config.forward_range_m) + 1, 5):
            pixel_x, pixel_y = self._vehicle_to_bev_coords(dist_m, 0)
            cv2.line(bev, (0, pixel_y), (self.config.width_pixels, pixel_y), self.config.grid_color, 1)

        # Draw vertical lines (every 2 meters)
        for lateral_m in range(int(-self.config.lateral_range_m), int(self.config.lateral_range_m) + 1, 2):
            pixel_x, pixel_y_top = self._vehicle_to_bev_coords(self.config.forward_range_m, lateral_m)
            _, pixel_y_bottom = self._vehicle_to_bev_coords(-self.config.rear_range_m, lateral_m)
            cv2.line(bev, (pixel_x, pixel_y_top), (pixel_x, pixel_y_bottom), self.config.grid_color, 1)

        return bev

    def _draw_distance_circles(self, bev: np.ndarray) -> np.ndarray:
        """Draw distance reference circles."""
        vehicle_pixel_x = self.config.width_pixels // 2
        vehicle_pixel_y = int(self.config.height_pixels - self.config.rear_range_m / self.config.meters_per_pixel)

        # Draw circles at 10m, 20m, 30m
        for distance_m in [10, 20, 30]:
            radius_pixels = int(distance_m / self.config.meters_per_pixel)
            cv2.circle(bev, (vehicle_pixel_x, vehicle_pixel_y), radius_pixels, self.config.grid_color, 1)

            # Add distance label
            label_x = vehicle_pixel_x + radius_pixels - 20
            label_y = vehicle_pixel_y
            cv2.putText(
                bev,
                f"{distance_m}m",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (100, 100, 100),
                1
            )

        return bev

    def _draw_camera_fov(self, bev: np.ndarray, camera_configs: Dict[int, CameraConfig]) -> np.ndarray:
        """Draw camera field of view cones."""
        for camera_id, config in camera_configs.items():
            # Get camera position in vehicle coordinates
            cam_x, cam_y, cam_z = config.position_offset
            cam_roll, cam_pitch, cam_yaw = config.orientation

            # Convert to BEV pixel coords
            cam_pixel = self._vehicle_to_bev_coords(cam_x, cam_y)

            # Estimate FOV cone (simplified)
            fov_angle = 60  # degrees
            fov_range = 15  # meters

            # Calculate FOV triangle points
            yaw_rad = np.radians(cam_yaw)
            left_angle = yaw_rad + np.radians(fov_angle / 2)
            right_angle = yaw_rad - np.radians(fov_angle / 2)

            # FOV endpoints
            left_x = cam_x + fov_range * np.cos(left_angle)
            left_y = cam_y + fov_range * np.sin(left_angle)
            right_x = cam_x + fov_range * np.cos(right_angle)
            right_y = cam_y + fov_range * np.sin(right_angle)

            left_pixel = self._vehicle_to_bev_coords(left_x, left_y)
            right_pixel = self._vehicle_to_bev_coords(right_x, right_y)

            # Draw FOV triangle
            pts = np.array([cam_pixel, left_pixel, right_pixel], dtype=np.int32)
            cv2.fillPoly(bev, [pts], self.config.fov_color)
            cv2.polylines(bev, [pts], True, (150, 150, 150), 1)

        return bev

    def _draw_ego_vehicle(self, bev: np.ndarray) -> np.ndarray:
        """Draw the ego vehicle (our vehicle) at center."""
        # Vehicle dimensions in pixels
        length_pixels = int(self.config.vehicle_length_m / self.config.meters_per_pixel)
        width_pixels = int(self.config.vehicle_width_m / self.config.meters_per_pixel)

        # Vehicle center position
        center_x = self.config.width_pixels // 2
        center_y = int(self.config.height_pixels - self.config.rear_range_m / self.config.meters_per_pixel)

        # Draw vehicle rectangle
        x1 = center_x - width_pixels // 2
        y1 = center_y - length_pixels // 2
        x2 = center_x + width_pixels // 2
        y2 = center_y + length_pixels // 2

        cv2.rectangle(bev, (x1, y1), (x2, y2), self.config.vehicle_color, -1)
        cv2.rectangle(bev, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Draw direction indicator (front of vehicle)
        arrow_start = (center_x, center_y)
        arrow_end = (center_x, y1 - 10)
        cv2.arrowedLine(bev, arrow_start, arrow_end, (0, 255, 255), 2)

        # Label
        cv2.putText(
            bev,
            "EGO",
            (center_x - 15, center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1
        )

        return bev

    def _draw_detections(self, bev: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """Draw detected objects on BEV."""
        for det in detections:
            if det.position_3d is None:
                continue

            x, y, z = det.position_3d

            # Check if within BEV range
            if not self._is_in_range(x, y):
                continue

            pixel_x, pixel_y = self._vehicle_to_bev_coords(x, y)

            # Draw object as circle
            cv2.circle(bev, (pixel_x, pixel_y), 5, self.config.object_color, -1)
            cv2.circle(bev, (pixel_x, pixel_y), 5, (255, 255, 255), 1)

            # Draw class label
            cv2.putText(
                bev,
                det.class_name[:3],
                (pixel_x - 10, pixel_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1
            )

        return bev

    def _draw_tracked_objects(self, bev: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """Draw tracked objects with trajectories on BEV."""
        for obj in tracked_objects:
            if obj.current_position is None:
                continue

            x, y, z = obj.current_position

            # Check if within BEV range
            if not self._is_in_range(x, y):
                continue

            pixel_x, pixel_y = self._vehicle_to_bev_coords(x, y)

            # Draw trajectory if available
            if self.config.show_trajectories and len(obj.trajectory) > 1:
                traj_points = []
                for timestamp, pos in obj.trajectory[-20:]:  # Last 20 points
                    if self._is_in_range(pos[0], pos[1]):
                        traj_pixel = self._vehicle_to_bev_coords(pos[0], pos[1])
                        traj_points.append(traj_pixel)

                if len(traj_points) > 1:
                    pts = np.array(traj_points, dtype=np.int32)
                    cv2.polylines(bev, [pts], False, self.config.trajectory_color, 1)

            # Draw object as circle
            cv2.circle(bev, (pixel_x, pixel_y), 6, self.config.object_color, -1)
            cv2.circle(bev, (pixel_x, pixel_y), 6, (255, 255, 255), 1)

            # Draw velocity vector if available
            if obj.current_velocity:
                vx, vy = obj.current_velocity
                # Scale velocity for visualization (1 m/s = 20 pixels)
                vel_pixel_x = pixel_x - int(vy / self.config.meters_per_pixel * 2)
                vel_pixel_y = pixel_y - int(vx / self.config.meters_per_pixel * 2)
                cv2.arrowedLine(bev, (pixel_x, pixel_y), (vel_pixel_x, vel_pixel_y), (0, 255, 255), 1)

            # Draw tracking ID and info
            label = f"ID:{obj.track_id}"
            if obj.current_velocity:
                vx, vy = obj.current_velocity
                speed = np.sqrt(vx**2 + vy**2)
                label += f" {speed:.1f}m/s"

            cv2.putText(
                bev,
                label,
                (pixel_x + 10, pixel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1
            )

        return bev

    def _is_in_range(self, x_m: float, y_m: float) -> bool:
        """Check if position is within BEV range."""
        return (
            -self.config.rear_range_m <= x_m <= self.config.forward_range_m and
            -self.config.lateral_range_m <= y_m <= self.config.lateral_range_m
        )

    def _add_labels(self, bev: np.ndarray) -> np.ndarray:
        """Add title and labels to BEV."""
        # Title
        cv2.putText(
            bev,
            "Bird's Eye View",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Scale indicator
        scale_text = f"Scale: {self.config.meters_per_pixel*100:.0f}cm/px"
        cv2.putText(
            bev,
            scale_text,
            (10, self.config.height_pixels - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (200, 200, 200),
            1
        )

        # North indicator
        cv2.putText(
            bev,
            "N",
            (self.config.width_pixels - 20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        cv2.arrowedLine(
            bev,
            (self.config.width_pixels - 15, 30),
            (self.config.width_pixels - 15, 15),
            (255, 255, 255),
            1
        )

        return bev
