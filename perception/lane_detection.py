"""
Lane detection module for autonomous vehicle perception.

Implements classical computer vision techniques for detecting lane markings:
- Canny edge detection
- Region of interest (ROI) masking
- Hough line transform
- Lane line fitting with polynomial regression
- Lane tracking with temporal smoothing
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from collections import deque

from utils.data_structures import LaneDetectionResult
from utils.logger import get_logger


logger = get_logger()


@dataclass
class LaneDetectionConfig:
    """Configuration for lane detection algorithm."""
    # Image preprocessing
    roi_vertices: List[Tuple[float, float]] = None  # ROI as percentage of image dims
    gaussian_kernel: int = 5

    # Canny edge detection
    canny_low: int = 50
    canny_high: int = 150

    # Hough transform
    hough_rho: int = 2
    hough_theta: float = np.pi / 180
    hough_threshold: int = 50
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10

    # Lane line filtering
    min_slope: float = 0.3
    max_slope: float = 3.0
    slope_tolerance: float = 0.2

    # Polynomial fitting
    poly_degree: int = 2

    # Temporal smoothing
    smoothing_window: int = 5

    # Lane width estimation (meters)
    lane_width_m: float = 3.7  # Standard US lane width

    # Departure detection
    lateral_offset_threshold: float = 0.3  # meters

    def __post_init__(self):
        if self.roi_vertices is None:
            # Default ROI (trapezoid)
            self.roi_vertices = [
                (0.1, 0.95),   # Bottom left
                (0.9, 0.95),   # Bottom right
                (0.6, 0.6),    # Top right
                (0.4, 0.6)     # Top left
            ]


class LaneDetector:
    """
    Lane detection using classical computer vision techniques.

    Detects lane lines from camera images and tracks them over time.
    """

    def __init__(self, config: Optional[LaneDetectionConfig] = None):
        """
        Initialize lane detector.

        Args:
            config: Lane detection configuration
        """
        self.config = config or LaneDetectionConfig()

        # Temporal smoothing buffers
        self.left_poly_history = deque(maxlen=self.config.smoothing_window)
        self.right_poly_history = deque(maxlen=self.config.smoothing_window)

        # Previous lane parameters
        self.prev_left_poly: Optional[np.ndarray] = None
        self.prev_right_poly: Optional[np.ndarray] = None

        # Statistics
        self.frames_processed = 0
        self.frames_with_detection = 0

        logger.info("Lane detector initialized")

    def detect(self, image: np.ndarray, camera_id: int, timestamp: float) -> LaneDetectionResult:
        """
        Detect lane lines in an image.

        Args:
            image: Input image (BGR format)
            camera_id: Camera identifier
            timestamp: Frame timestamp

        Returns:
            LaneDetectionResult object
        """
        self.frames_processed += 1

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.config.gaussian_kernel, self.config.gaussian_kernel), 0)

        # Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.config.canny_low,
            self.config.canny_high
        )

        # Apply region of interest mask
        masked_edges = self._apply_roi_mask(edges, image.shape)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            masked_edges,
            self.config.hough_rho,
            self.config.hough_theta,
            self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap
        )

        # Separate left and right lane lines
        left_lines, right_lines = self._separate_lane_lines(lines, image.shape)

        # Fit polynomials to lane lines
        left_poly, right_poly = self._fit_lane_polynomials(
            left_lines, right_lines, image.shape
        )

        # Apply temporal smoothing
        left_poly = self._smooth_polynomial(left_poly, self.left_poly_history)
        right_poly = self._smooth_polynomial(right_poly, self.right_poly_history)

        # Generate lane points from polynomials
        left_lane, right_lane, center_line = self._generate_lane_points(
            left_poly, right_poly, image.shape
        )

        # Calculate lane parameters
        lane_width = self._estimate_lane_width(left_poly, right_poly, image.shape)
        curvature = self._calculate_curvature(left_poly, right_poly, image.shape)
        lateral_offset = self._calculate_lateral_offset(left_poly, right_poly, image.shape)

        # Detect lane departure
        departure_warning = abs(lateral_offset) > self.config.lateral_offset_threshold if lateral_offset else False
        departure_direction = None
        if departure_warning and lateral_offset is not None:
            departure_direction = "right" if lateral_offset > 0 else "left"

        # Calculate confidence
        confidence = self._calculate_confidence(left_lines, right_lines)

        # Update statistics
        if left_lane or right_lane:
            self.frames_with_detection += 1

        # Store for next frame
        self.prev_left_poly = left_poly
        self.prev_right_poly = right_poly

        return LaneDetectionResult(
            timestamp=timestamp,
            camera_id=camera_id,
            left_lane=left_lane,
            right_lane=right_lane,
            center_line=center_line,
            lane_width=lane_width,
            curvature=curvature,
            lateral_offset=lateral_offset,
            left_poly=left_poly,
            right_poly=right_poly,
            detection_confidence=confidence,
            departure_warning=departure_warning,
            departure_direction=departure_direction
        )

    def _apply_roi_mask(self, image: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        """Apply region of interest mask to focus on road area."""
        height, width = shape[:2]
        mask = np.zeros_like(image)

        # Convert ROI vertices from percentages to pixels
        vertices = np.array([[
            (int(x * width), int(y * height))
            for x, y in self.config.roi_vertices
        ]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(image, mask)

        return masked_image

    def _separate_lane_lines(
        self,
        lines: Optional[np.ndarray],
        shape: Tuple[int, int, int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Separate detected lines into left and right lanes based on slope."""
        if lines is None:
            return [], []

        height, width = shape[:2]
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope magnitude
            if abs(slope) < self.config.min_slope or abs(slope) > self.config.max_slope:
                continue

            # Separate by slope sign (negative = left, positive = right in image coords)
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

        return left_lines, right_lines

    def _fit_lane_polynomials(
        self,
        left_lines: List[np.ndarray],
        right_lines: List[np.ndarray],
        shape: Tuple[int, int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit polynomial curves to lane line points."""
        height = shape[0]

        # Fit left lane
        left_poly = None
        if left_lines:
            left_points = np.concatenate(left_lines).reshape(-1, 2, 2)
            left_x = left_points[:, :, 0].flatten()
            left_y = left_points[:, :, 1].flatten()

            if len(left_x) > 2:
                try:
                    left_poly = np.polyfit(left_y, left_x, self.config.poly_degree)
                except np.linalg.LinAlgError:
                    left_poly = self.prev_left_poly

        # Fit right lane
        right_poly = None
        if right_lines:
            right_points = np.concatenate(right_lines).reshape(-1, 2, 2)
            right_x = right_points[:, :, 0].flatten()
            right_y = right_points[:, :, 1].flatten()

            if len(right_x) > 2:
                try:
                    right_poly = np.polyfit(right_y, right_x, self.config.poly_degree)
                except np.linalg.LinAlgError:
                    right_poly = self.prev_right_poly

        return left_poly, right_poly

    def _smooth_polynomial(
        self,
        poly: Optional[np.ndarray],
        history: deque
    ) -> Optional[np.ndarray]:
        """Apply temporal smoothing to polynomial coefficients."""
        if poly is not None:
            history.append(poly)

        if len(history) == 0:
            return poly

        # Average over history
        smoothed = np.mean(history, axis=0)
        return smoothed

    def _generate_lane_points(
        self,
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
        shape: Tuple[int, int, int]
    ) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
        """Generate lane line points from polynomial coefficients."""
        height, width = shape[:2]

        # Y coordinates to sample
        y_points = np.linspace(int(height * 0.6), height - 1, 50)

        # Left lane
        left_lane = None
        if left_poly is not None:
            left_x = np.polyval(left_poly, y_points)
            left_lane = [(int(x), int(y)) for x, y in zip(left_x, y_points) if 0 <= x < width]

        # Right lane
        right_lane = None
        if right_poly is not None:
            right_x = np.polyval(right_poly, y_points)
            right_lane = [(int(x), int(y)) for x, y in zip(right_x, y_points) if 0 <= x < width]

        # Center line
        center_line = None
        if left_lane and right_lane:
            center_x = (np.polyval(left_poly, y_points) + np.polyval(right_poly, y_points)) / 2
            center_line = [(int(x), int(y)) for x, y in zip(center_x, y_points) if 0 <= x < width]

        return left_lane, right_lane, center_line

    def _estimate_lane_width(
        self,
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
        shape: Tuple[int, int, int]
    ) -> Optional[float]:
        """Estimate lane width in meters."""
        if left_poly is None or right_poly is None:
            return None

        height = shape[0]
        y_eval = height - 1  # Bottom of image

        left_x = np.polyval(left_poly, y_eval)
        right_x = np.polyval(right_poly, y_eval)

        # Assume standard lane width at bottom of image
        pixel_width = abs(right_x - left_x)

        return self.config.lane_width_m

    def _calculate_curvature(
        self,
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
        shape: Tuple[int, int, int]
    ) -> Optional[float]:
        """Calculate lane curvature in 1/meters."""
        if left_poly is None and right_poly is None:
            return None

        height = shape[0]
        y_eval = height - 1

        # Use average of left and right curvature
        curvatures = []

        # Meters per pixel (approximate)
        ym_per_pix = 30 / 720  # 30 meters in 720 pixels
        xm_per_pix = 3.7 / 700  # 3.7 meters (lane width) in 700 pixels

        if left_poly is not None:
            # Convert to world space
            left_fit_cr = np.polyfit(
                np.linspace(0, height, 50) * ym_per_pix,
                np.polyval(left_poly, np.linspace(0, height, 50)) * xm_per_pix,
                2
            )
            left_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / abs(2 * left_fit_cr[0])
            curvatures.append(1 / left_curvature if left_curvature > 0 else 0)

        if right_poly is not None:
            right_fit_cr = np.polyfit(
                np.linspace(0, height, 50) * ym_per_pix,
                np.polyval(right_poly, np.linspace(0, height, 50)) * xm_per_pix,
                2
            )
            right_curvature = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / abs(2 * right_fit_cr[0])
            curvatures.append(1 / right_curvature if right_curvature > 0 else 0)

        return np.mean(curvatures) if curvatures else None

    def _calculate_lateral_offset(
        self,
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
        shape: Tuple[int, int, int]
    ) -> Optional[float]:
        """Calculate vehicle lateral offset from lane center (meters)."""
        if left_poly is None or right_poly is None:
            return None

        height, width = shape[:2]
        y_eval = height - 1

        # Lane center in pixels
        left_x = np.polyval(left_poly, y_eval)
        right_x = np.polyval(right_poly, y_eval)
        lane_center = (left_x + right_x) / 2

        # Vehicle center (assume camera is centered)
        vehicle_center = width / 2

        # Offset in pixels
        offset_pixels = vehicle_center - lane_center

        # Convert to meters (approximate)
        xm_per_pix = 3.7 / abs(right_x - left_x) if abs(right_x - left_x) > 0 else 3.7 / 700
        offset_meters = offset_pixels * xm_per_pix

        return offset_meters

    def _calculate_confidence(
        self,
        left_lines: List[np.ndarray],
        right_lines: List[np.ndarray]
    ) -> float:
        """Calculate detection confidence based on number of detected lines."""
        total_lines = len(left_lines) + len(right_lines)

        if total_lines == 0:
            return 0.0
        elif total_lines < 5:
            return 0.3
        elif total_lines < 10:
            return 0.6
        else:
            return min(1.0, 0.6 + (total_lines - 10) * 0.04)

    def get_statistics(self) -> dict:
        """Get lane detection statistics."""
        detection_rate = (
            self.frames_with_detection / self.frames_processed
            if self.frames_processed > 0 else 0.0
        )

        return {
            "frames_processed": self.frames_processed,
            "frames_with_detection": self.frames_with_detection,
            "detection_rate": detection_rate
        }

    def reset(self):
        """Reset temporal buffers and statistics."""
        self.left_poly_history.clear()
        self.right_poly_history.clear()
        self.prev_left_poly = None
        self.prev_right_poly = None
        self.frames_processed = 0
        self.frames_with_detection = 0
        logger.info("Lane detector reset")
