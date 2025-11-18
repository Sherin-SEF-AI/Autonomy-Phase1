"""
Traffic light detection and state recognition.

Detects traffic lights and classifies their state (red, yellow, green).
Critical for safety and autonomous driving decision making.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger


logger = get_logger()


class TrafficLightState(Enum):
    """Traffic light states."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"
    OFF = "off"


@dataclass
class TrafficLight:
    """Detected traffic light."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    state: TrafficLightState
    confidence: float  # 0-1
    color_confidence: Dict[str, float]  # Confidence per color
    position: Tuple[int, int]  # Center point


class TrafficLightDetector:
    """
    Detects and classifies traffic lights using color-based and shape-based methods.

    Uses HSV color segmentation and contour analysis for detection.
    For production, should be replaced with deep learning (YOLO, Faster R-CNN).
    """

    def __init__(
        self,
        min_area: int = 200,
        max_area: int = 10000,
        aspect_ratio_range: Tuple[float, float] = (0.3, 3.0),
        enable_detection: bool = True
    ):
        """
        Initialize traffic light detector.

        Args:
            min_area: Minimum blob area (pixels²)
            max_area: Maximum blob area (pixels²)
            aspect_ratio_range: Valid aspect ratio range (height/width)
            enable_detection: Enable detection
        """
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.enable_detection = enable_detection

        # HSV color ranges for traffic lights
        # Red has two ranges (wraps around hue 180)
        self.color_ranges = {
            'red1': {
                'lower': np.array([0, 100, 100]),
                'upper': np.array([10, 255, 255])
            },
            'red2': {
                'lower': np.array([160, 100, 100]),
                'upper': np.array([180, 255, 255])
            },
            'yellow': {
                'lower': np.array([15, 100, 100]),
                'upper': np.array([35, 255, 255])
            },
            'green': {
                'lower': np.array([40, 50, 50]),
                'upper': np.array([90, 255, 255])
            }
        }

        # Statistics
        self.total_detections = 0
        self.detections_by_state = {state: 0 for state in TrafficLightState}

        logger.info("Traffic light detector initialized")

    def detect(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[TrafficLight]:
        """
        Detect traffic lights in image.

        Args:
            image: BGR image
            roi: Optional region of interest (x1, y1, x2, y2) to limit search

        Returns:
            List of detected traffic lights
        """
        if not self.enable_detection:
            return []

        # Apply ROI if specified (typically upper half of image)
        if roi:
            x1, y1, x2, y2 = roi
            search_region = image[y1:y2, x1:x2].copy()
            roi_offset = (x1, y1)
        else:
            # Default: search upper half of image
            h = image.shape[0]
            search_region = image[0:h//2, :].copy()
            roi_offset = (0, 0)

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)

        # Detect colored regions for each traffic light color
        color_masks = self._create_color_masks(hsv)

        # Find candidate regions
        candidates = self._find_candidates(color_masks, search_region)

        # Classify each candidate
        traffic_lights = []
        for bbox, color_scores in candidates:
            # Adjust bbox coordinates by ROI offset
            adjusted_bbox = (
                bbox[0] + roi_offset[0],
                bbox[1] + roi_offset[1],
                bbox[2] + roi_offset[0],
                bbox[3] + roi_offset[1]
            )

            # Determine state and confidence
            state, confidence = self._classify_state(color_scores)

            # Calculate center
            center = (
                (adjusted_bbox[0] + adjusted_bbox[2]) // 2,
                (adjusted_bbox[1] + adjusted_bbox[3]) // 2
            )

            traffic_light = TrafficLight(
                bbox=adjusted_bbox,
                state=state,
                confidence=confidence,
                color_confidence=color_scores,
                position=center
            )

            traffic_lights.append(traffic_light)

            # Update statistics
            self.total_detections += 1
            self.detections_by_state[state] += 1

        return traffic_lights

    def _create_color_masks(self, hsv: np.ndarray) -> Dict[str, np.ndarray]:
        """Create binary masks for each traffic light color."""
        masks = {}

        # Red (two ranges)
        red_mask1 = cv2.inRange(
            hsv,
            self.color_ranges['red1']['lower'],
            self.color_ranges['red1']['upper']
        )
        red_mask2 = cv2.inRange(
            hsv,
            self.color_ranges['red2']['lower'],
            self.color_ranges['red2']['upper']
        )
        masks['red'] = cv2.bitwise_or(red_mask1, red_mask2)

        # Yellow
        masks['yellow'] = cv2.inRange(
            hsv,
            self.color_ranges['yellow']['lower'],
            self.color_ranges['yellow']['upper']
        )

        # Green
        masks['green'] = cv2.inRange(
            hsv,
            self.color_ranges['green']['lower'],
            self.color_ranges['green']['upper']
        )

        # Apply morphological operations to clean up masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for color in masks:
            masks[color] = cv2.morphologyEx(masks[color], cv2.MORPH_OPEN, kernel)
            masks[color] = cv2.morphologyEx(masks[color], cv2.MORPH_CLOSE, kernel)

        return masks

    def _find_candidates(
        self,
        color_masks: Dict[str, np.ndarray],
        image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], Dict[str, float]]]:
        """
        Find candidate traffic light regions.

        Returns:
            List of (bbox, color_scores) tuples
        """
        candidates = []

        # Combine all color masks to find regions with any color
        combined_mask = np.zeros_like(color_masks['red'])
        for mask in color_masks.values():
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Find contours in combined mask
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio
            aspect_ratio = h / w if w > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue

            # Calculate color scores in this region
            roi_masks = {
                color: mask[y:y+h, x:x+w]
                for color, mask in color_masks.items()
            }

            color_scores = {}
            total_pixels = w * h
            for color, roi_mask in roi_masks.items():
                color_pixels = np.count_nonzero(roi_mask)
                color_scores[color] = color_pixels / total_pixels if total_pixels > 0 else 0.0

            # Only add if at least one color has significant presence
            if max(color_scores.values()) > 0.1:  # At least 10% of pixels
                bbox = (x, y, x + w, y + h)
                candidates.append((bbox, color_scores))

        return candidates

    def _classify_state(
        self,
        color_scores: Dict[str, float]
    ) -> Tuple[TrafficLightState, float]:
        """
        Classify traffic light state from color scores.

        Args:
            color_scores: Dictionary of color scores

        Returns:
            (state, confidence) tuple
        """
        # Get maximum score
        max_color = max(color_scores.keys(), key=lambda k: color_scores[k])
        max_score = color_scores[max_color]

        # Map to state
        if max_score < 0.1:
            return TrafficLightState.UNKNOWN, 0.0

        state_map = {
            'red': TrafficLightState.RED,
            'yellow': TrafficLightState.YELLOW,
            'green': TrafficLightState.GREEN
        }

        state = state_map.get(max_color, TrafficLightState.UNKNOWN)
        confidence = min(max_score * 2.0, 1.0)  # Scale confidence

        return state, confidence

    def is_red_light(self, traffic_lights: List[TrafficLight]) -> bool:
        """
        Check if any traffic light is red.

        Args:
            traffic_lights: List of detected traffic lights

        Returns:
            True if any light is red
        """
        return any(
            light.state == TrafficLightState.RED and light.confidence > 0.5
            for light in traffic_lights
        )

    def is_green_light(self, traffic_lights: List[TrafficLight]) -> bool:
        """
        Check if any traffic light is green.

        Args:
            traffic_lights: List of detected traffic lights

        Returns:
            True if any light is green
        """
        return any(
            light.state == TrafficLightState.GREEN and light.confidence > 0.5
            for light in traffic_lights
        )

    def get_dominant_state(
        self,
        traffic_lights: List[TrafficLight]
    ) -> Optional[TrafficLightState]:
        """
        Get the dominant traffic light state.

        Args:
            traffic_lights: List of detected traffic lights

        Returns:
            Most confident state, or None if no lights detected
        """
        if not traffic_lights:
            return None

        # Get light with highest confidence
        best_light = max(traffic_lights, key=lambda l: l.confidence)
        return best_light.state if best_light.confidence > 0.3 else None

    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        return {
            "total_detections": self.total_detections,
            "detections_by_state": {
                state.value: count
                for state, count in self.detections_by_state.items()
            },
            "enabled": self.enable_detection
        }

    def reset_statistics(self):
        """Reset statistics."""
        self.total_detections = 0
        self.detections_by_state = {state: 0 for state in TrafficLightState}
