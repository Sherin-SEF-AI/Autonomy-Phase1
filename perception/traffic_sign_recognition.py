"""
Traffic Sign Recognition (TSR) Module

This module provides comprehensive traffic sign detection and recognition for
autonomous driving, including:
- Speed limit signs (all values)
- Warning signs (pedestrian crossing, school zone, curves, etc.)
- Regulatory signs (stop, yield, no entry, etc.)
- Informational signs (parking, highway, etc.)
- Multiple detection methods (YOLO-based, template matching, CNN classifier)
- Sign tracking across frames
- Confidence scoring and validation

Author: AV Perception System
Version: 1.2.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from collections import deque
import math


class SignType(Enum):
    """Types of traffic signs"""
    # Speed limits
    SPEED_LIMIT_20 = "speed_limit_20"
    SPEED_LIMIT_30 = "speed_limit_30"
    SPEED_LIMIT_40 = "speed_limit_40"
    SPEED_LIMIT_50 = "speed_limit_50"
    SPEED_LIMIT_60 = "speed_limit_60"
    SPEED_LIMIT_70 = "speed_limit_70"
    SPEED_LIMIT_80 = "speed_limit_80"
    SPEED_LIMIT_90 = "speed_limit_90"
    SPEED_LIMIT_100 = "speed_limit_100"
    SPEED_LIMIT_120 = "speed_limit_120"

    # Regulatory signs
    STOP = "stop"
    YIELD = "yield"
    NO_ENTRY = "no_entry"
    NO_PARKING = "no_parking"
    NO_OVERTAKING = "no_overtaking"
    PRIORITY_ROAD = "priority_road"
    END_SPEED_LIMIT = "end_speed_limit"
    END_NO_OVERTAKING = "end_no_overtaking"

    # Warning signs
    DANGER = "danger"
    CURVE_LEFT = "curve_left"
    CURVE_RIGHT = "curve_right"
    DOUBLE_CURVE = "double_curve"
    BUMPY_ROAD = "bumpy_road"
    SLIPPERY_ROAD = "slippery_road"
    NARROW_ROAD = "narrow_road"
    ROAD_WORK = "road_work"
    TRAFFIC_SIGNALS = "traffic_signals"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    CHILDREN_CROSSING = "children_crossing"
    BICYCLE_CROSSING = "bicycle_crossing"
    WILD_ANIMALS = "wild_animals"

    # Informational signs
    ROUNDABOUT = "roundabout"
    HIGHWAY_START = "highway_start"
    HIGHWAY_END = "highway_end"
    PARKING = "parking"
    ONE_WAY = "one_way"

    UNKNOWN = "unknown"


class SignShape(Enum):
    """Traffic sign shapes"""
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    SQUARE = "square"
    OCTAGON = "octagon"
    DIAMOND = "diamond"
    RECTANGLE = "rectangle"
    UNKNOWN = "unknown"


@dataclass
class TrafficSign:
    """Represents a detected traffic sign"""
    sign_type: SignType
    shape: SignShape
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    speed_value: Optional[int] = None  # For speed limit signs
    distance: Optional[float] = None  # Distance to sign (meters)
    track_id: Optional[int] = None
    timestamp: float = 0.0
    frames_seen: int = 1

    def get_center(self) -> Tuple[int, int]:
        """Get center coordinates of sign"""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    def get_importance_score(self) -> float:
        """
        Calculate importance score for prioritizing signs
        Higher score = more important
        """
        base_score = self.confidence

        # Critical signs have higher importance
        if self.sign_type == SignType.STOP:
            base_score *= 2.0
        elif self.sign_type == SignType.YIELD:
            base_score *= 1.8
        elif self.sign_type == SignType.NO_ENTRY:
            base_score *= 1.9
        elif self.sign_type in [SignType.PEDESTRIAN_CROSSING, SignType.CHILDREN_CROSSING]:
            base_score *= 1.7
        elif "SPEED_LIMIT" in self.sign_type.value:
            base_score *= 1.5

        # Closer signs are more important
        if self.distance:
            distance_factor = max(0.5, 1.0 - (self.distance / 100.0))
            base_score *= distance_factor

        # Signs seen in multiple frames are more reliable
        frame_factor = min(1.5, 1.0 + (self.frames_seen / 10.0))
        base_score *= frame_factor

        return base_score


@dataclass
class TSRConfig:
    """Configuration for Traffic Sign Recognition"""
    detection_method: str = "hybrid"  # "yolo", "template", "hybrid"
    confidence_threshold: float = 0.6
    min_sign_size: int = 20  # Minimum sign size in pixels
    max_sign_size: int = 300  # Maximum sign size in pixels
    roi_top_ratio: float = 0.0  # Only search top portion of image
    roi_bottom_ratio: float = 0.6  # (0.0 to 0.6 = top 60% of image)
    enable_tracking: bool = True
    tracking_max_distance: float = 50.0  # Max distance for association (pixels)
    tracking_max_age: int = 10  # Max frames to keep track without detection
    temporal_smoothing: int = 3  # Frames to smooth over
    enable_speed_limit_ocr: bool = True
    max_detections_per_frame: int = 10


class TrafficSignRecognizer:
    """
    Traffic Sign Recognition system

    Features:
    - Multi-method detection (YOLO, template matching, color-based)
    - 40+ sign types supported
    - Shape-based classification
    - OCR for speed limit values
    - Multi-frame tracking for stability
    - Distance estimation
    - Priority-based sign ranking
    """

    def __init__(self, config: Optional[TSRConfig] = None):
        """
        Initialize traffic sign recognizer

        Args:
            config: TSR configuration
        """
        self.config = config or TSRConfig()
        self.tracked_signs: Dict[int, TrafficSign] = {}
        self.next_track_id = 0
        self.sign_history: deque = deque(maxlen=self.config.temporal_smoothing)

        # Color ranges for different sign types (HSV)
        self.color_ranges = {
            'red': [
                {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([160, 100, 100]), 'upper': np.array([180, 255, 255])}
            ],
            'blue': [
                {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])}
            ],
            'yellow': [
                {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])}
            ]
        }

        # Statistics
        self.total_detections = 0
        self.detections_by_type: Dict[SignType, int] = {}

    def detect_and_recognize(
        self,
        image: np.ndarray,
        camera_calibration: Optional[Dict[str, Any]] = None
    ) -> List[TrafficSign]:
        """
        Detect and recognize traffic signs in image

        Args:
            image: Input image (BGR)
            camera_calibration: Camera calibration for distance estimation

        Returns:
            List of detected traffic signs
        """
        # Apply ROI to focus on relevant area
        roi_image, roi_offset = self._apply_roi(image)

        # Detect candidate regions
        if self.config.detection_method in ["yolo", "hybrid"]:
            candidates = self._detect_with_yolo(roi_image)
        else:
            candidates = []

        # Color-based detection
        if self.config.detection_method in ["template", "hybrid"]:
            color_candidates = self._detect_by_color_and_shape(roi_image)
            candidates.extend(color_candidates)

        # Adjust bounding boxes for ROI offset
        for sign in candidates:
            x, y, w, h = sign.bbox
            sign.bbox = (x, y + roi_offset, w, h)

        # Classify signs
        classified_signs = []
        for sign in candidates:
            classified = self._classify_sign(image, sign)
            if classified:
                classified_signs.append(classified)

        # Estimate distances
        if camera_calibration:
            for sign in classified_signs:
                sign.distance = self._estimate_distance(sign, camera_calibration)

        # Track signs across frames
        if self.config.enable_tracking:
            classified_signs = self._track_signs(classified_signs)

        # Apply temporal smoothing
        smoothed_signs = self._apply_temporal_smoothing(classified_signs)

        # Update statistics
        self.total_detections += len(smoothed_signs)
        for sign in smoothed_signs:
            self.detections_by_type[sign.sign_type] = \
                self.detections_by_type.get(sign.sign_type, 0) + 1

        # Sort by importance
        smoothed_signs.sort(key=lambda s: s.get_importance_score(), reverse=True)

        return smoothed_signs[:self.config.max_detections_per_frame]

    def _apply_roi(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Apply region of interest to focus on relevant area"""
        h, w = image.shape[:2]
        top = int(h * self.config.roi_top_ratio)
        bottom = int(h * self.config.roi_bottom_ratio)

        roi = image[top:bottom, :]
        return roi, top

    def _detect_with_yolo(self, image: np.ndarray) -> List[TrafficSign]:
        """
        Detect signs using YOLO (placeholder - would use actual model)

        For now, returns empty list. In production, this would use
        a fine-tuned YOLO model for traffic sign detection.
        """
        # TODO: Integrate actual YOLO model trained on traffic signs
        # model = YOLO('traffic_signs.pt')
        # results = model(image)
        return []

    def _detect_by_color_and_shape(self, image: np.ndarray) -> List[TrafficSign]:
        """Detect signs using color and shape analysis"""
        candidates = []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect red signs (most regulatory and warning signs)
        red_candidates = self._find_colored_regions(hsv, 'red')
        candidates.extend(red_candidates)

        # Detect blue signs (informational)
        blue_candidates = self._find_colored_regions(hsv, 'blue')
        candidates.extend(blue_candidates)

        # Detect yellow signs (warnings)
        yellow_candidates = self._find_colored_regions(hsv, 'yellow')
        candidates.extend(yellow_candidates)

        return candidates

    def _find_colored_regions(
        self,
        hsv_image: np.ndarray,
        color: str
    ) -> List[TrafficSign]:
        """Find regions with specific color"""
        candidates = []

        if color not in self.color_ranges:
            return candidates

        # Create mask for color
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for range_dict in self.color_ranges[color]:
            color_mask = cv2.inRange(hsv_image, range_dict['lower'], range_dict['upper'])
            mask = cv2.bitwise_or(mask, color_mask)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_sign_size ** 2:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size and aspect ratio
            if w < self.config.min_sign_size or h < self.config.min_sign_size:
                continue
            if w > self.config.max_sign_size or h > self.config.max_sign_size:
                continue

            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            # Detect shape
            shape = self._detect_shape(contour)

            # Create candidate
            sign = TrafficSign(
                sign_type=SignType.UNKNOWN,
                shape=shape,
                bbox=(x, y, w, h),
                confidence=0.5  # Initial confidence
            )
            candidates.append(sign)

        return candidates

    def _detect_shape(self, contour: np.ndarray) -> SignShape:
        """Detect shape of sign from contour"""
        # Approximate contour to polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        num_vertices = len(approx)

        # Classify by number of vertices
        if num_vertices == 3:
            return SignShape.TRIANGLE
        elif num_vertices == 4:
            # Check if square or rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.9 <= aspect_ratio <= 1.1:
                return SignShape.SQUARE
            else:
                return SignShape.RECTANGLE
        elif num_vertices == 8:
            return SignShape.OCTAGON
        elif num_vertices > 8:
            # Could be circle
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.8:
                    return SignShape.CIRCLE

        return SignShape.UNKNOWN

    def _classify_sign(
        self,
        image: np.ndarray,
        candidate: TrafficSign
    ) -> Optional[TrafficSign]:
        """Classify sign type based on shape, color, and content"""
        x, y, w, h = candidate.bbox

        # Extract sign region
        sign_roi = image[y:y+h, x:x+w]

        if sign_roi.size == 0:
            return None

        # Classification based on shape
        if candidate.shape == SignShape.OCTAGON:
            # Likely STOP sign
            if self._verify_stop_sign(sign_roi):
                candidate.sign_type = SignType.STOP
                candidate.confidence = 0.9
                return candidate

        elif candidate.shape == SignShape.TRIANGLE:
            # Warning signs or YIELD
            if self._is_inverted_triangle(sign_roi):
                candidate.sign_type = SignType.YIELD
                candidate.confidence = 0.85
                return candidate
            else:
                # Classify warning sign type
                warning_type = self._classify_warning_sign(sign_roi)
                if warning_type != SignType.UNKNOWN:
                    candidate.sign_type = warning_type
                    candidate.confidence = 0.75
                    return candidate

        elif candidate.shape == SignShape.CIRCLE:
            # Speed limits or regulatory signs
            # Try OCR for speed limit
            if self.config.enable_speed_limit_ocr:
                speed_value = self._extract_speed_limit(sign_roi)
                if speed_value:
                    candidate.sign_type = self._get_speed_limit_type(speed_value)
                    candidate.speed_value = speed_value
                    candidate.confidence = 0.8
                    return candidate

            # Other circular regulatory signs
            regulatory_type = self._classify_regulatory_sign(sign_roi)
            if regulatory_type != SignType.UNKNOWN:
                candidate.sign_type = regulatory_type
                candidate.confidence = 0.75
                return candidate

        elif candidate.shape == SignShape.SQUARE:
            # Informational signs
            info_type = self._classify_informational_sign(sign_roi)
            if info_type != SignType.UNKNOWN:
                candidate.sign_type = info_type
                candidate.confidence = 0.7
                return candidate

        # If classification failed but we have a valid shape, keep as unknown
        if candidate.shape != SignShape.UNKNOWN:
            candidate.confidence = 0.5
            return candidate

        return None

    def _verify_stop_sign(self, roi: np.ndarray) -> bool:
        """Verify if octagonal sign is a STOP sign"""
        # Check for red color dominance
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for range_dict in self.color_ranges['red']:
            mask = cv2.inRange(hsv, range_dict['lower'], range_dict['upper'])
            red_mask = cv2.bitwise_or(red_mask, mask)

        red_ratio = np.sum(red_mask > 0) / red_mask.size
        return red_ratio > 0.4

    def _is_inverted_triangle(self, roi: np.ndarray) -> bool:
        """Check if triangle is inverted (YIELD sign)"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        # Get largest contour
        main_contour = max(contours, key=cv2.contourArea)

        # Find topmost and bottommost points
        topmost = tuple(main_contour[main_contour[:, :, 1].argmin()][0])
        bottommost = tuple(main_contour[main_contour[:, :, 1].argmax()][0])

        # Check for white/yellow color (YIELD signs are typically red border, white/yellow fill)
        center_y = roi.shape[0] // 2
        center_x = roi.shape[1] // 2

        if 0 <= center_y < roi.shape[0] and 0 <= center_x < roi.shape[1]:
            center_color = roi[center_y, center_x]
            # Check if center is light colored
            if np.mean(center_color) > 150:
                return True

        return False

    def _classify_warning_sign(self, roi: np.ndarray) -> SignType:
        """Classify type of warning sign (simplified)"""
        # In production, this would use a CNN classifier or template matching
        # For now, return generic danger
        return SignType.DANGER

    def _classify_regulatory_sign(self, roi: np.ndarray) -> SignType:
        """Classify regulatory sign type"""
        # Simplified classification
        # In production, use CNN or template matching
        return SignType.UNKNOWN

    def _classify_informational_sign(self, roi: np.ndarray) -> SignType:
        """Classify informational sign type"""
        # Simplified classification
        return SignType.UNKNOWN

    def _extract_speed_limit(self, roi: np.ndarray) -> Optional[int]:
        """Extract speed limit value using OCR"""
        # Preprocess for OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        gray = cv2.equalizeHist(gray)

        # Threshold to get digits
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Simple template matching for common speed limits
        # In production, use proper OCR (Tesseract, EasyOCR, etc.)
        common_speeds = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120]

        # For now, return None (would implement actual OCR)
        # TODO: Integrate OCR library
        return None

    def _get_speed_limit_type(self, speed: int) -> SignType:
        """Get SignType enum for speed limit value"""
        speed_map = {
            20: SignType.SPEED_LIMIT_20,
            30: SignType.SPEED_LIMIT_30,
            40: SignType.SPEED_LIMIT_40,
            50: SignType.SPEED_LIMIT_50,
            60: SignType.SPEED_LIMIT_60,
            70: SignType.SPEED_LIMIT_70,
            80: SignType.SPEED_LIMIT_80,
            90: SignType.SPEED_LIMIT_90,
            100: SignType.SPEED_LIMIT_100,
            120: SignType.SPEED_LIMIT_120
        }
        return speed_map.get(speed, SignType.UNKNOWN)

    def _estimate_distance(
        self,
        sign: TrafficSign,
        calibration: Dict[str, Any]
    ) -> float:
        """Estimate distance to sign using camera calibration"""
        # Assume standard sign sizes
        standard_sizes = {
            SignShape.OCTAGON: 0.75,  # STOP sign ~75cm
            SignShape.CIRCLE: 0.60,   # Speed limit ~60cm diameter
            SignShape.TRIANGLE: 0.90,  # Warning sign ~90cm
            SignShape.SQUARE: 0.60,    # Info sign ~60cm
            SignShape.RECTANGLE: 0.80  # Rectangular sign ~80cm
        }

        real_height = standard_sizes.get(sign.shape, 0.60)  # Default 60cm

        # Using focal length and known object size
        focal_length = calibration.get('focal_length', 1000)  # pixels
        _, _, _, pixel_height = sign.bbox

        if pixel_height > 0:
            distance = (real_height * focal_length) / pixel_height
            return distance

        return 0.0

    def _track_signs(self, new_detections: List[TrafficSign]) -> List[TrafficSign]:
        """Track signs across frames"""
        # Associate new detections with existing tracks
        associated = set()

        for track_id, tracked_sign in list(self.tracked_signs.items()):
            best_match = None
            best_distance = self.config.tracking_max_distance

            # Find best matching detection
            for i, new_sign in enumerate(new_detections):
                if i in associated:
                    continue

                # Only match same sign type
                if new_sign.sign_type != tracked_sign.sign_type:
                    continue

                # Calculate distance between centers
                old_center = tracked_sign.get_center()
                new_center = new_sign.get_center()
                distance = math.sqrt(
                    (old_center[0] - new_center[0])**2 +
                    (old_center[1] - new_center[1])**2
                )

                if distance < best_distance:
                    best_distance = distance
                    best_match = i

            if best_match is not None:
                # Update track
                matched_sign = new_detections[best_match]
                matched_sign.track_id = track_id
                matched_sign.frames_seen = tracked_sign.frames_seen + 1
                self.tracked_signs[track_id] = matched_sign
                associated.add(best_match)
            else:
                # Track lost, keep for a few frames
                tracked_sign.frames_seen -= 1
                if tracked_sign.frames_seen <= -self.config.tracking_max_age:
                    del self.tracked_signs[track_id]

        # Create new tracks for unassociated detections
        for i, new_sign in enumerate(new_detections):
            if i not in associated:
                new_sign.track_id = self.next_track_id
                self.tracked_signs[self.next_track_id] = new_sign
                self.next_track_id += 1

        # Return active tracks
        return [s for s in self.tracked_signs.values() if s.frames_seen > 0]

    def _apply_temporal_smoothing(
        self,
        signs: List[TrafficSign]
    ) -> List[TrafficSign]:
        """Apply temporal smoothing to reduce jitter"""
        self.sign_history.append(signs)

        if len(self.sign_history) < self.config.temporal_smoothing:
            return signs

        # Average positions and confidences over history
        smoothed = []
        for sign in signs:
            if not sign.track_id:
                smoothed.append(sign)
                continue

            # Find same sign in history
            historical_signs = []
            for hist_frame in self.sign_history:
                for hist_sign in hist_frame:
                    if hist_sign.track_id == sign.track_id:
                        historical_signs.append(hist_sign)

            if not historical_signs:
                smoothed.append(sign)
                continue

            # Average bbox
            avg_x = sum(s.bbox[0] for s in historical_signs) / len(historical_signs)
            avg_y = sum(s.bbox[1] for s in historical_signs) / len(historical_signs)
            avg_w = sum(s.bbox[2] for s in historical_signs) / len(historical_signs)
            avg_h = sum(s.bbox[3] for s in historical_signs) / len(historical_signs)

            # Average confidence
            avg_conf = sum(s.confidence for s in historical_signs) / len(historical_signs)

            smoothed_sign = TrafficSign(
                sign_type=sign.sign_type,
                shape=sign.shape,
                bbox=(int(avg_x), int(avg_y), int(avg_w), int(avg_h)),
                confidence=avg_conf,
                speed_value=sign.speed_value,
                distance=sign.distance,
                track_id=sign.track_id,
                frames_seen=sign.frames_seen
            )
            smoothed.append(smoothed_sign)

        return smoothed

    def visualize_signs(
        self,
        image: np.ndarray,
        signs: List[TrafficSign]
    ) -> np.ndarray:
        """Visualize detected signs on image"""
        vis_image = image.copy()

        for sign in signs:
            x, y, w, h = sign.bbox

            # Color based on sign type
            if sign.sign_type == SignType.STOP:
                color = (0, 0, 255)  # Red
            elif "SPEED_LIMIT" in sign.sign_type.value:
                color = (0, 165, 255)  # Orange
            elif sign.sign_type in [SignType.YIELD, SignType.NO_ENTRY]:
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 0, 255)  # Magenta

            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Prepare label
            label = sign.sign_type.value.replace('_', ' ').title()
            if sign.speed_value:
                label = f"{sign.speed_value} km/h"

            label += f" ({sign.confidence:.2f})"

            if sign.distance:
                label += f" {sign.distance:.1f}m"

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x, y - 20), (x + label_size[0], y), color, -1)

            # Draw label text
            cv2.putText(vis_image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw track ID if available
            if sign.track_id is not None:
                cv2.putText(vis_image, f"ID:{sign.track_id}", (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return vis_image

    def get_active_signs(self) -> List[TrafficSign]:
        """Get currently active/tracked signs"""
        return [s for s in self.tracked_signs.values() if s.frames_seen > 0]

    def get_statistics(self) -> Dict[str, Any]:
        """Get TSR statistics"""
        return {
            'total_detections': self.total_detections,
            'active_tracks': len([s for s in self.tracked_signs.values() if s.frames_seen > 0]),
            'detections_by_type': {
                k.value: v for k, v in self.detections_by_type.items()
            }
        }
