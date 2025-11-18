"""
Overlay renderer for drawing perception results on camera images.

Draws:
- Lane detection overlays
- Object detection bounding boxes
- Tracking IDs and trajectories
- Distance and velocity information
- Safety warnings
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict

from utils.data_structures import (
    LaneDetectionResult,
    DetectedObject,
    TrackedObject,
    SafetyWarning,
    WarningLevel,
    ObjectClass
)
from utils.logger import get_logger


logger = get_logger()


# Color scheme (BGR format)
class Colors:
    """Color definitions for visualization."""
    # Lane colors
    LANE_LEFT = (0, 255, 0)       # Green
    LANE_RIGHT = (0, 255, 0)      # Green
    LANE_CENTER = (255, 255, 0)   # Cyan
    LANE_FILL = (0, 255, 0)       # Green with alpha

    # Object detection colors by class
    PERSON = (0, 255, 255)         # Yellow
    BICYCLE = (255, 128, 0)        # Orange
    CAR = (255, 0, 0)              # Blue
    MOTORCYCLE = (255, 0, 128)     # Purple
    BUS = (128, 0, 255)            # Magenta
    TRUCK = (0, 128, 255)          # Light blue
    TRAFFIC_LIGHT = (0, 255, 0)    # Green
    STOP_SIGN = (0, 0, 255)        # Red
    DEFAULT = (255, 255, 255)      # White

    # Warning colors
    WARNING_ADVISORY = (0, 255, 255)   # Yellow
    WARNING_CAUTION = (0, 165, 255)    # Orange
    WARNING_CRITICAL = (0, 0, 255)     # Red

    # UI colors
    TEXT = (255, 255, 255)         # White
    TEXT_BG = (0, 0, 0)            # Black
    TRAJECTORY = (255, 128, 255)   # Pink


CLASS_COLORS = {
    ObjectClass.PERSON: Colors.PERSON,
    ObjectClass.BICYCLE: Colors.BICYCLE,
    ObjectClass.CAR: Colors.CAR,
    ObjectClass.MOTORCYCLE: Colors.MOTORCYCLE,
    ObjectClass.BUS: Colors.BUS,
    ObjectClass.TRUCK: Colors.TRUCK,
    ObjectClass.TRAFFIC_LIGHT: Colors.TRAFFIC_LIGHT,
    ObjectClass.STOP_SIGN: Colors.STOP_SIGN,
}


class OverlayRenderer:
    """
    Renders perception overlays on camera images.
    """

    def __init__(
        self,
        show_lanes: bool = True,
        show_detections: bool = True,
        show_tracking: bool = True,
        show_distance: bool = True,
        show_warnings: bool = True,
        show_info_panel: bool = True,
        show_trajectory_prediction: bool = True
    ):
        """
        Initialize overlay renderer.

        Args:
            show_lanes: Draw lane detection overlays
            show_detections: Draw detection bounding boxes
            show_tracking: Draw tracking IDs and trajectories
            show_distance: Show distance information
            show_warnings: Show safety warnings
            show_info_panel: Show information panel
            show_trajectory_prediction: Show predicted trajectories and collision warnings
        """
        self.show_lanes = show_lanes
        self.show_detections = show_detections
        self.show_tracking = show_tracking
        self.show_distance = show_distance
        self.show_warnings = show_warnings
        self.show_info_panel = show_info_panel
        self.show_trajectory_prediction = show_trajectory_prediction

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

    def render(
        self,
        image: np.ndarray,
        lane_result: Optional[LaneDetectionResult] = None,
        detections: Optional[List[DetectedObject]] = None,
        tracked_objects: Optional[List[TrackedObject]] = None,
        warnings: Optional[List[SafetyWarning]] = None
    ) -> np.ndarray:
        """
        Render all overlays on image.

        Args:
            image: Input image (BGR)
            lane_result: Lane detection result
            detections: List of detected objects
            tracked_objects: List of tracked objects
            warnings: List of active warnings

        Returns:
            Image with overlays
        """
        # Create overlay image
        overlay = image.copy()

        # Draw lane detection
        if self.show_lanes and lane_result is not None:
            overlay = self._draw_lanes(overlay, lane_result)

        # Draw detections
        if self.show_detections and detections:
            overlay = self._draw_detections(overlay, detections)

        # Draw tracked objects
        if self.show_tracking and tracked_objects:
            overlay = self._draw_tracked_objects(overlay, tracked_objects)

        # Draw warnings
        if self.show_warnings and warnings:
            overlay = self._draw_warnings(overlay, warnings)

        # Draw info panel
        if self.show_info_panel:
            overlay = self._draw_info_panel(
                overlay, lane_result, detections, tracked_objects, warnings
            )

        return overlay

    def _draw_lanes(
        self,
        image: np.ndarray,
        lane_result: LaneDetectionResult
    ) -> np.ndarray:
        """Draw lane detection overlays."""
        overlay = image.copy()

        # Draw lane fill (between left and right lanes)
        if lane_result.left_lane and lane_result.right_lane:
            # Create polygon from lane points
            left_points = np.array(lane_result.left_lane, dtype=np.int32)
            right_points = np.array(lane_result.right_lane[::-1], dtype=np.int32)
            lane_polygon = np.concatenate([left_points, right_points])

            # Draw filled polygon with transparency
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [lane_polygon], Colors.LANE_FILL)
            image = cv2.addWeighted(image, 0.9, mask, 0.1, 0)

        # Draw left lane line
        if lane_result.left_lane:
            points = np.array(lane_result.left_lane, dtype=np.int32)
            cv2.polylines(image, [points], False, Colors.LANE_LEFT, 3)

        # Draw right lane line
        if lane_result.right_lane:
            points = np.array(lane_result.right_lane, dtype=np.int32)
            cv2.polylines(image, [points], False, Colors.LANE_RIGHT, 3)

        # Draw center line
        if lane_result.center_line:
            points = np.array(lane_result.center_line, dtype=np.int32)
            cv2.polylines(image, [points], False, Colors.LANE_CENTER, 2, cv2.LINE_AA)

        # Draw lane departure warning
        if lane_result.departure_warning:
            warning_text = f"LANE DEPARTURE {lane_result.departure_direction.upper()}!"
            self._draw_text_with_background(
                image,
                warning_text,
                (image.shape[1] // 2 - 150, 50),
                Colors.WARNING_CRITICAL,
                font_scale=0.8,
                thickness=2
            )

        return image

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[DetectedObject]
    ) -> np.ndarray:
        """Draw detection bounding boxes."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Get color for class
            try:
                obj_class = ObjectClass(det.class_id)
                color = CLASS_COLORS.get(obj_class, Colors.DEFAULT)
            except ValueError:
                color = Colors.DEFAULT

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label_parts = [det.class_name]
            if det.confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if self.show_distance and det.distance:
                label_parts.append(f"{det.distance:.1f}m")

            label = " ".join(label_parts)

            # Draw label
            self._draw_text_with_background(
                image,
                label,
                (x1, y1 - 5),
                color
            )

        return image

    def _draw_tracked_objects(
        self,
        image: np.ndarray,
        tracked_objects: List[TrackedObject]
    ) -> np.ndarray:
        """Draw tracked objects with IDs and trajectories."""
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.current_bbox

            # Get color
            try:
                obj_class = ObjectClass(obj.class_id)
                color = CLASS_COLORS.get(obj_class, Colors.DEFAULT)
            except ValueError:
                color = Colors.DEFAULT

            # Draw tracking ID
            track_text = f"ID:{obj.track_id}"
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self._draw_text_with_background(
                image,
                track_text,
                (center_x - 20, y2 + 15),
                color,
                font_scale=0.4
            )

            # Draw velocity if available
            if obj.current_velocity and self.show_distance:
                vx, vy = obj.current_velocity
                speed = np.sqrt(vx**2 + vy**2)
                vel_text = f"{speed:.1f}m/s"
                self._draw_text_with_background(
                    image,
                    vel_text,
                    (center_x - 25, y2 + 30),
                    Colors.TEXT,
                    font_scale=0.4
                )

            # Draw trajectory prediction if enabled
            if self.show_trajectory_prediction:
                # Draw collision warning based on time_to_collision
                if obj.time_to_collision is not None:
                    if obj.time_to_collision <= 1.5:
                        warning_color = Colors.WARNING_CRITICAL
                        warning_text = f"⚠ TTC: {obj.time_to_collision:.1f}s"
                        # Draw a red circle around the object
                        cv2.circle(image, (center_x, center_y),
                                 max((x2-x1), (y2-y1))//2 + 10,
                                 warning_color, 3)
                    elif obj.time_to_collision <= 2.5:
                        warning_color = Colors.WARNING_CAUTION
                        warning_text = f"TTC: {obj.time_to_collision:.1f}s"
                    else:
                        warning_color = Colors.WARNING_ADVISORY
                        warning_text = f"TTC: {obj.time_to_collision:.1f}s"

                    # Draw time to collision warning
                    self._draw_text_with_background(
                        image,
                        warning_text,
                        (center_x - 30, y1 - 10),
                        warning_color,
                        font_scale=0.5,
                        thickness=2
                    )

                # Draw predicted position arrow
                if obj.predicted_position and obj.current_position:
                    # Simple visualization: draw arrow from current to predicted position
                    # Note: This is a simplified 2D projection
                    # In reality would need proper camera projection
                    curr_x, curr_y, _ = obj.current_position
                    pred_x, pred_y, _ = obj.predicted_position

                    # Convert 3D deltas to approximate 2D screen deltas
                    # This is a rough approximation - proper camera projection would be better
                    dx = (pred_x - curr_x) * 20  # Scale factor for visualization
                    dy = (pred_y - curr_y) * 20

                    # Draw arrow from center to predicted position
                    end_x = int(center_x + dx)
                    end_y = int(center_y - dy)  # Negative because y-axis is flipped in image

                    cv2.arrowedLine(
                        image,
                        (center_x, center_y),
                        (end_x, end_y),
                        Colors.TRAJECTORY,
                        2,
                        tipLength=0.3
                    )

        return image

    def _draw_warnings(
        self,
        image: np.ndarray,
        warnings: List[SafetyWarning]
    ) -> np.ndarray:
        """Draw safety warnings."""
        y_offset = 80

        for warning in warnings:
            # Get warning color based on level
            if warning.level == WarningLevel.CRITICAL:
                color = Colors.WARNING_CRITICAL
            elif warning.level == WarningLevel.CAUTION:
                color = Colors.WARNING_CAUTION
            else:
                color = Colors.WARNING_ADVISORY

            # Draw warning text
            text = f"[{warning.level.name}] {warning.message}"
            self._draw_text_with_background(
                image,
                text,
                (10, y_offset),
                color,
                font_scale=0.6,
                thickness=2
            )

            y_offset += 25

        return image

    def _draw_info_panel(
        self,
        image: np.ndarray,
        lane_result: Optional[LaneDetectionResult],
        detections: Optional[List[DetectedObject]],
        tracked_objects: Optional[List[TrackedObject]],
        warnings: Optional[List[SafetyWarning]]
    ) -> np.ndarray:
        """Draw information panel with statistics."""
        height, width = image.shape[:2]
        panel_width = 250
        panel_height = 150
        x = width - panel_width - 10
        y = 10

        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        # Draw border
        cv2.rectangle(image, (x, y), (x + panel_width, y + panel_height), Colors.TEXT, 1)

        # Draw title
        cv2.putText(
            image,
            "PERCEPTION INFO",
            (x + 10, y + 20),
            self.font,
            0.5,
            Colors.TEXT,
            1
        )

        # Draw statistics
        y_text = y + 45
        line_height = 20

        # Lane detection info
        if lane_result:
            lane_status = "DETECTED" if (lane_result.left_lane or lane_result.right_lane) else "NONE"
            lane_color = Colors.LANE_LEFT if lane_status == "DETECTED" else Colors.WARNING_CAUTION
            cv2.putText(image, f"Lanes: {lane_status}", (x + 10, y_text), self.font, 0.4, lane_color, 1)
            y_text += line_height

            if lane_result.lateral_offset is not None:
                cv2.putText(
                    image,
                    f"Offset: {lane_result.lateral_offset:.2f}m",
                    (x + 10, y_text),
                    self.font,
                    0.4,
                    Colors.TEXT,
                    1
                )
                y_text += line_height

        # Detection count
        det_count = len(detections) if detections else 0
        cv2.putText(image, f"Detections: {det_count}", (x + 10, y_text), self.font, 0.4, Colors.TEXT, 1)
        y_text += line_height

        # Tracking count
        track_count = len(tracked_objects) if tracked_objects else 0
        cv2.putText(image, f"Tracks: {track_count}", (x + 10, y_text), self.font, 0.4, Colors.TEXT, 1)
        y_text += line_height

        # Warnings count
        warning_count = len(warnings) if warnings else 0
        warning_color = Colors.WARNING_CRITICAL if warning_count > 0 else Colors.TEXT
        cv2.putText(image, f"Warnings: {warning_count}", (x + 10, y_text), self.font, 0.4, warning_color, 1)

        return image

    def _draw_text_with_background(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        font_scale: Optional[float] = None,
        thickness: Optional[int] = None
    ):
        """Draw text with a background rectangle."""
        font_scale = font_scale or self.font_scale
        thickness = thickness or self.font_thickness

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, font_scale, thickness
        )

        x, y = position

        # Draw background rectangle
        cv2.rectangle(
            image,
            (x, y - text_height - 5),
            (x + text_width + 5, y + baseline),
            Colors.TEXT_BG,
            -1
        )

        # Draw text
        cv2.putText(
            image,
            text,
            (x + 2, y - 2),
            self.font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def toggle_lanes(self, enabled: bool):
        """Toggle lane overlay rendering."""
        self.show_lanes = enabled

    def toggle_detections(self, enabled: bool):
        """Toggle detection overlay rendering."""
        self.show_detections = enabled

    def toggle_tracking(self, enabled: bool):
        """Toggle tracking overlay rendering."""
        self.show_tracking = enabled

    def toggle_distance(self, enabled: bool):
        """Toggle distance information display."""
        self.show_distance = enabled

    def toggle_warnings(self, enabled: bool):
        """Toggle warning display."""
        self.show_warnings = enabled

    def toggle_info_panel(self, enabled: bool):
        """Toggle information panel display."""
        self.show_info_panel = enabled

    def toggle_trajectory_prediction(self, enabled: bool):
        """Toggle trajectory prediction display."""
        self.show_trajectory_prediction = enabled
