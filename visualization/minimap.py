"""
Minimap widget for displaying bird's eye view in the UI.

Provides a real-time top-down view of the vehicle's surroundings
with detected objects, trajectories, and camera FOV.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from typing import List, Optional, Dict

from utils.data_structures import DetectedObject, TrackedObject, CameraConfig
from visualization.bev_generator import BEVGenerator, BEVConfig
from utils.logger import get_logger


logger = get_logger()


class MinimapWidget(QWidget):
    """
    Widget for displaying bird's eye view minimap.

    Shows a real-time top-down visualization of the vehicle and its surroundings.
    """

    def __init__(self, parent=None):
        """
        Initialize minimap widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # BEV generator
        self.bev_config = BEVConfig(
            width_pixels=400,
            height_pixels=600,
            meters_per_pixel=0.05,
            forward_range_m=30.0,
            rear_range_m=10.0,
            lateral_range_m=10.0,
            show_grid=True,
            show_distance_circles=True,
            show_camera_fov=True,
            show_trajectories=True
        )
        self.bev_generator = BEVGenerator(self.bev_config)

        # Camera configurations (will be set externally)
        self.camera_configs: Optional[Dict[int, CameraConfig]] = None

        # Current data
        self.current_detections: List[DetectedObject] = []
        self.current_tracked_objects: List[TrackedObject] = []

        # Setup UI
        self._setup_ui()

        # Update timer (optional - updates are driven by data)
        self.last_update_time = 0

        logger.info("Minimap widget initialized")

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Frame for the minimap
        self.frame = QFrame()
        self.frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.frame.setLineWidth(2)
        self.frame.setStyleSheet("border: 2px solid #555555; background-color: #1a1a1a;")

        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 600)
        self.image_label.setStyleSheet("background-color: #1e1e1e;")

        # Initialize with empty BEV
        self._update_display()

        frame_layout.addWidget(self.image_label)
        layout.addWidget(self.frame)

    def set_camera_configs(self, camera_configs: Dict[int, CameraConfig]):
        """
        Set camera configurations for FOV visualization.

        Args:
            camera_configs: Dictionary mapping camera_id to CameraConfig
        """
        self.camera_configs = camera_configs
        logger.debug("Minimap camera configs updated")

    @pyqtSlot(list, list)
    def update_minimap(
        self,
        detections: Optional[List[DetectedObject]] = None,
        tracked_objects: Optional[List[TrackedObject]] = None
    ):
        """
        Update minimap with new detection/tracking data.

        Args:
            detections: List of detected objects
            tracked_objects: List of tracked objects
        """
        if detections is not None:
            self.current_detections = detections
        if tracked_objects is not None:
            self.current_tracked_objects = tracked_objects

        self._update_display()

    def _update_display(self):
        """Update the minimap display."""
        try:
            # Generate BEV image
            bev_image = self.bev_generator.generate(
                detections=self.current_detections if not self.current_tracked_objects else None,
                tracked_objects=self.current_tracked_objects,
                camera_configs=self.camera_configs
            )

            # Convert to QPixmap and display
            self._display_image(bev_image)

        except Exception as e:
            logger.error(f"Error updating minimap display: {e}")

    def _display_image(self, image: np.ndarray):
        """
        Display a BGR image on the widget.

        Args:
            image: BGR image
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create QImage
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def set_config(self, config: BEVConfig):
        """
        Update BEV configuration.

        Args:
            config: New BEV configuration
        """
        self.bev_config = config
        self.bev_generator = BEVGenerator(config)
        self._update_display()
        logger.info("Minimap configuration updated")

    def toggle_grid(self, enabled: bool):
        """Toggle grid display."""
        self.bev_config.show_grid = enabled
        self._update_display()

    def toggle_distance_circles(self, enabled: bool):
        """Toggle distance circles display."""
        self.bev_config.show_distance_circles = enabled
        self._update_display()

    def toggle_camera_fov(self, enabled: bool):
        """Toggle camera FOV display."""
        self.bev_config.show_camera_fov = enabled
        self._update_display()

    def toggle_trajectories(self, enabled: bool):
        """Toggle trajectory display."""
        self.bev_config.show_trajectories = enabled
        self._update_display()

    def clear(self):
        """Clear minimap data."""
        self.current_detections = []
        self.current_tracked_objects = []
        self._update_display()

    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        # Could rescale image here if needed
