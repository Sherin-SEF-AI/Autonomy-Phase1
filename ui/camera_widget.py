"""
Camera display widget for showing individual camera feeds with overlays.

Each camera gets its own widget that displays the video feed and can show
perception overlays (bounding boxes, lanes, etc.).
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QAction
from typing import Optional
from utils.data_structures import CameraFrame, CameraStatus, CameraPosition
from utils.logger import get_logger


logger = get_logger()


class CameraWidget(QWidget):
    """
    Widget for displaying a single camera feed with status and controls.
    """

    # Signals
    camera_clicked = pyqtSignal(int)  # camera_id
    settings_requested = pyqtSignal(int)  # camera_id
    enable_toggled = pyqtSignal(int, bool)  # (camera_id, enabled)

    def __init__(self, camera_id: int, camera_position: CameraPosition, parent=None):
        """
        Initialize camera widget.

        Args:
            camera_id: Camera identifier
            camera_position: Camera position (front, left, right, dashboard)
            parent: Parent widget
        """
        super().__init__(parent)

        self.camera_id = camera_id
        self.camera_position = camera_position
        self.current_frame: Optional[CameraFrame] = None
        self.current_status = CameraStatus.DISCONNECTED

        # Display settings
        self.show_overlays = True
        self.show_fps = True
        self.show_status = True

        # FPS tracking
        self.current_fps = 0.0
        self.latency_ms = 0.0

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Main frame
        self.frame = QFrame()
        self.frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.frame.setLineWidth(2)
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        # Header with camera name and status
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 3, 5, 3)

        self.name_label = QLabel(f"Camera {self.camera_id} - {self.camera_position.name}")
        self.name_label.setStyleSheet("font-weight: bold; color: white; background: rgba(0, 0, 0, 150);")

        self.status_label = QLabel("●")
        self.status_label.setStyleSheet("color: gray; font-size: 16px;")

        self.fps_label = QLabel("-- FPS")
        self.fps_label.setStyleSheet("color: white; background: rgba(0, 0, 0, 150);")

        header_layout.addWidget(self.name_label)
        header_layout.addStretch()
        header_layout.addWidget(self.fps_label)
        header_layout.addWidget(self.status_label)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.video_label.setText(f"Camera {self.camera_id}\n{self.camera_position.name}\nDisconnected")
        self.video_label.setStyleSheet("color: gray; background-color: black;")

        # Make video label clickable
        self.video_label.mousePressEvent = self._on_click
        self.video_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.video_label.customContextMenuRequested.connect(self._show_context_menu)

        # Add to frame layout
        frame_layout.addWidget(header)
        frame_layout.addWidget(self.video_label)

        layout.addWidget(self.frame)

        # Update frame color based on status
        self._update_frame_style()

    def update_frame(self, frame: CameraFrame):
        """
        Update the displayed frame.

        Args:
            frame: New camera frame to display
        """
        self.current_frame = frame

        # Convert frame to QPixmap
        image = frame.image

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create QImage
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to widget size while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def update_status(self, status: CameraStatus):
        """
        Update camera status indicator.

        Args:
            status: New camera status
        """
        self.current_status = status
        self._update_frame_style()

        # Update status indicator color
        if status == CameraStatus.ACTIVE:
            self.status_label.setStyleSheet("color: #00ff00; font-size: 16px;")  # Green
            self.video_label.setText("")
        elif status == CameraStatus.CONNECTING:
            self.status_label.setStyleSheet("color: #ffaa00; font-size: 16px;")  # Orange
            self.video_label.setText(f"Camera {self.camera_id}\nConnecting...")
        elif status == CameraStatus.ERROR:
            self.status_label.setStyleSheet("color: #ff0000; font-size: 16px;")  # Red
            self.video_label.setText(f"Camera {self.camera_id}\nError")
        elif status == CameraStatus.DISABLED:
            self.status_label.setStyleSheet("color: #888888; font-size: 16px;")  # Gray
            self.video_label.setText(f"Camera {self.camera_id}\nDisabled")
        else:  # DISCONNECTED
            self.status_label.setStyleSheet("color: #888888; font-size: 16px;")  # Gray
            self.video_label.setText(f"Camera {self.camera_id}\n{self.camera_position.name}\nDisconnected")

    def update_fps(self, fps: float):
        """
        Update FPS display.

        Args:
            fps: Current FPS
        """
        self.current_fps = fps
        self.fps_label.setText(f"{fps:.1f} FPS")

        # Color code FPS (green > 25, yellow > 15, red < 15)
        if fps >= 25:
            color = "#00ff00"
        elif fps >= 15:
            color = "#ffaa00"
        else:
            color = "#ff0000"

        self.fps_label.setStyleSheet(f"color: {color}; background: rgba(0, 0, 0, 150);")

    def _update_frame_style(self):
        """Update frame border color based on status."""
        if self.current_status == CameraStatus.ACTIVE:
            border_color = "#00ff00"  # Green
        elif self.current_status == CameraStatus.CONNECTING:
            border_color = "#ffaa00"  # Orange
        elif self.current_status == CameraStatus.ERROR:
            border_color = "#ff0000"  # Red
        else:
            border_color = "#555555"  # Gray

        self.frame.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {border_color};
                background-color: #1a1a1a;
            }}
        """)

    def _on_click(self, event):
        """Handle click on video display."""
        self.camera_clicked.emit(self.camera_id)

    def _show_context_menu(self, pos):
        """Show context menu for camera controls."""
        menu = QMenu(self)

        # Enable/Disable action
        if self.current_status == CameraStatus.DISABLED:
            enable_action = QAction("Enable Camera", self)
            enable_action.triggered.connect(lambda: self.enable_toggled.emit(self.camera_id, True))
            menu.addAction(enable_action)
        else:
            disable_action = QAction("Disable Camera", self)
            disable_action.triggered.connect(lambda: self.enable_toggled.emit(self.camera_id, False))
            menu.addAction(disable_action)

        menu.addSeparator()

        # Settings action
        settings_action = QAction("Camera Settings...", self)
        settings_action.triggered.connect(lambda: self.settings_requested.emit(self.camera_id))
        menu.addAction(settings_action)

        # Show overlays toggle
        overlays_action = QAction("Show Overlays", self)
        overlays_action.setCheckable(True)
        overlays_action.setChecked(self.show_overlays)
        overlays_action.triggered.connect(self._toggle_overlays)
        menu.addAction(overlays_action)

        # Show menu
        menu.exec(self.video_label.mapToGlobal(pos))

    def _toggle_overlays(self):
        """Toggle overlay display."""
        self.show_overlays = not self.show_overlays
        logger.debug(f"Camera {self.camera_id}: Overlays {'enabled' if self.show_overlays else 'disabled'}")

    def set_error_message(self, message: str):
        """
        Display an error message on the camera widget.

        Args:
            message: Error message to display
        """
        self.video_label.setText(f"Camera {self.camera_id}\nError:\n{message}")
        self.video_label.setStyleSheet("color: red; background-color: black;")

    def clear_display(self):
        """Clear the video display."""
        self.video_label.clear()
        self.video_label.setText(f"Camera {self.camera_id}\n{self.camera_position.name}\nNo Signal")
        self.video_label.setStyleSheet("color: gray; background-color: black;")

    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        # Redraw current frame at new size if available
        if self.current_frame is not None:
            self.update_frame(self.current_frame)
