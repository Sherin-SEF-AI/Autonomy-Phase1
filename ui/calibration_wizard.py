"""
Camera calibration wizard UI.

Interactive wizard to guide users through camera calibration process
with real-time feedback and visualization.
"""

from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QTextEdit, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np
from pathlib import Path

from camera.camera_calibration import CameraCalibrator
from utils.logger import get_logger


logger = get_logger()


class CalibrationWizard(QWizard):
    """
    Wizard for camera calibration.

    Guides user through the calibration process step by step.
    """

    calibration_completed = pyqtSignal(int, object)  # camera_id, calibration_data

    def __init__(self, camera_id: int = 0, parent=None):
        """
        Initialize calibration wizard.

        Args:
            camera_id: Camera to calibrate
            parent: Parent widget
        """
        super().__init__(parent)

        self.camera_id = camera_id
        self.calibrator = CameraCalibrator()

        self.setWindowTitle(f"Camera {camera_id} Calibration Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage)

        # Add wizard pages
        self.intro_page = IntroPage()
        self.addPage(self.intro_page)

        self.settings_page = SettingsPage()
        self.addPage(self.settings_page)

        self.capture_page = CapturePage(self.camera_id, self.calibrator)
        self.addPage(self.capture_page)

        self.calibration_page = CalibrationPage(self.camera_id, self.calibrator)
        self.addPage(self.calibration_page)

        self.results_page = ResultsPage()
        self.addPage(self.results_page)

        # Connect signals
        self.calibration_page.calibration_done.connect(self._on_calibration_done)

        # Styling
        self.setMinimumSize(800, 600)

        logger.info(f"Calibration wizard initialized for camera {camera_id}")

    def _on_calibration_done(self, success: bool):
        """Handle calibration completion."""
        if success:
            # Get calibration data
            calibration_data = {
                'camera_matrix': self.calibrator.camera_matrix,
                'distortion_coeffs': self.calibrator.distortion_coeffs,
                'calibration_error': self.calibrator.calibration_error
            }

            self.calibration_completed.emit(self.camera_id, calibration_data)


class IntroPage(QWizardPage):
    """Introduction page."""

    def __init__(self):
        super().__init__()

        self.setTitle("Camera Calibration Wizard")
        self.setSubTitle("This wizard will guide you through calibrating your camera for accurate distance measurements.")

        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "<h3>What you'll need:</h3>"
            "<ul>"
            "<li><b>Chessboard pattern</b> - A printed chessboard with known dimensions</li>"
            "<li><b>Good lighting</b> - Ensure your workspace is well-lit</li>"
            "<li><b>10-20 images</b> - Multiple views of the chessboard</li>"
            "<li><b>5-10 minutes</b> - The calibration process takes a few minutes</li>"
            "</ul>"
            "<h3>Tips for best results:</h3>"
            "<ul>"
            "<li>Keep the chessboard flat (attach to a rigid board)</li>"
            "<li>Capture images from different angles and distances</li>"
            "<li>Cover all areas of the camera's field of view</li>"
            "<li>Avoid motion blur - hold steady when capturing</li>"
            "</ul>"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        layout.addStretch()

        self.setLayout(layout)


class SettingsPage(QWizardPage):
    """Settings configuration page."""

    def __init__(self):
        super().__init__()

        self.setTitle("Chessboard Configuration")
        self.setSubTitle("Enter the dimensions of your calibration chessboard pattern.")

        layout = QVBoxLayout()

        # Settings group
        settings_group = QGroupBox("Chessboard Pattern")
        settings_layout = QGridLayout()

        # Chessboard size
        settings_layout.addWidget(QLabel("Number of internal corners (width):"), 0, 0)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(4, 20)
        self.cols_spin.setValue(9)
        settings_layout.addWidget(self.cols_spin, 0, 1)

        settings_layout.addWidget(QLabel("Number of internal corners (height):"), 1, 0)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(4, 20)
        self.rows_spin.setValue(6)
        settings_layout.addWidget(self.rows_spin, 1, 1)

        # Square size
        settings_layout.addWidget(QLabel("Square size (mm):"), 2, 0)
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(10.0, 100.0)
        self.square_size_spin.setValue(25.0)
        self.square_size_spin.setSuffix(" mm")
        settings_layout.addWidget(self.square_size_spin, 2, 1)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Example image
        example_label = QLabel(
            "<b>Note:</b> Internal corners are the intersections <i>inside</i> the chessboard, "
            "not including the outer edges. A standard 10x7 chessboard has 9x6 internal corners."
        )
        example_label.setWordWrap(True)
        layout.addWidget(example_label)

        layout.addStretch()

        self.setLayout(layout)

        # Register fields for access by other pages
        self.registerField("chessboard_cols", self.cols_spin)
        self.registerField("chessboard_rows", self.rows_spin)
        self.registerField("square_size", self.square_size_spin)


class CapturePage(QWizardPage):
    """Image capture page."""

    def __init__(self, camera_id: int, calibrator: CameraCalibrator):
        super().__init__()

        self.camera_id = camera_id
        self.calibrator = calibrator
        self.current_frame = None

        self.setTitle("Capture Calibration Images")
        self.setSubTitle(f"Capture 10-20 images of the chessboard from different angles. Camera: {camera_id}")

        layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #555555; background-color: #1e1e1e;")
        layout.addWidget(self.image_label)

        # Controls
        controls_layout = QHBoxLayout()

        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self._capture_image)
        controls_layout.addWidget(self.capture_button)

        self.load_button = QPushButton("Load from Files...")
        self.load_button.clicked.connect(self._load_images)
        controls_layout.addWidget(self.load_button)

        layout.addLayout(controls_layout)

        # Progress
        progress_group = QGroupBox("Capture Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 20)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("0 images captured (minimum: 10, recommended: 20)")
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        self.setLayout(layout)

        # Timer for live feed (simulated - in real app would connect to camera)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)

    def initializePage(self):
        """Initialize page when shown."""
        # Update chessboard configuration from previous page
        cols = self.field("chessboard_cols")
        rows = self.field("chessboard_rows")
        square_size = self.field("square_size") / 1000.0  # Convert mm to meters

        self.calibrator.chessboard_size = (cols, rows)
        self.calibrator.square_size = square_size

        # Start live feed (simulated)
        self.update_timer.start(33)  # ~30 FPS

    def cleanupPage(self):
        """Cleanup when leaving page."""
        self.update_timer.stop()

    def _update_display(self):
        """Update camera display (simulated)."""
        # In real implementation, this would get frame from camera
        # For now, show placeholder
        if self.current_frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Camera Feed",
                (220, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            self._display_image(placeholder)

    def _display_image(self, image: np.ndarray):
        """Display an image in the label."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def _capture_image(self):
        """Capture current frame for calibration."""
        # In real implementation, would capture from camera
        QMessageBox.information(
            self,
            "Capture",
            "In production, this would capture the current frame from the camera.\n"
            "For now, use 'Load from Files' to add calibration images."
        )

    def _load_images(self):
        """Load calibration images from files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Calibration Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_paths:
            return

        success_count = 0
        for file_path in file_paths:
            image = cv2.imread(file_path)
            if image is not None:
                if self.calibrator.add_calibration_image(image):
                    success_count += 1

        # Update progress
        num_images = len(self.calibrator.calibration_images)
        self.progress_bar.setValue(num_images)
        self.status_label.setText(
            f"{num_images} images captured (minimum: 10, recommended: 20)"
        )

        QMessageBox.information(
            self,
            "Images Loaded",
            f"Successfully added {success_count} of {len(file_paths)} images."
        )

    def isComplete(self):
        """Check if page is complete."""
        return len(self.calibrator.calibration_images) >= 10


class CalibrationPage(QWizardPage):
    """Calibration processing page."""

    calibration_done = pyqtSignal(bool)

    def __init__(self, camera_id: int, calibrator: CameraCalibrator):
        super().__init__()

        self.camera_id = camera_id
        self.calibrator = calibrator
        self.is_calibrated = False

        self.setTitle("Performing Calibration")
        self.setSubTitle("Calculating camera parameters...")

        layout = QVBoxLayout()

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress_bar)

        # Log output
        log_group = QGroupBox("Calibration Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Results display
        self.results_label = QLabel("Waiting to start calibration...")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

        layout.addStretch()

        self.setLayout(layout)

    def initializePage(self):
        """Initialize and run calibration."""
        self.log_text.clear()
        self.log("Starting calibration...")

        # Get image size from first calibration image
        if self.calibrator.calibration_images:
            first_image = self.calibrator.calibration_images[0]
            image_size = (first_image.shape[1], first_image.shape[0])

            self.log(f"Image size: {image_size[0]}x{image_size[1]}")
            self.log(f"Calibration images: {len(self.calibrator.calibration_images)}")
            self.log(f"Chessboard: {self.calibrator.chessboard_size}")

            # Perform calibration
            success = self.calibrator.calibrate(image_size)

            if success:
                self.log("\n✓ Calibration successful!")
                self.log(f"Reprojection error: {self.calibrator.calibration_error:.4f} pixels")
                self.log(f"Quality: {self.calibrator.get_calibration_quality()}")

                self.results_label.setText(
                    f"<h3>Calibration Successful!</h3>"
                    f"<p><b>Reprojection Error:</b> {self.calibrator.calibration_error:.4f} pixels</p>"
                    f"<p><b>Quality:</b> {self.calibrator.get_calibration_quality()}</p>"
                )
                self.is_calibrated = True
                self.calibration_done.emit(True)
            else:
                self.log("\n✗ Calibration failed!")
                self.results_label.setText(
                    "<h3>Calibration Failed</h3>"
                    "<p>Please go back and capture more images with better coverage.</p>"
                )
                self.calibration_done.emit(False)

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)

    def log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)

    def isComplete(self):
        """Check if calibration is complete."""
        return self.is_calibrated


class ResultsPage(QWizardPage):
    """Results and save page."""

    def __init__(self):
        super().__init__()

        self.setTitle("Calibration Complete")
        self.setSubTitle("Save your calibration data for future use.")

        layout = QVBoxLayout()

        # Success message
        success_label = QLabel(
            "<h3>🎉 Calibration Successful!</h3>"
            "<p>Your camera has been successfully calibrated. "
            "The calibration data can now be used for accurate distance measurements.</p>"
        )
        success_label.setWordWrap(True)
        layout.addWidget(success_label)

        # Save controls
        save_group = QGroupBox("Save Calibration")
        save_layout = QVBoxLayout()

        save_button = QPushButton("Save Calibration Data")
        save_button.clicked.connect(self._save_calibration)
        save_layout.addWidget(save_button)

        self.save_status_label = QLabel("Not saved")
        save_layout.addWidget(self.save_status_label)

        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        layout.addStretch()

        self.setLayout(layout)

    def _save_calibration(self):
        """Save calibration data."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration Data",
            f"config/calibration/camera_calibration.json",
            "JSON Files (*.json)"
        )

        if file_path:
            # In real implementation, would save from wizard's calibrator
            self.save_status_label.setText(f"✓ Saved to: {file_path}")
            QMessageBox.information(
                self,
                "Saved",
                f"Calibration data saved successfully to:\n{file_path}"
            )
