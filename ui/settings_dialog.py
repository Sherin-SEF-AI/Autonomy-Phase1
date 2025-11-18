"""
Settings and configuration dialogs for the perception system.

Provides UI for camera settings, algorithm parameters, and system configuration.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QGroupBox, QTabWidget, QWidget,
    QCheckBox, QSlider, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Optional
from utils.data_structures import CameraConfig, CameraPosition
from utils.logger import get_logger


logger = get_logger()


class CameraSettingsDialog(QDialog):
    """
    Dialog for configuring individual camera settings.
    """

    settings_updated = pyqtSignal(CameraConfig)  # Emitted when settings are saved

    def __init__(self, camera_config: Optional[CameraConfig] = None, parent=None):
        """
        Initialize camera settings dialog.

        Args:
            camera_config: Current camera configuration (None for new camera)
            parent: Parent widget
        """
        super().__init__(parent)

        self.camera_config = camera_config
        self.setWindowTitle("Camera Settings")
        self.setMinimumWidth(500)

        self._setup_ui()
        if camera_config is not None:
            self._load_config(camera_config)

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Tabs for different setting categories
        tabs = QTabWidget()

        # Basic Settings Tab
        basic_tab = self._create_basic_settings_tab()
        tabs.addTab(basic_tab, "Basic")

        # Camera Properties Tab
        properties_tab = self._create_properties_tab()
        tabs.addTab(properties_tab, "Properties")

        # Calibration Tab
        calibration_tab = self._create_calibration_tab()
        tabs.addTab(calibration_tab, "Calibration")

        layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._on_save)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def _create_basic_settings_tab(self) -> QWidget:
        """Create basic settings tab."""
        widget = QWidget()
        layout = QFormLayout(widget)

        # Camera ID
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        layout.addRow("Camera ID:", self.camera_id_spin)

        # Device Index
        self.device_index_spin = QSpinBox()
        self.device_index_spin.setRange(0, 10)
        layout.addRow("Device Index:", self.device_index_spin)

        # Camera Position
        self.position_combo = QComboBox()
        for pos in CameraPosition:
            self.position_combo.addItem(pos.name, pos)
        layout.addRow("Position:", self.position_combo)

        # Resolution
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("640x480", (640, 480))
        self.resolution_combo.addItem("800x600", (800, 600))
        self.resolution_combo.addItem("1280x720", (1280, 720))
        self.resolution_combo.addItem("1920x1080", (1920, 1080))
        layout.addRow("Resolution:", self.resolution_combo)

        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        layout.addRow("FPS:", self.fps_spin)

        # Enabled
        self.enabled_check = QCheckBox("Camera Enabled")
        self.enabled_check.setChecked(True)
        layout.addRow("", self.enabled_check)

        return widget

    def _create_properties_tab(self) -> QWidget:
        """Create camera properties tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Exposure
        exposure_group = QGroupBox("Exposure")
        exposure_layout = QFormLayout(exposure_group)

        self.exposure_auto_check = QCheckBox("Auto Exposure")
        self.exposure_auto_check.setChecked(True)
        self.exposure_auto_check.toggled.connect(self._on_exposure_auto_toggled)
        exposure_layout.addRow("", self.exposure_auto_check)

        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(-13, -1)
        self.exposure_slider.setValue(-6)
        self.exposure_slider.setEnabled(False)
        self.exposure_label = QLabel("0")
        exposure_layout.addRow("Exposure:", self.exposure_slider)
        self.exposure_slider.valueChanged.connect(
            lambda v: self.exposure_label.setText(str(v))
        )

        layout.addWidget(exposure_group)

        # Brightness
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QFormLayout(brightness_group)

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 255)
        self.brightness_slider.setValue(128)
        self.brightness_label = QLabel("128")
        brightness_layout.addRow("Brightness:", self.brightness_slider)
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_label.setText(str(v))
        )

        layout.addWidget(brightness_group)

        # Contrast
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QFormLayout(contrast_group)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 255)
        self.contrast_slider.setValue(128)
        self.contrast_label = QLabel("128")
        contrast_layout.addRow("Contrast:", self.contrast_slider)
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_label.setText(str(v))
        )

        layout.addWidget(contrast_group)

        # Saturation
        saturation_group = QGroupBox("Saturation")
        saturation_layout = QFormLayout(saturation_group)

        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(0, 255)
        self.saturation_slider.setValue(128)
        self.saturation_label = QLabel("128")
        saturation_layout.addRow("Saturation:", self.saturation_slider)
        self.saturation_slider.valueChanged.connect(
            lambda v: self.saturation_label.setText(str(v))
        )

        layout.addWidget(saturation_group)

        layout.addStretch()

        return widget

    def _create_calibration_tab(self) -> QWidget:
        """Create calibration tab."""
        widget = QWidget()
        layout = QFormLayout(widget)

        # Position offset
        layout.addRow(QLabel("<b>Position Offset (meters)</b>"))

        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-10.0, 10.0)
        self.offset_x_spin.setSingleStep(0.1)
        self.offset_x_spin.setDecimals(2)
        layout.addRow("X (forward):", self.offset_x_spin)

        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-10.0, 10.0)
        self.offset_y_spin.setSingleStep(0.1)
        self.offset_y_spin.setDecimals(2)
        layout.addRow("Y (left):", self.offset_y_spin)

        self.offset_z_spin = QDoubleSpinBox()
        self.offset_z_spin.setRange(-10.0, 10.0)
        self.offset_z_spin.setSingleStep(0.1)
        self.offset_z_spin.setDecimals(2)
        layout.addRow("Z (up):", self.offset_z_spin)

        # Orientation
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("<b>Orientation (degrees)</b>"))

        self.roll_spin = QDoubleSpinBox()
        self.roll_spin.setRange(-180.0, 180.0)
        self.roll_spin.setSingleStep(1.0)
        self.roll_spin.setDecimals(1)
        layout.addRow("Roll:", self.roll_spin)

        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-180.0, 180.0)
        self.pitch_spin.setSingleStep(1.0)
        self.pitch_spin.setDecimals(1)
        layout.addRow("Pitch:", self.pitch_spin)

        self.yaw_spin = QDoubleSpinBox()
        self.yaw_spin.setRange(-180.0, 180.0)
        self.yaw_spin.setSingleStep(1.0)
        self.yaw_spin.setDecimals(1)
        layout.addRow("Yaw:", self.yaw_spin)

        # Calibration file
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("<b>Calibration Data</b>"))

        calib_layout = QHBoxLayout()
        self.calib_file_edit = QLineEdit()
        self.calib_file_edit.setPlaceholderText("No calibration file loaded")
        calib_layout.addWidget(self.calib_file_edit)

        self.load_calib_button = QPushButton("Load...")
        self.load_calib_button.clicked.connect(self._on_load_calibration)
        calib_layout.addWidget(self.load_calib_button)

        layout.addRow("Calibration File:", calib_layout)

        return widget

    def _on_exposure_auto_toggled(self, checked: bool):
        """Handle auto exposure toggle."""
        self.exposure_slider.setEnabled(not checked)

    def _on_load_calibration(self):
        """Handle load calibration button."""
        # TODO: Implement file dialog for loading calibration
        logger.info("Load calibration clicked")

    def _load_config(self, config: CameraConfig):
        """
        Load configuration into UI.

        Args:
            config: Camera configuration to load
        """
        self.camera_id_spin.setValue(config.camera_id)
        self.device_index_spin.setValue(config.device_index)

        # Set position
        index = self.position_combo.findData(config.position)
        if index >= 0:
            self.position_combo.setCurrentIndex(index)

        # Set resolution
        for i in range(self.resolution_combo.count()):
            if self.resolution_combo.itemData(i) == config.resolution:
                self.resolution_combo.setCurrentIndex(i)
                break

        self.fps_spin.setValue(config.fps)
        self.enabled_check.setChecked(config.enabled)

        # Camera properties
        if config.exposure is not None:
            self.exposure_auto_check.setChecked(False)
            self.exposure_slider.setValue(int(config.exposure))

        if config.brightness is not None:
            self.brightness_slider.setValue(int(config.brightness))

        if config.contrast is not None:
            self.contrast_slider.setValue(int(config.contrast))

        if config.saturation is not None:
            self.saturation_slider.setValue(int(config.saturation))

        # Calibration
        self.offset_x_spin.setValue(config.position_offset[0])
        self.offset_y_spin.setValue(config.position_offset[1])
        self.offset_z_spin.setValue(config.position_offset[2])

        self.roll_spin.setValue(config.orientation[0])
        self.pitch_spin.setValue(config.orientation[1])
        self.yaw_spin.setValue(config.orientation[2])

    def _on_save(self):
        """Handle save button click."""
        # Create configuration from UI values
        config = CameraConfig(
            camera_id=self.camera_id_spin.value(),
            device_index=self.device_index_spin.value(),
            position=self.position_combo.currentData(),
            resolution=self.resolution_combo.currentData(),
            fps=self.fps_spin.value(),
            enabled=self.enabled_check.isChecked(),
            exposure=self.exposure_slider.value() if not self.exposure_auto_check.isChecked() else None,
            brightness=float(self.brightness_slider.value()),
            contrast=float(self.contrast_slider.value()),
            saturation=float(self.saturation_slider.value()),
            position_offset=(
                self.offset_x_spin.value(),
                self.offset_y_spin.value(),
                self.offset_z_spin.value()
            ),
            orientation=(
                self.roll_spin.value(),
                self.pitch_spin.value(),
                self.yaw_spin.value()
            )
        )

        self.settings_updated.emit(config)
        self.accept()
