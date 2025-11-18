"""
Main window for the autonomous vehicle perception system.

Provides the primary UI with camera grid, menu bar, control panel, and status display.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QMenuBar, QMenu, QStatusBar, QLabel, QPushButton, QGroupBox,
    QMessageBox, QToolBar, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from typing import Dict, Optional
import time

from utils.data_structures import CameraConfig, CameraPosition, CameraStatus
from utils.logger import get_logger
from camera.camera_manager import CameraManager
from ui.camera_widget import CameraWidget
from ui.settings_dialog import CameraSettingsDialog


logger = get_logger()


class MainWindow(QMainWindow):
    """
    Main application window for the perception system.
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        self.setWindowTitle("Autonomous Vehicle Perception System")
        self.setGeometry(100, 100, 1600, 900)

        # Camera manager
        self.camera_manager = CameraManager(enable_synchronization=True)
        self.camera_widgets: Dict[int, CameraWidget] = {}

        # System state
        self.is_running = False
        self.session_start_time: Optional[float] = None

        # Setup UI components
        self._setup_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()

        # Setup camera update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(1000)  # Update every second

        # Connect camera manager signals
        self._connect_camera_signals()

        # Initialize default camera configuration
        self._initialize_default_cameras()

        logger.info("Main window initialized")

    def _setup_ui(self):
        """Setup the main user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Camera grid (4 cameras in 2x2 grid)
        camera_container = self._create_camera_grid()
        splitter.addWidget(camera_container)

        # Right side: Control panel (placeholder for now)
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)

        # Set splitter sizes (camera grid gets 75%, control panel gets 25%)
        splitter.setSizes([1200, 400])

        main_layout.addWidget(splitter)

    def _create_camera_grid(self) -> QWidget:
        """
        Create the 2x2 camera grid layout.

        Returns:
            Widget containing camera grid
        """
        container = QWidget()
        grid_layout = QGridLayout(container)
        grid_layout.setSpacing(5)

        # Create 4 camera widgets in 2x2 grid
        positions = [
            (0, CameraPosition.DASHBOARD),  # Top-left: Dashboard camera
            (1, CameraPosition.FRONT),      # Top-right: Front camera
            (2, CameraPosition.LEFT),       # Bottom-left: Left camera
            (3, CameraPosition.RIGHT)       # Bottom-right: Right camera
        ]

        for camera_id, position in positions:
            widget = CameraWidget(camera_id, position)

            # Connect signals
            widget.camera_clicked.connect(self._on_camera_clicked)
            widget.settings_requested.connect(self._on_camera_settings_requested)
            widget.enable_toggled.connect(self._on_camera_enable_toggled)

            self.camera_widgets[camera_id] = widget

            # Add to grid
            row = camera_id // 2
            col = camera_id % 2
            grid_layout.addWidget(widget, row, col)

        return container

    def _create_control_panel(self) -> QWidget:
        """
        Create the control panel.

        Returns:
            Control panel widget
        """
        container = QWidget()
        layout = QVBoxLayout(container)

        # System Control Group
        control_group = QGroupBox("System Control")
        control_layout = QVBoxLayout(control_group)

        # Start/Stop button
        self.start_stop_button = QPushButton("Start Perception System")
        self.start_stop_button.setMinimumHeight(50)
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.start_stop_button.clicked.connect(self._on_start_stop_clicked)
        control_layout.addWidget(self.start_stop_button)

        # Emergency stop button
        self.emergency_stop_button = QPushButton("EMERGENCY STOP")
        self.emergency_stop_button.setMinimumHeight(40)
        self.emergency_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.emergency_stop_button.clicked.connect(self._on_emergency_stop)
        self.emergency_stop_button.setEnabled(False)
        control_layout.addWidget(self.emergency_stop_button)

        layout.addWidget(control_group)

        # System Status Group
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)

        self.status_text_label = QLabel("System Idle")
        self.status_text_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        status_layout.addWidget(self.status_text_label)

        self.uptime_label = QLabel("Uptime: 00:00:00")
        status_layout.addWidget(self.uptime_label)

        self.frames_label = QLabel("Frames Processed: 0")
        status_layout.addWidget(self.frames_label)

        layout.addWidget(status_group)

        # Camera Status Group
        cameras_group = QGroupBox("Camera Status")
        cameras_layout = QVBoxLayout(cameras_group)

        self.camera_status_labels = {}
        for camera_id in range(4):
            label = QLabel(f"Camera {camera_id}: Disconnected")
            label.setStyleSheet("color: gray;")
            self.camera_status_labels[camera_id] = label
            cameras_layout.addWidget(label)

        layout.addWidget(cameras_group)

        layout.addStretch()

        return container

    def _create_menus(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut(QKeySequence.StandardKey.New)
        new_session_action.triggered.connect(self._on_new_session)
        file_menu.addAction(new_session_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Cameras Menu
        cameras_menu = menubar.addMenu("&Cameras")

        camera_settings_action = QAction("Camera &Settings...", self)
        camera_settings_action.triggered.connect(self._on_camera_menu_settings)
        cameras_menu.addAction(camera_settings_action)

        discover_cameras_action = QAction("&Discover Cameras", self)
        discover_cameras_action.triggered.connect(self._on_discover_cameras)
        cameras_menu.addAction(discover_cameras_action)

        cameras_menu.addSeparator()

        test_cameras_action = QAction("&Test Cameras", self)
        test_cameras_action.triggered.connect(self._on_test_cameras)
        cameras_menu.addAction(test_cameras_action)

        # View Menu
        view_menu = menubar.addMenu("&View")

        fullscreen_action = QAction("&Full Screen", self)
        fullscreen_action.setShortcut(QKeySequence.StandardKey.FullScreen)
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self._on_toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Add quick access buttons
        start_action = QAction("Start", self)
        start_action.triggered.connect(self._on_start_stop_clicked)
        toolbar.addAction(start_action)

        toolbar.addSeparator()

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._on_camera_menu_settings)
        toolbar.addAction(settings_action)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status message
        self.status_bar.showMessage("Ready")

        # Add permanent widgets to status bar
        self.system_status_label = QLabel("System: Idle")
        self.status_bar.addPermanentWidget(self.system_status_label)

        self.fps_status_label = QLabel("FPS: --")
        self.status_bar.addPermanentWidget(self.fps_status_label)

    def _connect_camera_signals(self):
        """Connect camera manager signals to handlers."""
        self.camera_manager.frame_received.connect(self._on_frame_received)
        self.camera_manager.camera_status_changed.connect(self._on_camera_status_changed)
        self.camera_manager.camera_error.connect(self._on_camera_error)
        self.camera_manager.camera_fps_updated.connect(self._on_camera_fps_updated)

    def _initialize_default_cameras(self):
        """Initialize default camera configurations."""
        logger.info("Initializing default camera configuration...")

        # Discover available cameras
        available_cameras = self.camera_manager.discover_cameras()

        # Create default configuration for 4 cameras
        positions = [CameraPosition.DASHBOARD, CameraPosition.FRONT, CameraPosition.LEFT, CameraPosition.RIGHT]

        for camera_id in range(4):
            # Use available device or default to camera_id
            device_index = available_cameras[camera_id] if camera_id < len(available_cameras) else camera_id

            config = CameraConfig(
                camera_id=camera_id,
                device_index=device_index,
                position=positions[camera_id],
                resolution=(640, 480),
                fps=30,
                enabled=camera_id < len(available_cameras)  # Only enable if device exists
            )

            self.camera_manager.add_camera(config)

        logger.info(f"Initialized {len(available_cameras)} cameras")

    def _on_start_stop_clicked(self):
        """Handle start/stop button click."""
        if self.is_running:
            self._stop_system()
        else:
            self._start_system()

    def _start_system(self):
        """Start the perception system."""
        logger.info("Starting perception system...")

        # Start all cameras
        self.camera_manager.start_all_cameras()

        self.is_running = True
        self.session_start_time = time.time()

        # Update UI
        self.start_stop_button.setText("Stop Perception System")
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.emergency_stop_button.setEnabled(True)
        self.status_text_label.setText("System Running")
        self.system_status_label.setText("System: Running")
        self.status_bar.showMessage("Perception system started")

        logger.info("Perception system started successfully")

    def _stop_system(self):
        """Stop the perception system."""
        logger.info("Stopping perception system...")

        # Stop all cameras
        self.camera_manager.stop_all_cameras()

        self.is_running = False

        # Update UI
        self.start_stop_button.setText("Start Perception System")
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.emergency_stop_button.setEnabled(False)
        self.status_text_label.setText("System Stopped")
        self.system_status_label.setText("System: Idle")
        self.status_bar.showMessage("Perception system stopped")

        logger.info("Perception system stopped")

    def _on_emergency_stop(self):
        """Handle emergency stop button."""
        logger.warning("EMERGENCY STOP activated!")

        QMessageBox.warning(
            self,
            "Emergency Stop",
            "Emergency stop activated!\nAll cameras will be stopped immediately."
        )

        self._stop_system()

    def _on_frame_received(self, frame):
        """Handle new frame from camera."""
        camera_id = frame.camera_id
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].update_frame(frame)

    def _on_camera_status_changed(self, camera_id: int, status: CameraStatus):
        """Handle camera status change."""
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].update_status(status)

        # Update camera status label
        if camera_id in self.camera_status_labels:
            status_text = f"Camera {camera_id}: {status.value.capitalize()}"
            self.camera_status_labels[camera_id].setText(status_text)

            if status == CameraStatus.ACTIVE:
                color = "green"
            elif status == CameraStatus.ERROR:
                color = "red"
            elif status == CameraStatus.CONNECTING:
                color = "orange"
            else:
                color = "gray"

            self.camera_status_labels[camera_id].setStyleSheet(f"color: {color};")

    def _on_camera_error(self, camera_id: int, error_message: str):
        """Handle camera error."""
        logger.error(f"Camera {camera_id} error: {error_message}")
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].set_error_message(error_message)

    def _on_camera_fps_updated(self, camera_id: int, fps: float):
        """Handle camera FPS update."""
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].update_fps(fps)

    def _on_camera_clicked(self, camera_id: int):
        """Handle camera widget clicked."""
        logger.debug(f"Camera {camera_id} clicked")

    def _on_camera_settings_requested(self, camera_id: int):
        """Handle camera settings request."""
        config = self.camera_manager.get_camera_config(camera_id)
        self._show_camera_settings_dialog(config)

    def _on_camera_enable_toggled(self, camera_id: int, enabled: bool):
        """Handle camera enable/disable toggle."""
        if enabled:
            self.camera_manager.start_camera(camera_id)
            logger.info(f"Camera {camera_id} enabled")
        else:
            self.camera_manager.stop_camera(camera_id)
            logger.info(f"Camera {camera_id} disabled")

    def _on_camera_menu_settings(self):
        """Handle camera settings menu action."""
        # Show settings for camera 1 (front camera) by default
        config = self.camera_manager.get_camera_config(1)
        self._show_camera_settings_dialog(config)

    def _show_camera_settings_dialog(self, config: Optional[CameraConfig]):
        """Show camera settings dialog."""
        dialog = CameraSettingsDialog(config, self)
        dialog.settings_updated.connect(self._on_settings_updated)
        dialog.exec()

    def _on_settings_updated(self, config: CameraConfig):
        """Handle updated camera settings."""
        logger.info(f"Updating settings for camera {config.camera_id}")
        self.camera_manager.update_camera_config(config.camera_id, config)
        self.status_bar.showMessage(f"Camera {config.camera_id} settings updated", 3000)

    def _on_discover_cameras(self):
        """Handle discover cameras action."""
        available = self.camera_manager.discover_cameras()
        QMessageBox.information(
            self,
            "Camera Discovery",
            f"Found {len(available)} camera(s):\n" + "\n".join(f"Device {idx}" for idx in available)
        )

    def _on_test_cameras(self):
        """Handle test cameras action."""
        QMessageBox.information(
            self,
            "Test Cameras",
            "Camera test mode not yet implemented.\nThis will allow testing individual cameras."
        )

    def _on_new_session(self):
        """Handle new session action."""
        if self.is_running:
            reply = QMessageBox.question(
                self,
                "New Session",
                "Stop current session and start a new one?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._stop_system()

        logger.info("New session started")
        self.status_bar.showMessage("New session started", 3000)

    def _on_toggle_fullscreen(self, checked: bool):
        """Handle fullscreen toggle."""
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    def _on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Autonomous Vehicle Perception System",
            "<h2>Autonomous Vehicle Perception System</h2>"
            "<p>Version 1.0.0</p>"
            "<p>Professional-grade multi-camera perception system for autonomous vehicle development.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Multi-camera capture (4 cameras)</li>"
            "<li>Real-time perception pipeline</li>"
            "<li>Lane detection and object tracking</li>"
            "<li>Sensor fusion</li>"
            "<li>Safety warnings</li>"
            "</ul>"
            "<p>Built with PyQt6, OpenCV, and YOLOv8</p>"
        )

    def _update_status(self):
        """Update status information periodically."""
        if self.is_running and self.session_start_time is not None:
            # Calculate uptime
            uptime_seconds = int(time.time() - self.session_start_time)
            hours = uptime_seconds // 3600
            minutes = (uptime_seconds % 3600) // 60
            seconds = uptime_seconds % 60
            self.uptime_label.setText(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.is_running:
            reply = QMessageBox.question(
                self,
                "Quit Application",
                "Perception system is running. Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self._stop_system()

        # Cleanup
        self.camera_manager.cleanup()
        event.accept()
        logger.info("Application closed")
