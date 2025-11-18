"""
Telemetry dashboard widget with real-time performance graphs.

Provides visualization of system metrics including FPS, latency,
CPU/memory usage, and perception statistics.
"""

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QGridLayout, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from collections import deque
import time

from utils.logger import get_logger


logger = get_logger()


class TelemetryDashboard(QWidget):
    """
    Telemetry dashboard widget with real-time graphs and metrics.

    Displays:
    - Camera FPS per camera
    - Processing latency
    - CPU and memory usage
    - Detection and tracking statistics
    - Safety system status
    """

    def __init__(self, parent=None):
        """
        Initialize telemetry dashboard.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Data buffers (time-series data)
        self.max_points = 300  # 10 seconds at 30 FPS
        self.time_buffer = deque(maxlen=self.max_points)
        self.start_time = time.time()

        # Camera FPS buffers (per camera)
        self.camera_fps_buffers = {
            0: deque(maxlen=self.max_points),
            1: deque(maxlen=self.max_points),
            2: deque(maxlen=self.max_points),
            3: deque(maxlen=self.max_points),
        }

        # Processing metrics buffers
        self.latency_buffer = deque(maxlen=self.max_points)
        self.cpu_buffer = deque(maxlen=self.max_points)
        self.memory_buffer = deque(maxlen=self.max_points)

        # Detection/tracking metrics buffers
        self.detections_buffer = deque(maxlen=self.max_points)
        self.tracked_objects_buffer = deque(maxlen=self.max_points)

        # Setup UI
        self._setup_ui()

        logger.info("Telemetry dashboard initialized")

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("System Telemetry Dashboard")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Tab widget for different metric categories
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: white;
                padding: 8px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4a4a4a;
                border-bottom: 2px solid #0078d7;
            }
        """)

        # Camera Performance Tab
        self.camera_tab = self._create_camera_performance_tab()
        self.tabs.addTab(self.camera_tab, "Camera Performance")

        # System Resources Tab
        self.resources_tab = self._create_system_resources_tab()
        self.tabs.addTab(self.resources_tab, "System Resources")

        # Perception Tab
        self.perception_tab = self._create_perception_tab()
        self.tabs.addTab(self.perception_tab, "Perception")

        # Safety Tab
        self.safety_tab = self._create_safety_tab()
        self.tabs.addTab(self.safety_tab, "Safety")

        layout.addWidget(self.tabs)

        # Summary stats at bottom
        self.summary_widget = self._create_summary_widget()
        layout.addWidget(self.summary_widget)

    def _create_camera_performance_tab(self) -> QWidget:
        """Create camera performance tab with FPS graphs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # FPS Graph
        fps_group = QGroupBox("Camera FPS")
        fps_layout = QVBoxLayout(fps_group)

        self.fps_plot = pg.PlotWidget()
        self.fps_plot.setBackground('#2a2a2a')
        self.fps_plot.setLabel('left', 'FPS', units='fps')
        self.fps_plot.setLabel('bottom', 'Time', units='s')
        self.fps_plot.setTitle("Camera Frame Rate")
        self.fps_plot.addLegend()
        self.fps_plot.showGrid(x=True, y=True, alpha=0.3)

        # Create plot curves for each camera
        self.fps_curves = {}
        colors = ['#00ff00', '#ff0000', '#0000ff', '#ffff00']  # Green, Red, Blue, Yellow
        for camera_id, color in enumerate(colors):
            curve = self.fps_plot.plot(
                pen=pg.mkPen(color, width=2),
                name=f'Camera {camera_id}'
            )
            self.fps_curves[camera_id] = curve

        fps_layout.addWidget(self.fps_plot)
        layout.addWidget(fps_group)

        # Latency Graph
        latency_group = QGroupBox("Processing Latency")
        latency_layout = QVBoxLayout(latency_group)

        self.latency_plot = pg.PlotWidget()
        self.latency_plot.setBackground('#2a2a2a')
        self.latency_plot.setLabel('left', 'Latency', units='ms')
        self.latency_plot.setLabel('bottom', 'Time', units='s')
        self.latency_plot.setTitle("Perception Pipeline Latency")
        self.latency_plot.showGrid(x=True, y=True, alpha=0.3)

        self.latency_curve = self.latency_plot.plot(
            pen=pg.mkPen('#ff8800', width=2)
        )

        latency_layout.addWidget(self.latency_plot)
        layout.addWidget(latency_group)

        return widget

    def _create_system_resources_tab(self) -> QWidget:
        """Create system resources tab with CPU/Memory graphs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # CPU Usage Graph
        cpu_group = QGroupBox("CPU Usage")
        cpu_layout = QVBoxLayout(cpu_group)

        self.cpu_plot = pg.PlotWidget()
        self.cpu_plot.setBackground('#2a2a2a')
        self.cpu_plot.setLabel('left', 'Usage', units='%')
        self.cpu_plot.setLabel('bottom', 'Time', units='s')
        self.cpu_plot.setTitle("CPU Utilization")
        self.cpu_plot.setYRange(0, 100)
        self.cpu_plot.showGrid(x=True, y=True, alpha=0.3)

        self.cpu_curve = self.cpu_plot.plot(
            pen=pg.mkPen('#00ffff', width=2),
            fillLevel=0,
            brush=(0, 255, 255, 50)
        )

        cpu_layout.addWidget(self.cpu_plot)
        layout.addWidget(cpu_group)

        # Memory Usage Graph
        memory_group = QGroupBox("Memory Usage")
        memory_layout = QVBoxLayout(memory_group)

        self.memory_plot = pg.PlotWidget()
        self.memory_plot.setBackground('#2a2a2a')
        self.memory_plot.setLabel('left', 'Memory', units='GB')
        self.memory_plot.setLabel('bottom', 'Time', units='s')
        self.memory_plot.setTitle("Memory Consumption")
        self.memory_plot.showGrid(x=True, y=True, alpha=0.3)

        self.memory_curve = self.memory_plot.plot(
            pen=pg.mkPen('#ff00ff', width=2),
            fillLevel=0,
            brush=(255, 0, 255, 50)
        )

        memory_layout.addWidget(self.memory_plot)
        layout.addWidget(memory_group)

        return widget

    def _create_perception_tab(self) -> QWidget:
        """Create perception tab with detection/tracking graphs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Detection Count Graph
        detection_group = QGroupBox("Object Detection")
        detection_layout = QVBoxLayout(detection_group)

        self.detection_plot = pg.PlotWidget()
        self.detection_plot.setBackground('#2a2a2a')
        self.detection_plot.setLabel('left', 'Count', units='objects')
        self.detection_plot.setLabel('bottom', 'Time', units='s')
        self.detection_plot.setTitle("Detected Objects")
        self.detection_plot.showGrid(x=True, y=True, alpha=0.3)

        self.detection_curve = self.detection_plot.plot(
            pen=pg.mkPen('#ffaa00', width=2)
        )

        detection_layout.addWidget(self.detection_plot)
        layout.addWidget(detection_group)

        # Tracked Objects Graph
        tracking_group = QGroupBox("Object Tracking")
        tracking_layout = QVBoxLayout(tracking_group)

        self.tracking_plot = pg.PlotWidget()
        self.tracking_plot.setBackground('#2a2a2a')
        self.tracking_plot.setLabel('left', 'Count', units='tracks')
        self.tracking_plot.setLabel('bottom', 'Time', units='s')
        self.tracking_plot.setTitle("Tracked Objects")
        self.tracking_plot.showGrid(x=True, y=True, alpha=0.3)

        self.tracking_curve = self.tracking_plot.plot(
            pen=pg.mkPen('#00ff88', width=2)
        )

        tracking_layout.addWidget(self.tracking_plot)
        layout.addWidget(tracking_group)

        return widget

    def _create_safety_tab(self) -> QWidget:
        """Create safety tab with warning statistics."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Safety Status Display
        status_group = QGroupBox("Safety Status")
        status_layout = QGridLayout(status_group)

        self.safety_status_label = QLabel("SAFE")
        self.safety_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.safety_status_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                border-radius: 10px;
                background-color: #00aa00;
                color: white;
            }
        """)

        status_layout.addWidget(self.safety_status_label, 0, 0, 1, 2)
        layout.addWidget(status_group)

        # Warning Counters
        warnings_group = QGroupBox("Warning Statistics")
        warnings_layout = QGridLayout(warnings_group)

        self.fcw_count_label = QLabel("FCW: 0")
        self.ldw_count_label = QLabel("LDW: 0")
        self.bsw_count_label = QLabel("BSW: 0")

        label_style = """
            QLabel {
                font-size: 16px;
                padding: 10px;
                background-color: #3a3a3a;
                border-radius: 5px;
                color: white;
            }
        """
        self.fcw_count_label.setStyleSheet(label_style)
        self.ldw_count_label.setStyleSheet(label_style)
        self.bsw_count_label.setStyleSheet(label_style)

        warnings_layout.addWidget(QLabel("Forward Collision:"), 0, 0)
        warnings_layout.addWidget(self.fcw_count_label, 0, 1)
        warnings_layout.addWidget(QLabel("Lane Departure:"), 1, 0)
        warnings_layout.addWidget(self.ldw_count_label, 1, 1)
        warnings_layout.addWidget(QLabel("Blind Spot:"), 2, 0)
        warnings_layout.addWidget(self.bsw_count_label, 2, 1)

        layout.addWidget(warnings_group)
        layout.addStretch()

        return widget

    def _create_summary_widget(self) -> QWidget:
        """Create summary statistics widget."""
        widget = QGroupBox("Current Statistics")
        layout = QGridLayout(widget)

        # Create stat labels
        self.stat_labels = {}
        stats = [
            ("avg_fps", "Avg FPS:"),
            ("total_detections", "Total Detections:"),
            ("active_tracks", "Active Tracks:"),
            ("cpu_usage", "CPU:"),
            ("memory_usage", "Memory:"),
            ("uptime", "Uptime:")
        ]

        for i, (key, label_text) in enumerate(stats):
            label = QLabel(label_text)
            value_label = QLabel("--")
            value_label.setStyleSheet("font-weight: bold; color: #00ff88;")

            row = i // 3
            col = (i % 3) * 2

            layout.addWidget(label, row, col)
            layout.addWidget(value_label, row, col + 1)

            self.stat_labels[key] = value_label

        return widget

    @pyqtSlot(dict)
    def update_telemetry(self, data: dict):
        """
        Update telemetry with new data.

        Args:
            data: Dictionary with telemetry data
        """
        try:
            current_time = time.time() - self.start_time
            self.time_buffer.append(current_time)

            # Update camera FPS
            if 'camera_fps' in data:
                for camera_id, fps in data['camera_fps'].items():
                    self.camera_fps_buffers[camera_id].append(fps)

            # Update latency
            if 'latency' in data:
                self.latency_buffer.append(data['latency'])

            # Update system resources
            if 'cpu_usage' in data:
                self.cpu_buffer.append(data['cpu_usage'])
            if 'memory_usage' in data:
                self.memory_buffer.append(data['memory_usage'])

            # Update perception metrics
            if 'detections' in data:
                self.detections_buffer.append(data['detections'])
            if 'tracked_objects' in data:
                self.tracked_objects_buffer.append(data['tracked_objects'])

            # Update plots
            self._update_plots()

            # Update summary stats
            self._update_summary(data)

            # Update safety status
            if 'safety_status' in data:
                self._update_safety_status(data['safety_status'])

        except Exception as e:
            logger.error(f"Error updating telemetry: {e}")

    def _update_plots(self):
        """Update all plots with current data."""
        time_data = list(self.time_buffer)

        # Update FPS curves
        for camera_id, curve in self.fps_curves.items():
            fps_data = list(self.camera_fps_buffers[camera_id])
            if len(fps_data) == len(time_data):
                curve.setData(time_data, fps_data)

        # Update latency curve
        if len(self.latency_buffer) == len(time_data):
            self.latency_curve.setData(time_data, list(self.latency_buffer))

        # Update CPU curve
        if len(self.cpu_buffer) == len(time_data):
            self.cpu_curve.setData(time_data, list(self.cpu_buffer))

        # Update memory curve
        if len(self.memory_buffer) == len(time_data):
            self.memory_curve.setData(time_data, list(self.memory_buffer))

        # Update detection curve
        if len(self.detections_buffer) == len(time_data):
            self.detection_curve.setData(time_data, list(self.detections_buffer))

        # Update tracking curve
        if len(self.tracked_objects_buffer) == len(time_data):
            self.tracking_curve.setData(time_data, list(self.tracked_objects_buffer))

    def _update_summary(self, data: dict):
        """Update summary statistics."""
        if 'camera_fps' in data and data['camera_fps']:
            avg_fps = sum(data['camera_fps'].values()) / len(data['camera_fps'])
            self.stat_labels['avg_fps'].setText(f"{avg_fps:.1f}")

        if 'total_detections' in data:
            self.stat_labels['total_detections'].setText(str(data['total_detections']))

        if 'tracked_objects' in data:
            self.stat_labels['active_tracks'].setText(str(data['tracked_objects']))

        if 'cpu_usage' in data:
            self.stat_labels['cpu_usage'].setText(f"{data['cpu_usage']:.1f}%")

        if 'memory_usage' in data:
            self.stat_labels['memory_usage'].setText(f"{data['memory_usage']:.2f} GB")

        if 'uptime' in data:
            uptime = int(data['uptime'])
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            seconds = uptime % 60
            self.stat_labels['uptime'].setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def _update_safety_status(self, safety_data: dict):
        """Update safety status display."""
        status = safety_data.get('status', 'safe').upper()
        self.safety_status_label.setText(status)

        # Update background color based on status
        colors = {
            'SAFE': '#00aa00',
            'CAUTION': '#ffaa00',
            'WARNING': '#ff6600',
            'CRITICAL': '#ff0000'
        }
        color = colors.get(status, '#00aa00')
        self.safety_status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                border-radius: 10px;
                background-color: {color};
                color: white;
            }}
        """)

        # Update warning counts
        if 'fcw_warnings' in safety_data:
            self.fcw_count_label.setText(f"FCW: {safety_data['fcw_warnings']}")
        if 'ldw_warnings' in safety_data:
            self.ldw_count_label.setText(f"LDW: {safety_data['ldw_warnings']}")
        if 'bsw_warnings' in safety_data:
            self.bsw_count_label.setText(f"BSW: {safety_data['bsw_warnings']}")

    def clear(self):
        """Clear all telemetry data."""
        self.time_buffer.clear()
        for buffer in self.camera_fps_buffers.values():
            buffer.clear()
        self.latency_buffer.clear()
        self.cpu_buffer.clear()
        self.memory_buffer.clear()
        self.detections_buffer.clear()
        self.tracked_objects_buffer.clear()

        self.start_time = time.time()

        logger.debug("Telemetry dashboard cleared")
