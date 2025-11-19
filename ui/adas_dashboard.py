"""
ADAS Dashboard Widget - Integrated visualization of all advanced features.

Displays:
- Safety score with color-coded level
- Traffic light status
- Scene context (weather, road type, time, lighting)
- Depth visualization toggle
- Segmentation overlay toggle
- Recording status
- Trip statistics
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QProgressBar, QPushButton, QGridLayout, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

from safety.safety_scorer import SafetyScore, SafetyLevel
from perception.scene_recognition import SceneContext
from perception.traffic_light_detection import TrafficLight, TrafficLightState


class SafetyScoreWidget(QWidget):
    """
    Displays safety score with visual indicators.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("SAFETY SCORE")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title.setFont(title_font)
        layout.addWidget(title)

        # Score display
        self.score_label = QLabel("--")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_font = QFont()
        score_font.setBold(True)
        score_font.setPointSize(36)
        self.score_label.setFont(score_font)
        layout.addWidget(self.score_label)

        # Level label
        self.level_label = QLabel("UNKNOWN")
        self.level_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        level_font = QFont()
        level_font.setBold(True)
        level_font.setPointSize(12)
        self.level_label.setFont(level_font)
        layout.addWidget(self.level_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(20)
        layout.addWidget(self.progress)

        # Component scores
        components_group = QGroupBox("Component Scores")
        components_layout = QVBoxLayout()

        self.component_labels = {}
        components = [
            ("Collision Avoidance", "collision"),
            ("Lane Keeping", "lane"),
            ("Following Distance", "following"),
            ("Speed", "speed"),
            ("Environment", "environment")
        ]

        for name, key in components:
            row = QHBoxLayout()
            label = QLabel(f"{name}:")
            label.setMinimumWidth(150)
            value = QLabel("--")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.component_labels[key] = value

            row.addWidget(label)
            row.addWidget(value)
            components_layout.addLayout(row)

        components_group.setLayout(components_layout)
        layout.addWidget(components_group)

        layout.addStretch()

    @pyqtSlot(object)
    def update_score(self, safety_score: SafetyScore):
        """Update safety score display."""
        if safety_score is None:
            return

        # Update main score
        score = int(safety_score.overall_score)
        self.score_label.setText(f"{score}")
        self.progress.setValue(score)

        # Update level
        level = safety_score.safety_level.value.upper()
        self.level_label.setText(level)

        # Color code based on level
        if safety_score.safety_level == SafetyLevel.EXCELLENT:
            color = "#00FF00"  # Green
        elif safety_score.safety_level == SafetyLevel.GOOD:
            color = "#90EE90"  # Light green
        elif safety_score.safety_level == SafetyLevel.FAIR:
            color = "#FFFF00"  # Yellow
        elif safety_score.safety_level == SafetyLevel.POOR:
            color = "#FFA500"  # Orange
        else:  # CRITICAL
            color = "#FF0000"  # Red

        self.score_label.setStyleSheet(f"color: {color};")
        self.level_label.setStyleSheet(f"color: {color};")
        self.progress.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

        # Update component scores
        self.component_labels["collision"].setText(
            f"{safety_score.collision_avoidance_score:.0f}"
        )
        self.component_labels["lane"].setText(
            f"{safety_score.lane_keeping_score:.0f}"
        )
        self.component_labels["following"].setText(
            f"{safety_score.following_distance_score:.0f}"
        )
        self.component_labels["speed"].setText(
            f"{safety_score.speed_appropriateness_score:.0f}"
        )
        self.component_labels["environment"].setText(
            f"{safety_score.environmental_awareness_score:.0f}"
        )


class SceneContextWidget(QWidget):
    """
    Displays scene recognition context.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("SCENE CONTEXT")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title.setFont(title_font)
        layout.addWidget(title)

        # Context info
        grid = QGridLayout()

        # Weather
        grid.addWidget(QLabel("Weather:"), 0, 0)
        self.weather_label = QLabel("--")
        self.weather_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.weather_label, 0, 1)

        # Road type
        grid.addWidget(QLabel("Road Type:"), 1, 0)
        self.road_label = QLabel("--")
        self.road_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.road_label, 1, 1)

        # Time of day
        grid.addWidget(QLabel("Time:"), 2, 0)
        self.time_label = QLabel("--")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.time_label, 2, 1)

        # Lighting
        grid.addWidget(QLabel("Lighting:"), 3, 0)
        self.lighting_label = QLabel("--")
        self.lighting_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.lighting_label, 3, 1)

        # Visibility
        grid.addWidget(QLabel("Visibility:"), 4, 0)
        self.visibility_label = QLabel("--")
        self.visibility_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.visibility_label, 4, 1)

        layout.addLayout(grid)
        layout.addStretch()

    @pyqtSlot(object)
    def update_context(self, context: SceneContext):
        """Update scene context display."""
        if context is None:
            return

        self.weather_label.setText(context.weather.value.title())
        self.road_label.setText(context.road_type.value.replace('_', ' ').title())
        self.time_label.setText(context.time_of_day.value.title())
        self.lighting_label.setText(context.lighting.value.title())
        self.visibility_label.setText(f"{context.visibility_score:.2f}")

        # Color code visibility
        if context.visibility_score > 0.7:
            vis_color = "green"
        elif context.visibility_score > 0.4:
            vis_color = "orange"
        else:
            vis_color = "red"

        self.visibility_label.setStyleSheet(f"color: {vis_color}; font-weight: bold;")


class TrafficLightWidget(QWidget):
    """
    Displays traffic light status.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("TRAFFIC LIGHT")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title.setFont(title_font)
        layout.addWidget(title)

        # Light indicators
        self.red_light = QLabel("●")
        self.red_light.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.red_light.setStyleSheet("color: #555555; font-size: 48px;")
        layout.addWidget(self.red_light)

        self.yellow_light = QLabel("●")
        self.yellow_light.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.yellow_light.setStyleSheet("color: #555555; font-size: 48px;")
        layout.addWidget(self.yellow_light)

        self.green_light = QLabel("●")
        self.green_light.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.green_light.setStyleSheet("color: #555555; font-size: 48px;")
        layout.addWidget(self.green_light)

        # Status
        self.status_label = QLabel("NO SIGNAL")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_font = QFont()
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        layout.addWidget(self.status_label)

        layout.addStretch()

    @pyqtSlot(list)
    def update_lights(self, traffic_lights: list):
        """Update traffic light display."""
        # Reset all lights
        self.red_light.setStyleSheet("color: #555555; font-size: 48px;")
        self.yellow_light.setStyleSheet("color: #555555; font-size: 48px;")
        self.green_light.setStyleSheet("color: #555555; font-size: 48px;")
        self.status_label.setText("NO SIGNAL")
        self.status_label.setStyleSheet("")

        if not traffic_lights:
            return

        # Get highest confidence light
        best_light = max(traffic_lights, key=lambda l: l.confidence)

        # Update display based on state
        if best_light.state == TrafficLightState.RED:
            self.red_light.setStyleSheet("color: #FF0000; font-size: 48px;")
            self.status_label.setText("STOP")
            self.status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
        elif best_light.state == TrafficLightState.YELLOW:
            self.yellow_light.setStyleSheet("color: #FFFF00; font-size: 48px;")
            self.status_label.setText("CAUTION")
            self.status_label.setStyleSheet("color: #FFFF00; font-weight: bold;")
        elif best_light.state == TrafficLightState.GREEN:
            self.green_light.setStyleSheet("color: #00FF00; font-size: 48px;")
            self.status_label.setText("GO")
            self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")


class RecordingStatusWidget(QWidget):
    """
    Displays recording status and controls.
    """

    start_recording = pyqtSignal()
    stop_recording = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_recording = False
        self._setup_ui()

        # Blink timer for recording indicator
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._blink_indicator)
        self.blink_state = False

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("RECORDING")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title.setFont(title_font)
        layout.addWidget(title)

        # Recording indicator
        self.indicator = QLabel("●")
        self.indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.indicator.setStyleSheet("color: #555555; font-size: 64px;")
        layout.addWidget(self.indicator)

        # Status
        self.status_label = QLabel("STOPPED")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_font = QFont()
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        layout.addWidget(self.status_label)

        # Control button
        self.control_button = QPushButton("START RECORDING")
        self.control_button.clicked.connect(self._on_button_clicked)
        layout.addWidget(self.control_button)

        layout.addStretch()

    def _on_button_clicked(self):
        """Handle button click."""
        if self.is_recording:
            self.stop_recording.emit()
        else:
            self.start_recording.emit()

    def _blink_indicator(self):
        """Blink recording indicator."""
        if self.blink_state:
            self.indicator.setStyleSheet("color: #FF0000; font-size: 64px;")
        else:
            self.indicator.setStyleSheet("color: #555555; font-size: 64px;")

        self.blink_state = not self.blink_state

    @pyqtSlot(bool)
    def set_recording(self, recording: bool):
        """Set recording state."""
        self.is_recording = recording

        if recording:
            self.control_button.setText("STOP RECORDING")
            self.status_label.setText("RECORDING")
            self.status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
            self.blink_timer.start(500)  # Blink every 500ms
        else:
            self.control_button.setText("START RECORDING")
            self.status_label.setText("STOPPED")
            self.status_label.setStyleSheet("")
            self.blink_timer.stop()
            self.indicator.setStyleSheet("color: #555555; font-size: 64px;")


class ADASDashboard(QWidget):
    """
    Main ADAS dashboard widget integrating all features.
    """

    start_recording = pyqtSignal()
    stop_recording = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        main_layout = QHBoxLayout(self)

        # Left column - Safety and Scene
        left_column = QVBoxLayout()

        self.safety_widget = SafetyScoreWidget()
        left_column.addWidget(self.safety_widget)

        self.scene_widget = SceneContextWidget()
        left_column.addWidget(self.scene_widget)

        main_layout.addLayout(left_column, stretch=2)

        # Right column - Traffic lights and recording
        right_column = QVBoxLayout()

        self.traffic_widget = TrafficLightWidget()
        right_column.addWidget(self.traffic_widget)

        self.recording_widget = RecordingStatusWidget()
        self.recording_widget.start_recording.connect(self.start_recording)
        self.recording_widget.stop_recording.connect(self.stop_recording)
        right_column.addWidget(self.recording_widget)

        main_layout.addLayout(right_column, stretch=1)

        # Style
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)

    @pyqtSlot(object)
    def update_safety_score(self, safety_score: SafetyScore):
        """Update safety score."""
        self.safety_widget.update_score(safety_score)

    @pyqtSlot(object)
    def update_scene_context(self, context: SceneContext):
        """Update scene context."""
        self.scene_widget.update_context(context)

    @pyqtSlot(list)
    def update_traffic_lights(self, lights: list):
        """Update traffic lights."""
        self.traffic_widget.update_lights(lights)

    @pyqtSlot(bool)
    def set_recording_state(self, recording: bool):
        """Set recording state."""
        self.recording_widget.set_recording(recording)
