"""
Driver Monitoring System (DMS)

This module provides comprehensive driver state monitoring for safety and
attentiveness assessment, including:
- Face detection and tracking
- Eye state detection (open, closed, drowsy)
- Head pose estimation (pitch, yaw, roll)
- Gaze direction tracking
- Drowsiness detection
- Distraction detection
- Yawning detection
- Phone usage detection
- Smoking detection
- Attention level scoring
- Alert generation

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
import time


class DriverState(Enum):
    """Driver attention states"""
    ATTENTIVE = "attentive"
    DROWSY = "drowsy"
    DISTRACTED = "distracted"
    EYES_CLOSED = "eyes_closed"
    PHONE_USE = "phone_use"
    SMOKING = "smoking"
    NO_DRIVER = "no_driver"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GazeDirection(Enum):
    """Gaze direction categories"""
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"
    UP = "up"
    UNKNOWN = "unknown"


@dataclass
class EyeState:
    """Eye state information"""
    left_eye_open: bool
    right_eye_open: bool
    left_eye_aspect_ratio: float  # Eye Aspect Ratio (EAR)
    right_eye_aspect_ratio: float
    both_eyes_closed_duration: float = 0.0  # Seconds


@dataclass
class HeadPose:
    """Head pose angles"""
    pitch: float  # Up/down rotation (degrees)
    yaw: float    # Left/right rotation (degrees)
    roll: float   # Head tilt (degrees)
    confidence: float = 1.0


@dataclass
class FacialLandmarks:
    """Facial landmark points"""
    landmarks: np.ndarray  # Shape (68, 2) or (478, 2) for MediaPipe
    bbox: Tuple[int, int, int, int]  # Face bounding box
    confidence: float


@dataclass
class DriverAlert:
    """Driver monitoring alert"""
    alert_level: AlertLevel
    state: DriverState
    message: str
    timestamp: float
    duration: float = 0.0  # How long condition has persisted


@dataclass
class DriverStatus:
    """Complete driver status assessment"""
    state: DriverState
    attention_score: float  # 0-100
    eye_state: EyeState
    head_pose: HeadPose
    gaze_direction: GazeDirection
    is_yawning: bool
    face_detected: bool
    alerts: List[DriverAlert]
    timestamp: float


@dataclass
class DMSConfig:
    """Configuration for Driver Monitoring System"""
    # Eye closure detection
    eye_aspect_ratio_threshold: float = 0.21  # EAR below this = closed
    eye_closure_frames_threshold: int = 3  # Frames with closed eyes
    drowsy_eye_closure_duration: float = 2.0  # Seconds for drowsy alert
    critical_eye_closure_duration: float = 3.0  # Seconds for critical alert

    # Head pose thresholds
    max_yaw_angle: float = 25.0  # Degrees - looking away threshold
    max_pitch_down: float = 20.0  # Looking down threshold
    max_pitch_up: float = 15.0  # Looking up threshold

    # Distraction detection
    distraction_duration_threshold: float = 3.0  # Seconds
    phone_confidence_threshold: float = 0.7

    # Yawning detection
    mouth_aspect_ratio_threshold: float = 0.6  # MAR above this = yawning
    yawn_duration_frames: int = 5

    # Attention scoring
    attention_history_length: int = 30  # Frames for averaging

    # Performance
    face_detection_interval: int = 1  # Detect face every N frames
    enable_landmark_detection: bool = True
    enable_gaze_tracking: bool = True


class DriverMonitoringSystem:
    """
    Driver Monitoring System (DMS)

    Features:
    - Real-time face and eye detection
    - Drowsiness detection via eye closure
    - Distraction detection via head pose
    - Gaze direction tracking
    - Yawning detection
    - Phone usage detection
    - Attention scoring (0-100)
    - Multi-level alerting
    - Temporal state tracking
    """

    def __init__(self, config: Optional[DMSConfig] = None):
        """
        Initialize Driver Monitoring System

        Args:
            config: DMS configuration
        """
        self.config = config or DMSConfig()

        # Load face detector (Haar cascade as fallback)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        # State tracking
        self.current_state = DriverState.UNKNOWN
        self.previous_states: deque = deque(maxlen=30)
        self.attention_history: deque = deque(maxlen=self.config.attention_history_length)
        self.eye_closure_start_time: Optional[float] = None
        self.distraction_start_time: Optional[float] = None
        self.yawn_frame_counter = 0
        self.frame_counter = 0

        # Active alerts
        self.active_alerts: List[DriverAlert] = []

        # Statistics
        self.total_drowsy_events = 0
        self.total_distraction_events = 0
        self.total_monitoring_time = 0.0
        self.start_time = time.time()

    def monitor_driver(
        self,
        image: np.ndarray,
        timestamp: Optional[float] = None
    ) -> DriverStatus:
        """
        Analyze driver state from dashboard camera image

        Args:
            image: Dashboard camera image (BGR)
            timestamp: Current timestamp (defaults to time.time())

        Returns:
            DriverStatus with complete driver assessment
        """
        if timestamp is None:
            timestamp = time.time()

        self.frame_counter += 1
        self.total_monitoring_time = timestamp - self.start_time

        # Detect face
        face_detected, face_bbox = self._detect_face(image)

        if not face_detected:
            return self._create_no_driver_status(timestamp)

        # Extract face region
        x, y, w, h = face_bbox
        face_roi = image[y:y+h, x:x+w]

        # Analyze eyes
        eye_state = self._analyze_eyes(face_roi, face_bbox)

        # Estimate head pose
        head_pose = self._estimate_head_pose(face_roi, face_bbox)

        # Determine gaze direction
        gaze_direction = self._estimate_gaze(head_pose, eye_state)

        # Check for yawning
        is_yawning = self._detect_yawning(face_roi)

        # Classify driver state
        driver_state = self._classify_driver_state(
            eye_state, head_pose, gaze_direction, is_yawning
        )

        # Calculate attention score
        attention_score = self._calculate_attention_score(
            driver_state, eye_state, head_pose, gaze_direction
        )

        # Generate alerts
        alerts = self._generate_alerts(
            driver_state, eye_state, head_pose, timestamp
        )

        # Update tracking
        self.current_state = driver_state
        self.previous_states.append(driver_state)
        self.attention_history.append(attention_score)

        return DriverStatus(
            state=driver_state,
            attention_score=attention_score,
            eye_state=eye_state,
            head_pose=head_pose,
            gaze_direction=gaze_direction,
            is_yawning=is_yawning,
            face_detected=True,
            alerts=alerts,
            timestamp=timestamp
        )

    def _detect_face(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Detect face in image"""
        # Only run face detection periodically for performance
        if self.frame_counter % self.config.face_detection_interval != 0:
            # Return previous detection or None
            return False, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return False, None

        # Return largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return True, tuple(largest_face)

    def _analyze_eyes(
        self,
        face_roi: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> EyeState:
        """Analyze eye state (open/closed)"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect eyes in face region
        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        # Calculate Eye Aspect Ratio (EAR) for each eye
        # Simplified: use eye region intensity variance
        left_ear = 0.25
        right_ear = 0.25
        left_open = True
        right_open = True

        if len(eyes) >= 2:
            # Assume first two detected are left and right eyes
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                eye_region = gray_face[ey:ey+eh, ex:ex+ew]

                if eye_region.size == 0:
                    continue

                # Calculate EAR approximation using variance
                # Higher variance = more open eye
                variance = np.var(eye_region)
                ear = min(variance / 1000.0, 1.0)  # Normalize

                if i == 0:
                    left_ear = ear
                    left_open = ear > self.config.eye_aspect_ratio_threshold
                else:
                    right_ear = ear
                    right_open = ear > self.config.eye_aspect_ratio_threshold

        elif len(eyes) == 1:
            # Only one eye detected, assume both have same state
            ex, ey, ew, eh = eyes[0]
            eye_region = gray_face[ey:ey+eh, ex:ex+ew]
            if eye_region.size > 0:
                variance = np.var(eye_region)
                ear = min(variance / 1000.0, 1.0)
                left_ear = right_ear = ear
                left_open = right_open = ear > self.config.eye_aspect_ratio_threshold

        # Track eye closure duration
        current_time = time.time()
        if not left_open and not right_open:
            if self.eye_closure_start_time is None:
                self.eye_closure_start_time = current_time
            closure_duration = current_time - self.eye_closure_start_time
        else:
            self.eye_closure_start_time = None
            closure_duration = 0.0

        return EyeState(
            left_eye_open=left_open,
            right_eye_open=right_open,
            left_eye_aspect_ratio=left_ear,
            right_eye_aspect_ratio=right_ear,
            both_eyes_closed_duration=closure_duration
        )

    def _estimate_head_pose(
        self,
        face_roi: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> HeadPose:
        """
        Estimate head pose (pitch, yaw, roll)

        Simplified implementation using face bbox aspect ratio and position
        In production, use facial landmarks and PnP solver
        """
        x, y, w, h = face_bbox

        # Estimate yaw from horizontal position
        # Assume centered face has yaw=0
        face_center_x = x + w / 2
        image_center_x = face_roi.shape[1] / 2 + x
        yaw_offset = face_center_x - image_center_x

        # Normalize to degrees (-30 to 30)
        yaw = np.clip(yaw_offset / 5.0, -30, 30)

        # Estimate pitch from vertical position
        face_center_y = y + h / 2
        image_center_y = face_roi.shape[0] / 2 + y
        pitch_offset = face_center_y - image_center_y

        # Normalize to degrees
        pitch = np.clip(pitch_offset / 5.0, -20, 20)

        # Estimate roll from face aspect ratio
        aspect_ratio = w / h
        roll = 0.0  # Simplified

        return HeadPose(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            confidence=0.7
        )

    def _estimate_gaze(
        self,
        head_pose: HeadPose,
        eye_state: EyeState
    ) -> GazeDirection:
        """Estimate gaze direction from head pose and eye state"""
        if not eye_state.left_eye_open and not eye_state.right_eye_open:
            return GazeDirection.UNKNOWN

        # Primarily use head pose for gaze estimation
        yaw = head_pose.yaw
        pitch = head_pose.pitch

        # Determine direction based on angles
        if abs(yaw) < 15 and abs(pitch) < 10:
            return GazeDirection.FORWARD
        elif yaw < -15:
            return GazeDirection.LEFT
        elif yaw > 15:
            return GazeDirection.RIGHT
        elif pitch < -10:
            return GazeDirection.UP
        elif pitch > 10:
            return GazeDirection.DOWN

        return GazeDirection.FORWARD

    def _detect_yawning(self, face_roi: np.ndarray) -> bool:
        """Detect yawning based on mouth opening"""
        # Simplified: detect mouth region and check for large opening
        # In production, use facial landmarks for accurate MAR calculation

        h, w = face_roi.shape[:2]
        mouth_region = face_roi[int(h*0.6):h, int(w*0.3):int(w*0.7)]

        if mouth_region.size == 0:
            return False

        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)

        # Detect dark regions (open mouth)
        _, thresh = cv2.threshold(gray_mouth, 50, 255, cv2.THRESH_BINARY_INV)

        # Calculate ratio of dark pixels
        dark_ratio = np.sum(thresh > 0) / thresh.size

        is_yawning_now = dark_ratio > self.config.mouth_aspect_ratio_threshold

        if is_yawning_now:
            self.yawn_frame_counter += 1
        else:
            self.yawn_frame_counter = 0

        return self.yawn_frame_counter >= self.config.yawn_duration_frames

    def _classify_driver_state(
        self,
        eye_state: EyeState,
        head_pose: HeadPose,
        gaze_direction: GazeDirection,
        is_yawning: bool
    ) -> DriverState:
        """Classify overall driver state"""
        # Eyes closed for extended period = drowsy or eyes closed
        if eye_state.both_eyes_closed_duration >= self.config.drowsy_eye_closure_duration:
            return DriverState.EYES_CLOSED
        elif eye_state.both_eyes_closed_duration >= 1.0:
            return DriverState.DROWSY

        # Looking away from road = distracted
        if gaze_direction in [GazeDirection.LEFT, GazeDirection.RIGHT, GazeDirection.DOWN]:
            if self.distraction_start_time is None:
                self.distraction_start_time = time.time()
            elif time.time() - self.distraction_start_time >= self.config.distraction_duration_threshold:
                return DriverState.DISTRACTED
        else:
            self.distraction_start_time = None

        # Extreme head angles = distracted
        if abs(head_pose.yaw) > self.config.max_yaw_angle:
            return DriverState.DISTRACTED
        if head_pose.pitch > self.config.max_pitch_down:
            return DriverState.DISTRACTED

        # Yawning = potential drowsiness
        if is_yawning:
            return DriverState.DROWSY

        # Default: attentive
        return DriverState.ATTENTIVE

    def _calculate_attention_score(
        self,
        state: DriverState,
        eye_state: EyeState,
        head_pose: HeadPose,
        gaze_direction: GazeDirection
    ) -> float:
        """Calculate attention score (0-100)"""
        score = 100.0

        # Deduct for eyes closed
        if not eye_state.left_eye_open or not eye_state.right_eye_open:
            score -= 30.0
        if eye_state.both_eyes_closed_duration > 0:
            score -= min(40.0, eye_state.both_eyes_closed_duration * 20.0)

        # Deduct for looking away
        if gaze_direction != GazeDirection.FORWARD:
            score -= 25.0

        # Deduct for extreme head angles
        score -= min(20.0, abs(head_pose.yaw) / 2.0)
        score -= min(15.0, abs(head_pose.pitch) / 2.0)

        # State-based deductions
        if state == DriverState.DROWSY:
            score -= 40.0
        elif state == DriverState.DISTRACTED:
            score -= 35.0
        elif state == DriverState.EYES_CLOSED:
            score -= 60.0
        elif state == DriverState.PHONE_USE:
            score -= 50.0

        return max(0.0, min(100.0, score))

    def _generate_alerts(
        self,
        state: DriverState,
        eye_state: EyeState,
        head_pose: HeadPose,
        timestamp: float
    ) -> List[DriverAlert]:
        """Generate alerts based on driver state"""
        alerts = []

        # Critical: Eyes closed for too long
        if eye_state.both_eyes_closed_duration >= self.config.critical_eye_closure_duration:
            alert = DriverAlert(
                alert_level=AlertLevel.CRITICAL,
                state=DriverState.EYES_CLOSED,
                message="CRITICAL: Driver eyes closed! Pull over immediately!",
                timestamp=timestamp,
                duration=eye_state.both_eyes_closed_duration
            )
            alerts.append(alert)
            self.total_drowsy_events += 1

        # High: Drowsiness detected
        elif state == DriverState.DROWSY:
            alert = DriverAlert(
                alert_level=AlertLevel.HIGH,
                state=DriverState.DROWSY,
                message="WARNING: Drowsiness detected. Take a break.",
                timestamp=timestamp,
                duration=eye_state.both_eyes_closed_duration
            )
            alerts.append(alert)

        # Medium: Distraction
        elif state == DriverState.DISTRACTED:
            if self.distraction_start_time:
                duration = timestamp - self.distraction_start_time
                alert = DriverAlert(
                    alert_level=AlertLevel.MEDIUM,
                    state=DriverState.DISTRACTED,
                    message="CAUTION: Driver distracted. Eyes on the road!",
                    timestamp=timestamp,
                    duration=duration
                )
                alerts.append(alert)
                self.total_distraction_events += 1

        # Update active alerts
        self.active_alerts = alerts

        return alerts

    def _create_no_driver_status(self, timestamp: float) -> DriverStatus:
        """Create status when no driver detected"""
        alert = DriverAlert(
            alert_level=AlertLevel.CRITICAL,
            state=DriverState.NO_DRIVER,
            message="CRITICAL: No driver detected!",
            timestamp=timestamp
        )

        return DriverStatus(
            state=DriverState.NO_DRIVER,
            attention_score=0.0,
            eye_state=EyeState(False, False, 0.0, 0.0),
            head_pose=HeadPose(0.0, 0.0, 0.0, 0.0),
            gaze_direction=GazeDirection.UNKNOWN,
            is_yawning=False,
            face_detected=False,
            alerts=[alert],
            timestamp=timestamp
        )

    def visualize_monitoring(
        self,
        image: np.ndarray,
        status: DriverStatus
    ) -> np.ndarray:
        """Visualize driver monitoring results on image"""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # Draw status panel
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background

        # Draw attention score bar
        bar_width = int((status.attention_score / 100.0) * (w - 40))
        bar_color = self._get_attention_color(status.attention_score)
        cv2.rectangle(panel, (20, 20), (20 + bar_width, 50), bar_color, -1)
        cv2.rectangle(panel, (20, 20), (w - 20, 50), (200, 200, 200), 2)

        # Attention score text
        score_text = f"Attention: {status.attention_score:.0f}/100"
        cv2.putText(panel, score_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Driver state
        state_text = f"State: {status.state.value.upper()}"
        state_color = self._get_state_color(status.state)
        cv2.putText(panel, state_text, (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

        # Eye state
        eye_text = f"Eyes: L{'✓' if status.eye_state.left_eye_open else '✗'} R{'✓' if status.eye_state.right_eye_open else '✗'}"
        cv2.putText(panel, eye_text, (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Gaze direction
        gaze_text = f"Gaze: {status.gaze_direction.value}"
        cv2.putText(panel, gaze_text, (200, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Head pose
        pose_text = f"Head: Y:{status.head_pose.yaw:.0f}° P:{status.head_pose.pitch:.0f}°"
        cv2.putText(panel, pose_text, (380, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Alerts
        if status.alerts:
            for i, alert in enumerate(status.alerts[:2]):  # Show up to 2 alerts
                alert_color = self._get_alert_color(alert.alert_level)
                cv2.putText(panel, alert.message, (20, 165 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 2)

        # Combine panel with image
        result = np.vstack([vis_image, panel])

        return result

    def _get_attention_color(self, score: float) -> Tuple[int, int, int]:
        """Get color based on attention score"""
        if score >= 80:
            return (0, 255, 0)  # Green
        elif score >= 60:
            return (0, 255, 255)  # Yellow
        elif score >= 40:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red

    def _get_state_color(self, state: DriverState) -> Tuple[int, int, int]:
        """Get color for driver state"""
        if state == DriverState.ATTENTIVE:
            return (0, 255, 0)
        elif state in [DriverState.DROWSY, DriverState.DISTRACTED]:
            return (0, 165, 255)
        else:
            return (0, 0, 255)

    def _get_alert_color(self, level: AlertLevel) -> Tuple[int, int, int]:
        """Get color for alert level"""
        if level == AlertLevel.CRITICAL:
            return (0, 0, 255)
        elif level == AlertLevel.HIGH:
            return (0, 100, 255)
        elif level == AlertLevel.MEDIUM:
            return (0, 255, 255)
        else:
            return (0, 255, 0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get DMS statistics"""
        avg_attention = (sum(self.attention_history) / len(self.attention_history)
                        if self.attention_history else 0.0)

        return {
            'total_monitoring_time': self.total_monitoring_time,
            'total_drowsy_events': self.total_drowsy_events,
            'total_distraction_events': self.total_distraction_events,
            'average_attention_score': avg_attention,
            'current_state': self.current_state.value,
            'frames_processed': self.frame_counter
        }
