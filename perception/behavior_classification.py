"""
Object behavior classification for understanding object intentions and actions.

Classifies object behaviors such as:
- Motion state (stationary, moving, stopping, starting)
- Turning behavior (straight, turning left/right)
- Speed changes (accelerating, decelerating, constant)
- Maneuvers (lane change, parking, reversing)
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from collections import deque

from utils.data_structures import TrackedObject
from utils.logger import get_logger


logger = get_logger()


class MotionState(Enum):
    """Object motion state."""
    STATIONARY = "stationary"
    MOVING = "moving"
    STOPPING = "stopping"
    STARTING = "starting"
    UNKNOWN = "unknown"


class TurningBehavior(Enum):
    """Object turning behavior."""
    STRAIGHT = "straight"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    UNKNOWN = "unknown"


class SpeedChange(Enum):
    """Object speed change behavior."""
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    CONSTANT_SPEED = "constant"
    UNKNOWN = "unknown"


class ManeuverType(Enum):
    """Detected maneuver types."""
    NORMAL_DRIVING = "normal_driving"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    PARKING = "parking"
    REVERSING = "reversing"
    U_TURN = "u_turn"
    UNKNOWN = "unknown"


@dataclass
class ObjectBehavior:
    """Classified behavior for a tracked object."""
    track_id: int
    motion_state: MotionState
    turning_behavior: TurningBehavior
    speed_change: SpeedChange
    maneuver_type: ManeuverType
    confidence: float  # Overall classification confidence (0-1)

    # Additional metrics
    current_speed: Optional[float] = None  # m/s
    acceleration: Optional[float] = None  # m/s²
    turn_rate: Optional[float] = None  # degrees/s
    lateral_movement: Optional[float] = None  # m/s (perpendicular to forward direction)


class BehaviorClassifier:
    """
    Classifies object behavior based on trajectory and motion history.

    Uses motion patterns, velocity changes, and position history to
    determine what an object is doing.
    """

    def __init__(
        self,
        stationary_speed_threshold: float = 0.3,  # m/s
        min_turn_rate: float = 5.0,  # degrees/s
        min_acceleration: float = 0.5,  # m/s²
        history_window: int = 30  # frames
    ):
        """
        Initialize behavior classifier.

        Args:
            stationary_speed_threshold: Speed below which object is considered stationary
            min_turn_rate: Minimum turn rate to classify as turning (degrees/s)
            min_acceleration: Minimum acceleration to classify as accelerating/decelerating
            history_window: Number of frames to keep in history
        """
        self.stationary_threshold = stationary_speed_threshold
        self.min_turn_rate = min_turn_rate
        self.min_acceleration = min_acceleration
        self.history_window = history_window

        # Behavior history per track
        self.behavior_history: Dict[int, deque] = {}

        logger.info("Behavior classifier initialized")

    def classify(self, tracked_object: TrackedObject) -> Optional[ObjectBehavior]:
        """
        Classify behavior for a tracked object.

        Args:
            tracked_object: Tracked object to classify

        Returns:
            ObjectBehavior if classification successful, None otherwise
        """
        track_id = tracked_object.track_id

        # Initialize history if needed
        if track_id not in self.behavior_history:
            self.behavior_history[track_id] = deque(maxlen=self.history_window)

        # Need sufficient history for classification
        if len(tracked_object.trajectory) < 3:
            return None

        # Calculate motion metrics
        current_speed = self._calculate_speed(tracked_object)
        acceleration = self._calculate_acceleration(tracked_object)
        turn_rate = self._calculate_turn_rate(tracked_object)
        lateral_movement = self._calculate_lateral_movement(tracked_object)

        # Classify motion state
        motion_state = self._classify_motion_state(
            current_speed, acceleration, tracked_object
        )

        # Classify turning behavior
        turning_behavior = self._classify_turning(turn_rate)

        # Classify speed change
        speed_change = self._classify_speed_change(acceleration)

        # Classify maneuver type
        maneuver_type = self._classify_maneuver(
            motion_state, turning_behavior, lateral_movement, tracked_object
        )

        # Calculate overall confidence
        confidence = self._calculate_confidence(tracked_object)

        behavior = ObjectBehavior(
            track_id=track_id,
            motion_state=motion_state,
            turning_behavior=turning_behavior,
            speed_change=speed_change,
            maneuver_type=maneuver_type,
            confidence=confidence,
            current_speed=current_speed,
            acceleration=acceleration,
            turn_rate=turn_rate,
            lateral_movement=lateral_movement
        )

        # Store in history
        self.behavior_history[track_id].append(behavior)

        return behavior

    def classify_all(
        self,
        tracked_objects: List[TrackedObject]
    ) -> Dict[int, ObjectBehavior]:
        """
        Classify behavior for all tracked objects.

        Args:
            tracked_objects: List of tracked objects

        Returns:
            Dictionary mapping track_id to ObjectBehavior
        """
        behaviors = {}
        for obj in tracked_objects:
            behavior = self.classify(obj)
            if behavior:
                behaviors[obj.track_id] = behavior

        return behaviors

    def _calculate_speed(self, obj: TrackedObject) -> Optional[float]:
        """Calculate current speed from velocity."""
        if obj.current_velocity:
            vx, vy = obj.current_velocity
            return np.sqrt(vx**2 + vy**2)
        return None

    def _calculate_acceleration(self, obj: TrackedObject) -> Optional[float]:
        """Calculate acceleration from trajectory history."""
        if len(obj.trajectory) < 3:
            return None

        try:
            # Get last 3 positions
            recent = obj.trajectory[-3:]
            times = [t for t, _ in recent]
            positions = np.array([pos for _, pos in recent])

            # Calculate velocities
            dt1 = times[1] - times[0]
            dt2 = times[2] - times[1]

            if dt1 <= 0 or dt2 <= 0:
                return None

            vel1 = np.linalg.norm(positions[1] - positions[0]) / dt1
            vel2 = np.linalg.norm(positions[2] - positions[1]) / dt2

            # Calculate acceleration
            acceleration = (vel2 - vel1) / ((dt1 + dt2) / 2)

            return acceleration

        except Exception as e:
            logger.debug(f"Acceleration calculation failed: {e}")
            return None

    def _calculate_turn_rate(self, obj: TrackedObject) -> Optional[float]:
        """Calculate turn rate in degrees per second."""
        if len(obj.trajectory) < 3:
            return None

        try:
            # Get last 3 positions
            recent = obj.trajectory[-3:]
            times = [t for t, _ in recent]
            positions = np.array([pos for _, pos in recent])

            # Calculate heading angles
            vec1 = positions[1] - positions[0]
            vec2 = positions[2] - positions[1]

            # Only calculate if movement is significant
            if np.linalg.norm(vec1) < 0.1 or np.linalg.norm(vec2) < 0.1:
                return None

            angle1 = np.arctan2(vec1[1], vec1[0])
            angle2 = np.arctan2(vec2[1], vec2[0])

            # Calculate angular change
            angle_diff = angle2 - angle1

            # Normalize to [-π, π]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            # Convert to degrees and calculate rate
            dt = times[2] - times[0]
            if dt <= 0:
                return None

            turn_rate = np.degrees(angle_diff) / dt

            return turn_rate

        except Exception as e:
            logger.debug(f"Turn rate calculation failed: {e}")
            return None

    def _calculate_lateral_movement(self, obj: TrackedObject) -> Optional[float]:
        """Calculate lateral (perpendicular) movement rate."""
        if len(obj.trajectory) < 3:
            return None

        try:
            recent = obj.trajectory[-3:]
            positions = np.array([pos for _, pos in recent])

            # Calculate forward direction (average heading)
            forward_vec = positions[-1] - positions[0]
            if np.linalg.norm(forward_vec) < 0.1:
                return None

            # Normalize forward direction
            forward_unit = forward_vec[:2] / np.linalg.norm(forward_vec[:2])

            # Perpendicular direction (rotate 90 degrees)
            perp_unit = np.array([-forward_unit[1], forward_unit[0]])

            # Calculate lateral displacement
            lateral_vec = positions[-1][:2] - positions[-2][:2]
            lateral_speed = np.dot(lateral_vec, perp_unit)

            # Normalize by time
            times = [t for t, _ in recent]
            dt = times[-1] - times[-2]
            if dt <= 0:
                return None

            lateral_speed /= dt

            return abs(lateral_speed)

        except Exception as e:
            logger.debug(f"Lateral movement calculation failed: {e}")
            return None

    def _classify_motion_state(
        self,
        speed: Optional[float],
        acceleration: Optional[float],
        obj: TrackedObject
    ) -> MotionState:
        """Classify motion state."""
        if speed is None:
            return MotionState.UNKNOWN

        # Stationary
        if speed < self.stationary_threshold:
            return MotionState.STATIONARY

        # Stopping (moving but decelerating significantly)
        if acceleration is not None and acceleration < -self.min_acceleration:
            if speed < 2.0:  # Low speed and decelerating
                return MotionState.STOPPING

        # Starting (low speed but accelerating)
        if acceleration is not None and acceleration > self.min_acceleration:
            if speed < 2.0:  # Low speed and accelerating
                return MotionState.STARTING

        # Moving
        if speed >= self.stationary_threshold:
            return MotionState.MOVING

        return MotionState.UNKNOWN

    def _classify_turning(self, turn_rate: Optional[float]) -> TurningBehavior:
        """Classify turning behavior."""
        if turn_rate is None:
            return TurningBehavior.UNKNOWN

        if abs(turn_rate) < self.min_turn_rate:
            return TurningBehavior.STRAIGHT

        if turn_rate > self.min_turn_rate:
            return TurningBehavior.TURNING_LEFT

        if turn_rate < -self.min_turn_rate:
            return TurningBehavior.TURNING_RIGHT

        return TurningBehavior.UNKNOWN

    def _classify_speed_change(self, acceleration: Optional[float]) -> SpeedChange:
        """Classify speed change behavior."""
        if acceleration is None:
            return SpeedChange.UNKNOWN

        if acceleration > self.min_acceleration:
            return SpeedChange.ACCELERATING

        if acceleration < -self.min_acceleration:
            return SpeedChange.DECELERATING

        return SpeedChange.CONSTANT_SPEED

    def _classify_maneuver(
        self,
        motion_state: MotionState,
        turning_behavior: TurningBehavior,
        lateral_movement: Optional[float],
        obj: TrackedObject
    ) -> ManeuverType:
        """Classify maneuver type based on combined behaviors."""
        # Check for parking (very slow lateral movement)
        if motion_state == MotionState.STOPPING and lateral_movement and lateral_movement > 0.5:
            return ManeuverType.PARKING

        # Check for lane change (significant lateral movement while moving straight-ish)
        if (motion_state == MotionState.MOVING and
            lateral_movement and lateral_movement > 1.0):

            if turning_behavior == TurningBehavior.TURNING_LEFT:
                return ManeuverType.LANE_CHANGE_LEFT
            elif turning_behavior == TurningBehavior.TURNING_RIGHT:
                return ManeuverType.LANE_CHANGE_RIGHT

        # Check for U-turn (sharp turn)
        if turning_behavior in [TurningBehavior.TURNING_LEFT, TurningBehavior.TURNING_RIGHT]:
            track_id = obj.track_id
            if track_id in self.behavior_history and len(self.behavior_history[track_id]) > 10:
                # If been turning for extended period, might be U-turn
                recent_behaviors = list(self.behavior_history[track_id])[-10:]
                turning_count = sum(
                    1 for b in recent_behaviors
                    if b.turning_behavior in [TurningBehavior.TURNING_LEFT, TurningBehavior.TURNING_RIGHT]
                )
                if turning_count >= 8:  # Turning for most of recent history
                    return ManeuverType.U_TURN

        # Normal driving
        if motion_state == MotionState.MOVING and turning_behavior in [
            TurningBehavior.STRAIGHT, TurningBehavior.TURNING_LEFT, TurningBehavior.TURNING_RIGHT
        ]:
            return ManeuverType.NORMAL_DRIVING

        return ManeuverType.UNKNOWN

    def _calculate_confidence(self, obj: TrackedObject) -> float:
        """Calculate classification confidence based on data quality."""
        confidence = 0.5  # Base confidence

        # More trajectory history = higher confidence
        trajectory_length = len(obj.trajectory)
        if trajectory_length >= 20:
            confidence += 0.3
        elif trajectory_length >= 10:
            confidence += 0.2
        elif trajectory_length >= 5:
            confidence += 0.1

        # Longer tracking duration = higher confidence
        tracking_duration = obj.last_seen - obj.first_seen
        if tracking_duration >= 3.0:  # 3+ seconds
            confidence += 0.2
        elif tracking_duration >= 1.0:  # 1+ second
            confidence += 0.1

        return min(confidence, 1.0)

    def get_behavior_history(self, track_id: int) -> List[ObjectBehavior]:
        """
        Get behavior history for a track.

        Args:
            track_id: Track ID

        Returns:
            List of ObjectBehavior in chronological order
        """
        if track_id in self.behavior_history:
            return list(self.behavior_history[track_id])
        return []

    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Remove behavior history for tracks that no longer exist.

        Args:
            active_track_ids: List of currently active track IDs
        """
        active_set = set(active_track_ids)
        to_remove = [tid for tid in self.behavior_history.keys() if tid not in active_set]

        for track_id in to_remove:
            del self.behavior_history[track_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old behavior histories")
