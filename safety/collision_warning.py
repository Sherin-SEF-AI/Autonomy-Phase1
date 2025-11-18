"""
Forward Collision Warning (FCW) system.

Monitors detected objects ahead of the vehicle and issues warnings
when collision risk is detected based on time-to-collision (TTC) calculations.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from utils.data_structures import DetectedObject, TrackedObject
from utils.logger import get_logger


logger = get_logger()


class CollisionRisk(Enum):
    """Collision risk levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FCWConfig:
    """Configuration for Forward Collision Warning system."""
    # TTC thresholds (seconds)
    ttc_critical: float = 1.0  # Critical warning
    ttc_high: float = 2.0  # High risk
    ttc_medium: float = 3.5  # Medium risk
    ttc_low: float = 5.0  # Low risk

    # Distance thresholds (meters)
    min_distance_warning: float = 2.0  # Min distance for any warning
    max_distance_monitor: float = 50.0  # Max distance to monitor

    # Velocity thresholds
    min_ego_velocity: float = 5.0  # Min ego velocity for FCW (m/s) ~18 km/h
    min_closing_velocity: float = 1.0  # Min closing velocity for warning (m/s)

    # Filtering
    smoothing_window: int = 5  # Frames to smooth TTC calculations
    confidence_threshold: float = 0.5  # Min detection confidence

    # Object filtering
    relevant_classes: List[str] = None  # None = all classes

    def __post_init__(self):
        if self.relevant_classes is None:
            # By default, warn for vehicles, pedestrians, cyclists
            self.relevant_classes = [
                'person', 'bicycle', 'car', 'motorcycle',
                'bus', 'truck', 'traffic light', 'stop sign'
            ]


class ForwardCollisionWarning:
    """
    Forward Collision Warning system.

    Analyzes tracked objects in front of the vehicle and calculates
    time-to-collision (TTC) to issue appropriate warnings.
    """

    def __init__(self, config: Optional[FCWConfig] = None):
        """
        Initialize FCW system.

        Args:
            config: FCW configuration
        """
        self.config = config or FCWConfig()

        # State tracking
        self.ego_velocity = 0.0  # m/s
        self.ttc_history = {}  # track_id -> list of TTC values

        # Statistics
        self.warnings_issued = 0
        self.last_warning_time = 0.0

        logger.info("Forward Collision Warning system initialized")

    def set_ego_velocity(self, velocity: float):
        """
        Set the ego vehicle velocity.

        Args:
            velocity: Velocity in m/s
        """
        self.ego_velocity = velocity

    def analyze(
        self,
        tracked_objects: List[TrackedObject],
        ego_velocity: Optional[float] = None
    ) -> Tuple[CollisionRisk, List[TrackedObject]]:
        """
        Analyze tracked objects for collision risk.

        Args:
            tracked_objects: List of tracked objects
            ego_velocity: Optional ego vehicle velocity (m/s)

        Returns:
            Tuple of (highest_risk_level, objects_at_risk)
        """
        if ego_velocity is not None:
            self.ego_velocity = ego_velocity

        # Filter for relevant objects in front
        relevant_objects = self._filter_relevant_objects(tracked_objects)

        if not relevant_objects:
            return (CollisionRisk.NONE, [])

        # Calculate TTC for each object
        risk_objects = []
        highest_risk = CollisionRisk.NONE

        for obj in relevant_objects:
            risk_level, ttc = self._calculate_collision_risk(obj)

            if risk_level != CollisionRisk.NONE:
                risk_objects.append(obj)
                highest_risk = max(highest_risk, risk_level, key=lambda x: x.value)

        # Issue warning if needed
        if highest_risk != CollisionRisk.NONE:
            self._issue_warning(highest_risk, risk_objects)

        return (highest_risk, risk_objects)

    def _filter_relevant_objects(
        self,
        tracked_objects: List[TrackedObject]
    ) -> List[TrackedObject]:
        """
        Filter for objects relevant to FCW.

        Args:
            tracked_objects: All tracked objects

        Returns:
            Filtered list of relevant objects
        """
        relevant = []

        for obj in tracked_objects:
            # Check class
            if obj.class_name not in self.config.relevant_classes:
                continue

            # Check confidence
            if obj.confidence < self.config.confidence_threshold:
                continue

            # Check position (must be in front)
            if obj.current_position is None:
                continue

            x, y, z = obj.current_position

            # Must be ahead of vehicle (positive X in vehicle frame)
            if x <= 0:
                continue

            # Must be within lateral bounds (roughly in lane)
            if abs(y) > 5.0:  # More than 5m laterally
                continue

            # Check distance
            distance = np.sqrt(x**2 + y**2)
            if distance < self.config.min_distance_warning:
                continue
            if distance > self.config.max_distance_monitor:
                continue

            relevant.append(obj)

        return relevant

    def _calculate_collision_risk(
        self,
        obj: TrackedObject
    ) -> Tuple[CollisionRisk, Optional[float]]:
        """
        Calculate collision risk for an object.

        Args:
            obj: Tracked object

        Returns:
            Tuple of (risk_level, ttc_seconds)
        """
        if obj.current_position is None or obj.current_velocity is None:
            return (CollisionRisk.NONE, None)

        # Get object position and velocity
        x, y, z = obj.current_position
        vx, vy = obj.current_velocity

        # Calculate distance
        distance = np.sqrt(x**2 + y**2)

        # Calculate relative velocity (closing velocity)
        # Positive = approaching, negative = moving away
        relative_vx = self.ego_velocity - vx

        # If object is moving away, no collision risk
        if relative_vx <= self.config.min_closing_velocity:
            return (CollisionRisk.NONE, None)

        # Calculate Time-to-Collision (TTC)
        ttc = distance / relative_vx

        # Smooth TTC with history
        if obj.track_id not in self.ttc_history:
            self.ttc_history[obj.track_id] = []

        self.ttc_history[obj.track_id].append(ttc)

        # Keep only recent history
        if len(self.ttc_history[obj.track_id]) > self.config.smoothing_window:
            self.ttc_history[obj.track_id].pop(0)

        # Calculate smoothed TTC
        smoothed_ttc = np.mean(self.ttc_history[obj.track_id])

        # Determine risk level based on TTC
        if smoothed_ttc <= self.config.ttc_critical:
            risk = CollisionRisk.CRITICAL
        elif smoothed_ttc <= self.config.ttc_high:
            risk = CollisionRisk.HIGH
        elif smoothed_ttc <= self.config.ttc_medium:
            risk = CollisionRisk.MEDIUM
        elif smoothed_ttc <= self.config.ttc_low:
            risk = CollisionRisk.LOW
        else:
            risk = CollisionRisk.NONE

        return (risk, smoothed_ttc)

    def _issue_warning(self, risk_level: CollisionRisk, objects: List[TrackedObject]):
        """
        Issue a collision warning.

        Args:
            risk_level: Collision risk level
            objects: Objects at risk
        """
        current_time = time.time()

        # Rate limit warnings (don't spam)
        if current_time - self.last_warning_time < 0.5:  # 500ms cooldown
            return

        self.warnings_issued += 1
        self.last_warning_time = current_time

        # Log warning
        object_info = []
        for obj in objects:
            if obj.current_position:
                x, y, _ = obj.current_position
                distance = np.sqrt(x**2 + y**2)
                object_info.append(f"{obj.class_name} at {distance:.1f}m")

        logger.warning(
            f"[FCW] {risk_level.name} RISK: {', '.join(object_info)}"
        )

    def get_statistics(self) -> dict:
        """
        Get FCW statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'warnings_issued': self.warnings_issued,
            'ego_velocity': self.ego_velocity,
            'tracked_objects': len(self.ttc_history)
        }

    def reset(self):
        """Reset FCW state."""
        self.ttc_history.clear()
        self.warnings_issued = 0
        self.last_warning_time = 0.0
        logger.debug("FCW reset")
