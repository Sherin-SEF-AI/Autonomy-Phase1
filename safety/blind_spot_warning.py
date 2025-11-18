"""
Blind Spot Warning (BSW) system.

Monitors blind spots on both sides of the vehicle and issues warnings
when objects are detected in potentially dangerous positions.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import time

from utils.data_structures import TrackedObject
from utils.logger import get_logger


logger = get_logger()


class BlindSpotRisk(Enum):
    """Blind spot risk levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class BlindSpotSide(Enum):
    """Side of blind spot."""
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass
class BlindSpotZone:
    """Definition of a blind spot zone."""
    # Longitudinal range (vehicle frame)
    x_min: float  # meters behind vehicle center
    x_max: float  # meters ahead of vehicle center

    # Lateral range (vehicle frame)
    y_min: float  # meters from vehicle center
    y_max: float  # meters from vehicle center

    # Risk level for this zone
    risk_level: BlindSpotRisk

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within this zone."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)


@dataclass
class BSWConfig:
    """Configuration for Blind Spot Warning system."""
    # Velocity thresholds
    min_ego_velocity: float = 5.0  # Min velocity for BSW (m/s) ~18 km/h

    # Blind spot zone definitions (relative to vehicle)
    # Typically: slightly behind to slightly ahead, laterally offset

    # Left side zones
    left_zones: List[BlindSpotZone] = None

    # Right side zones
    right_zones: List[BlindSpotZone] = None

    # Object filtering
    relevant_classes: List[str] = None
    confidence_threshold: float = 0.5

    # Warning settings
    warning_cooldown: float = 1.0  # Seconds between warnings
    persistent_warning: bool = True  # Keep warning while object in blind spot

    def __post_init__(self):
        if self.relevant_classes is None:
            self.relevant_classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'
            ]

        if self.left_zones is None:
            # Define left blind spot zones (positive Y = left)
            self.left_zones = [
                # Critical zone: directly beside vehicle
                BlindSpotZone(
                    x_min=-1.0, x_max=1.0,
                    y_min=1.5, y_max=3.0,
                    risk_level=BlindSpotRisk.CRITICAL
                ),
                # High zone: slightly behind and beside
                BlindSpotZone(
                    x_min=-2.5, x_max=-1.0,
                    y_min=1.5, y_max=3.5,
                    risk_level=BlindSpotRisk.HIGH
                ),
                # Medium zone: approaching blind spot
                BlindSpotZone(
                    x_min=-4.0, x_max=-2.5,
                    y_min=2.0, y_max=4.0,
                    risk_level=BlindSpotRisk.MEDIUM
                ),
            ]

        if self.right_zones is None:
            # Define right blind spot zones (negative Y = right)
            self.right_zones = [
                # Critical zone: directly beside vehicle
                BlindSpotZone(
                    x_min=-1.0, x_max=1.0,
                    y_min=-3.0, y_max=-1.5,
                    risk_level=BlindSpotRisk.CRITICAL
                ),
                # High zone: slightly behind and beside
                BlindSpotZone(
                    x_min=-2.5, x_max=-1.0,
                    y_min=-3.5, y_max=-1.5,
                    risk_level=BlindSpotRisk.HIGH
                ),
                # Medium zone: approaching blind spot
                BlindSpotZone(
                    x_min=-4.0, x_max=-2.5,
                    y_min=-4.0, y_max=-2.0,
                    risk_level=BlindSpotRisk.MEDIUM
                ),
            ]


class BlindSpotWarning:
    """
    Blind Spot Warning system.

    Monitors areas beside and slightly behind the vehicle for objects
    that may not be visible to the driver.
    """

    def __init__(self, config: Optional[BSWConfig] = None):
        """
        Initialize BSW system.

        Args:
            config: BSW configuration
        """
        self.config = config or BSWConfig()

        # State tracking
        self.ego_velocity = 0.0  # m/s

        # Warning state
        self.last_warning_time = {
            BlindSpotSide.LEFT: 0.0,
            BlindSpotSide.RIGHT: 0.0
        }
        self.objects_in_blind_spot = {
            BlindSpotSide.LEFT: set(),
            BlindSpotSide.RIGHT: set()
        }

        # Statistics
        self.warnings_issued = 0
        self.left_warnings = 0
        self.right_warnings = 0

        logger.info("Blind Spot Warning system initialized")

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
    ) -> Tuple[BlindSpotRisk, BlindSpotSide, List[TrackedObject]]:
        """
        Analyze tracked objects for blind spot risks.

        Args:
            tracked_objects: List of tracked objects
            ego_velocity: Optional ego velocity (m/s)

        Returns:
            Tuple of (highest_risk_level, affected_sides, objects_at_risk)
        """
        if ego_velocity is not None:
            self.ego_velocity = ego_velocity

        # Check velocity threshold
        if self.ego_velocity < self.config.min_ego_velocity:
            return (BlindSpotRisk.NONE, BlindSpotSide.NONE, [])

        # Analyze left and right blind spots
        left_risk, left_objects = self._analyze_side(tracked_objects, BlindSpotSide.LEFT)
        right_risk, right_objects = self._analyze_side(tracked_objects, BlindSpotSide.RIGHT)

        # Determine overall risk and side
        if left_risk == BlindSpotRisk.NONE and right_risk == BlindSpotRisk.NONE:
            return (BlindSpotRisk.NONE, BlindSpotSide.NONE, [])

        # Get highest risk
        highest_risk = max(left_risk, right_risk, key=lambda x: x.value)

        # Determine affected side
        if left_risk != BlindSpotRisk.NONE and right_risk != BlindSpotRisk.NONE:
            affected_side = BlindSpotSide.BOTH
        elif left_risk != BlindSpotRisk.NONE:
            affected_side = BlindSpotSide.LEFT
        else:
            affected_side = BlindSpotSide.RIGHT

        # Combine objects
        all_risk_objects = left_objects + right_objects

        return (highest_risk, affected_side, all_risk_objects)

    def _analyze_side(
        self,
        tracked_objects: List[TrackedObject],
        side: BlindSpotSide
    ) -> Tuple[BlindSpotRisk, List[TrackedObject]]:
        """
        Analyze one side for blind spot risks.

        Args:
            tracked_objects: All tracked objects
            side: Which side to analyze

        Returns:
            Tuple of (risk_level, objects_at_risk)
        """
        # Get zones for this side
        zones = self.config.left_zones if side == BlindSpotSide.LEFT else self.config.right_zones

        # Track objects currently in blind spot
        current_objects = set()
        risk_objects = []
        highest_risk = BlindSpotRisk.NONE

        for obj in tracked_objects:
            # Filter by class
            if obj.class_name not in self.config.relevant_classes:
                continue

            # Filter by confidence
            if obj.confidence < self.config.confidence_threshold:
                continue

            # Check position
            if obj.current_position is None:
                continue

            x, y, z = obj.current_position

            # Check if object is in any blind spot zone
            for zone in zones:
                if zone.contains(x, y):
                    current_objects.add(obj.track_id)
                    risk_objects.append(obj)
                    highest_risk = max(highest_risk, zone.risk_level, key=lambda x: x.value)
                    break

        # Update tracked objects in blind spot
        previous_objects = self.objects_in_blind_spot[side]
        self.objects_in_blind_spot[side] = current_objects

        # Issue warning if needed
        if highest_risk != BlindSpotRisk.NONE:
            # New objects or persistent warning
            new_objects = current_objects - previous_objects
            if new_objects or self.config.persistent_warning:
                self._issue_warning(highest_risk, side, risk_objects)

        return (highest_risk, risk_objects)

    def _issue_warning(
        self,
        risk_level: BlindSpotRisk,
        side: BlindSpotSide,
        objects: List[TrackedObject]
    ):
        """
        Issue a blind spot warning.

        Args:
            risk_level: Risk level
            side: Affected side
            objects: Objects in blind spot
        """
        current_time = time.time()

        # Rate limit warnings per side
        if current_time - self.last_warning_time[side] < self.config.warning_cooldown:
            return

        self.warnings_issued += 1
        self.last_warning_time[side] = current_time

        # Track warnings by side
        if side == BlindSpotSide.LEFT:
            self.left_warnings += 1
        elif side == BlindSpotSide.RIGHT:
            self.right_warnings += 1

        # Build object description
        object_info = []
        for obj in objects:
            if obj.current_position:
                x, y, _ = obj.current_position
                distance = np.sqrt(x**2 + y**2)
                object_info.append(f"{obj.class_name} at {distance:.1f}m")

        # Log warning
        logger.warning(
            f"[BSW] {risk_level.name} RISK in {side.value.upper()} blind spot: "
            f"{', '.join(object_info)}"
        )

    def get_statistics(self) -> dict:
        """
        Get BSW statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'warnings_issued': self.warnings_issued,
            'left_warnings': self.left_warnings,
            'right_warnings': self.right_warnings,
            'objects_in_left_blind_spot': len(self.objects_in_blind_spot[BlindSpotSide.LEFT]),
            'objects_in_right_blind_spot': len(self.objects_in_blind_spot[BlindSpotSide.RIGHT]),
            'ego_velocity': self.ego_velocity
        }

    def reset(self):
        """Reset BSW state."""
        self.objects_in_blind_spot[BlindSpotSide.LEFT].clear()
        self.objects_in_blind_spot[BlindSpotSide.RIGHT].clear()
        self.warnings_issued = 0
        self.left_warnings = 0
        self.right_warnings = 0
        self.last_warning_time = {
            BlindSpotSide.LEFT: 0.0,
            BlindSpotSide.RIGHT: 0.0
        }
        logger.debug("BSW reset")
