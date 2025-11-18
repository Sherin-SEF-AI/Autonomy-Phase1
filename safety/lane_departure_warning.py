"""
Lane Departure Warning (LDW) system.

Monitors lane position and issues warnings when the vehicle
departs from its lane without signaling.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from utils.data_structures import LaneInfo
from utils.logger import get_logger


logger = get_logger()


class DepartureRisk(Enum):
    """Lane departure risk levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class DepartureSide(Enum):
    """Side of lane departure."""
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class LDWConfig:
    """Configuration for Lane Departure Warning system."""
    # Lateral offset thresholds (meters from lane center)
    offset_low: float = 0.3  # 30cm from center
    offset_medium: float = 0.5  # 50cm from center
    offset_high: float = 0.7  # 70cm from center
    offset_critical: float = 0.9  # 90cm from center (near lane edge)

    # Time-to-lane-crossing thresholds (seconds)
    ttlc_critical: float = 0.5
    ttlc_high: float = 1.0
    ttlc_medium: float = 1.5
    ttlc_low: float = 2.0

    # Velocity thresholds
    min_velocity: float = 10.0  # Min velocity for LDW (m/s) ~36 km/h

    # Lane quality requirements
    min_lane_confidence: float = 0.6  # Min confidence for reliable LDW
    require_both_lanes: bool = False  # Require both lanes detected

    # Filtering
    smoothing_window: int = 5  # Frames for smoothing
    warning_cooldown: float = 2.0  # Seconds between warnings


class LaneDepartureWarning:
    """
    Lane Departure Warning system.

    Monitors the vehicle's position within the lane and issues warnings
    when departure is detected or imminent.
    """

    def __init__(self, config: Optional[LDWConfig] = None):
        """
        Initialize LDW system.

        Args:
            config: LDW configuration
        """
        self.config = config or LDWConfig()

        # State tracking
        self.ego_velocity = 0.0  # m/s
        self.lateral_velocity = 0.0  # m/s (positive = moving left)
        self.offset_history = []  # History of lateral offsets

        # Warning state
        self.last_warning_time = 0.0
        self.last_warning_side = DepartureSide.NONE

        # Statistics
        self.warnings_issued = 0
        self.left_departures = 0
        self.right_departures = 0

        logger.info("Lane Departure Warning system initialized")

    def set_ego_velocity(self, velocity: float, lateral_velocity: float = 0.0):
        """
        Set the ego vehicle velocity.

        Args:
            velocity: Longitudinal velocity in m/s
            lateral_velocity: Lateral velocity in m/s (positive = left)
        """
        self.ego_velocity = velocity
        self.lateral_velocity = lateral_velocity

    def analyze(
        self,
        lane_info: Optional[LaneInfo],
        ego_velocity: Optional[float] = None,
        lateral_velocity: Optional[float] = None
    ) -> Tuple[DepartureRisk, DepartureSide]:
        """
        Analyze lane position for departure risk.

        Args:
            lane_info: Lane detection information
            ego_velocity: Optional ego velocity (m/s)
            lateral_velocity: Optional lateral velocity (m/s)

        Returns:
            Tuple of (risk_level, departure_side)
        """
        if ego_velocity is not None:
            self.ego_velocity = ego_velocity
        if lateral_velocity is not None:
            self.lateral_velocity = lateral_velocity

        # Check if we have valid lane info
        if lane_info is None or lane_info.confidence < self.config.min_lane_confidence:
            return (DepartureRisk.NONE, DepartureSide.NONE)

        # Check if both lanes are required
        if self.config.require_both_lanes:
            if not (lane_info.left_lane_detected and lane_info.right_lane_detected):
                return (DepartureRisk.NONE, DepartureSide.NONE)

        # Check velocity threshold
        if self.ego_velocity < self.config.min_velocity:
            return (DepartureRisk.NONE, DepartureSide.NONE)

        # Analyze lateral offset
        risk_level, departure_side = self._analyze_lateral_offset(lane_info)

        # Calculate Time-to-Lane-Crossing if moving laterally
        if abs(self.lateral_velocity) > 0.1:  # Moving laterally
            ttlc_risk, ttlc_side = self._calculate_ttlc(lane_info)
            # Use the higher risk
            if ttlc_risk.value > risk_level.value:
                risk_level = ttlc_risk
                departure_side = ttlc_side

        # Issue warning if needed
        if risk_level != DepartureRisk.NONE:
            self._issue_warning(risk_level, departure_side, lane_info)

        return (risk_level, departure_side)

    def _analyze_lateral_offset(
        self,
        lane_info: LaneInfo
    ) -> Tuple[DepartureRisk, DepartureSide]:
        """
        Analyze lateral offset from lane center.

        Args:
            lane_info: Lane detection information

        Returns:
            Tuple of (risk_level, departure_side)
        """
        offset = lane_info.lateral_offset

        # Add to history and smooth
        self.offset_history.append(offset)
        if len(self.offset_history) > self.config.smoothing_window:
            self.offset_history.pop(0)

        smoothed_offset = np.mean(self.offset_history)

        # Determine departure side
        if abs(smoothed_offset) < self.config.offset_low:
            return (DepartureRisk.NONE, DepartureSide.NONE)

        departure_side = DepartureSide.LEFT if smoothed_offset > 0 else DepartureSide.RIGHT
        abs_offset = abs(smoothed_offset)

        # Determine risk level based on offset
        if abs_offset >= self.config.offset_critical:
            risk = DepartureRisk.CRITICAL
        elif abs_offset >= self.config.offset_high:
            risk = DepartureRisk.HIGH
        elif abs_offset >= self.config.offset_medium:
            risk = DepartureRisk.MEDIUM
        elif abs_offset >= self.config.offset_low:
            risk = DepartureRisk.LOW
        else:
            risk = DepartureRisk.NONE

        return (risk, departure_side)

    def _calculate_ttlc(
        self,
        lane_info: LaneInfo
    ) -> Tuple[DepartureRisk, DepartureSide]:
        """
        Calculate Time-to-Lane-Crossing (TTLC).

        Args:
            lane_info: Lane detection information

        Returns:
            Tuple of (risk_level, departure_side)
        """
        if abs(self.lateral_velocity) < 0.1:  # Not moving laterally
            return (DepartureRisk.NONE, DepartureSide.NONE)

        # Estimate distance to lane edge
        lane_width = lane_info.lane_width if lane_info.lane_width else 3.5  # Standard lane width
        offset = lane_info.lateral_offset

        # Distance to lane edge depends on direction of travel
        if self.lateral_velocity > 0:  # Moving left
            distance_to_edge = (lane_width / 2) - offset
            departure_side = DepartureSide.LEFT
        else:  # Moving right
            distance_to_edge = (lane_width / 2) + offset
            departure_side = DepartureSide.RIGHT

        # If already past lane edge, critical
        if distance_to_edge <= 0:
            return (DepartureRisk.CRITICAL, departure_side)

        # Calculate TTLC
        ttlc = distance_to_edge / abs(self.lateral_velocity)

        # Determine risk level based on TTLC
        if ttlc <= self.config.ttlc_critical:
            risk = DepartureRisk.CRITICAL
        elif ttlc <= self.config.ttlc_high:
            risk = DepartureRisk.HIGH
        elif ttlc <= self.config.ttlc_medium:
            risk = DepartureRisk.MEDIUM
        elif ttlc <= self.config.ttlc_low:
            risk = DepartureRisk.LOW
        else:
            risk = DepartureRisk.NONE

        return (risk, departure_side)

    def _issue_warning(
        self,
        risk_level: DepartureRisk,
        departure_side: DepartureSide,
        lane_info: LaneInfo
    ):
        """
        Issue a lane departure warning.

        Args:
            risk_level: Departure risk level
            departure_side: Side of departure
            lane_info: Lane detection information
        """
        current_time = time.time()

        # Rate limit warnings
        if current_time - self.last_warning_time < self.config.warning_cooldown:
            return

        # Don't repeat same warning too quickly
        if departure_side == self.last_warning_side:
            if current_time - self.last_warning_time < self.config.warning_cooldown * 2:
                return

        self.warnings_issued += 1
        self.last_warning_time = current_time
        self.last_warning_side = departure_side

        # Track departure side
        if departure_side == DepartureSide.LEFT:
            self.left_departures += 1
        elif departure_side == DepartureSide.RIGHT:
            self.right_departures += 1

        # Log warning
        logger.warning(
            f"[LDW] {risk_level.name} RISK: {departure_side.value.upper()} departure "
            f"(offset: {lane_info.lateral_offset:.2f}m)"
        )

    def get_statistics(self) -> dict:
        """
        Get LDW statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'warnings_issued': self.warnings_issued,
            'left_departures': self.left_departures,
            'right_departures': self.right_departures,
            'ego_velocity': self.ego_velocity,
            'lateral_velocity': self.lateral_velocity
        }

    def reset(self):
        """Reset LDW state."""
        self.offset_history.clear()
        self.warnings_issued = 0
        self.left_departures = 0
        self.right_departures = 0
        self.last_warning_time = 0.0
        self.last_warning_side = DepartureSide.NONE
        logger.debug("LDW reset")
